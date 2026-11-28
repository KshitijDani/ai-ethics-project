"""Shared validation, metrics, and output pipeline for multi-turn analysis."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import re
import sys
import tempfile

import numpy as np
import pandas as pd

from analysis.embeddings import normalize_vectors
from analysis import embeddings
from analysis.csv_embeddings import persist_embeddings

ROOT = Path(__file__).resolve().parents[1]
MODES = ('baseline', 'prompt-prepend', 'prompt-prepend-once')
COLUMNS = ('model', 'mode', 'persona', 'question_file_name', 'question_number',
           'run_id', 'run_number', 'turn', 'question', 'user_message', 'response', 'timestamp')
QUESTION_KEY = ['model', 'question_file_name', 'question_number', 'turn', 'question']
IDENTITY = ['model', 'mode', 'persona', 'question_file_name', 'run_number', 'turn']


def discover_inputs(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f'Input root is not a directory: {root}')
    paths = sorted(root.rglob('*.csv'))
    if not paths:
        raise ValueError(f'No response CSVs found beneath {root}')
    return paths


def load_responses(paths):
    paths = [Path(path).resolve() for path in paths]
    if not paths:
        raise ValueError('Supply at least one response CSV.')
    if len(set(paths)) != len(paths):
        raise ValueError('Duplicate input CSV paths.')
    frames = []
    for path in paths:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, engine='python')
        missing = set(COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f'{path}: missing response columns {sorted(missing)}')
        if frame.empty:
            raise ValueError(f'{path}: no response rows.')
        embedding_columns = [column for column in ('embedding_vector', 'embedding_metadata')
                             if column in frame]
        frame = frame[list(COLUMNS) + embedding_columns].copy()
        for column in ('embedding_vector', 'embedding_metadata'):
            if column not in frame:
                frame[column] = ''
        frame['source_file'] = str(path)
        frame['source_row'] = np.arange(1, len(frame) + 1)
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    for column in COLUMNS:
        if frame[column].str.strip().eq('').any():
            raise ValueError(f'Input contains blank {column} values.')
    for column in ('turn', 'run_number'):
        values = pd.to_numeric(frame[column], errors='coerce')
        if not (np.isfinite(values) & values.gt(0) & values.eq(np.floor(values))).all():
            raise ValueError(f'{column} must contain positive integers.')
        frame[column] = values.astype(int)
    if not frame['mode'].isin(MODES).all():
        raise ValueError('Unknown mode; expected baseline, prompt-prepend, or prompt-prepend-once.')
    for column in ('model', 'persona'):
        if not frame[column].str.fullmatch(r'[A-Za-z0-9_-]+').all():
            raise ValueError(f'{column} must use filename-safe names (letters, digits, _ or -).')
    if not frame.run_id.str.startswith('multi-turn/').all():
        raise ValueError('Only multi-turn run IDs are supported; do not mix single-turn results.')
    if (frame['mode'].eq('baseline') != frame.persona.eq('baseline')).any():
        raise ValueError('Only baseline rows must have persona=baseline.')
    if frame.duplicated(IDENTITY).any():
        raise ValueError('Duplicate response identities; choose one experiment per condition.')
    # Turn and question ID must each map to one exact question within a model/topic.
    for keys, targets in [(['model', 'question_file_name', 'turn'], ['question_number', 'question']),
                          (['model', 'question_file_name', 'question_number'], ['turn', 'question'])]:
        if frame.groupby(keys)[targets].nunique().gt(1).any().any():
            raise ValueError('Question IDs, turns, or exact question text do not align across conditions.')
    run_keys = ['model', 'mode', 'persona', 'question_file_name', 'run_number']
    if frame.groupby(run_keys).run_id.nunique().gt(1).any():
        raise ValueError('A repetition has inconsistent run_id values.')
    if frame.groupby('run_id')[run_keys].nunique().gt(1).any().any():
        raise ValueError('A run_id identifies multiple conversations.')
    baseline = frame[frame['mode'] == 'baseline']
    if baseline.empty:
        raise ValueError('No baseline responses were supplied.')
    personalized = frame[frame['mode'] != 'baseline']
    if personalized.empty:
        raise ValueError('No personalized responses were supplied.')
    baseline_keys = set(baseline[QUESTION_KEY].itertuples(index=False, name=None))
    for key in personalized[QUESTION_KEY].itertuples(index=False, name=None):
        if key not in baseline_keys:
            raise ValueError(f'Missing baseline for question: {key}')
    return frame


def coverage_table(frame):
    rows = []
    keys = ['model', 'mode', 'persona', 'question_file_name']
    for key, group in frame.groupby(keys, sort=True):
        model, _, _, topic = key
        topic_rows = frame[(frame.model == model) & (frame.question_file_name == topic)]
        # Expose numeric gaps as incomplete coverage, including gaps in all conditions.
        expected = set(range(1, int(topic_rows.turn.max()) + 1))
        complete = sum(set(run.turn) == expected for _, run in group.groupby('run_number'))
        rows.append(dict(zip(keys, key)) | dict(
            response_count=len(group), run_count=group.run_number.nunique(),
            complete_run_count=complete, expected_turn_count=len(expected),
            observed_turn_count=group.turn.nunique(),
            run_numbers=','.join(map(str, sorted(group.run_number.unique()))),
            missing_turns=','.join(map(str, sorted(expected - set(group.turn)))),
        ))
    return pd.DataFrame(rows)


def descriptive(values):
    values = np.asarray(values, dtype=float)
    return dict(mean_similarity=float(values.mean()) if values.size else np.nan,
                std_similarity=float(values.std(ddof=1)) if values.size > 1 else np.nan)


def pair_consistency(vectors):
    matrix = vectors @ vectors.T
    pairs = np.clip(matrix[np.triu_indices(len(vectors), k=1)], -1, 1)
    return dict(pair_count=len(pairs)) | descriptive(pairs)


def score_responses(frame, vectors, method):
    if method not in ('average', 'pairing'):
        raise ValueError('Method must be average or pairing.')
    vectors = normalize_vectors(vectors)
    if len(frame) != len(vectors):
        raise ValueError('One vector is required per response row.')
    frame = frame.reset_index(drop=True)
    raw_rows, summaries, consistencies = [], [], []
    for key, question in frame.groupby(QUESTION_KEY, sort=True):
        context = dict(zip(QUESTION_KEY, key))
        baseline = question[question['mode'] == 'baseline'].sort_values('run_number')
        baseline_vectors = vectors[baseline.index]
        mean_baseline = baseline_vectors.mean(axis=0)
        concentration = float(np.linalg.norm(mean_baseline))
        if method == 'average' and concentration < 1e-12:
            raise ValueError(f'Baseline centroid has near-zero norm for {context}.')
        baseline_consistency = pair_consistency(baseline_vectors)
        for (mode, persona), group in question.groupby(['mode', 'persona'], sort=True):
            group = group.sort_values('run_number')
            group_vectors = vectors[group.index]
            consistency = pair_consistency(group_vectors)
            common = context | dict(mode=mode, persona=persona, run_count=len(group))
            consistencies.append(common | consistency)
            if mode == 'baseline':
                continue
            if method == 'average':
                scores = np.clip(group_vectors @ (mean_baseline / concentration), -1, 1)
            else:
                scores = np.clip(group_vectors @ baseline_vectors.T, -1, 1)
            summaries.append(common | dict(
                method=method, baseline_run_count=len(baseline), score_count=int(scores.size),
                baseline_concentration=concentration,
                baseline_consistency=baseline_consistency['mean_similarity'],
                persona_consistency=consistency['mean_similarity'],
            ) | descriptive(scores.ravel()))
            for position, (_, row) in enumerate(group.iterrows()):
                provenance = dict(persona_run_id=row.run_id, persona_run_number=row.run_number,
                                  persona_source_file=row.source_file, persona_source_row=row.source_row)
                if method == 'average':
                    raw_rows.append(context | dict(mode=mode, persona=persona, method=method,
                                    similarity=float(scores[position]), baseline_run_id='',
                                    baseline_run_number=np.nan, baseline_source_file='',
                                    baseline_source_row=np.nan) | provenance)
                else:
                    for baseline_position, (_, reference) in enumerate(baseline.iterrows()):
                        raw_rows.append(context | dict(
                            mode=mode, persona=persona, method=method,
                            similarity=float(scores[position, baseline_position]),
                            baseline_run_id=reference.run_id, baseline_run_number=reference.run_number,
                            baseline_source_file=reference.source_file,
                            baseline_source_row=reference.source_row,
                        ) | provenance)
    return pd.DataFrame(raw_rows), pd.DataFrame(summaries), pd.DataFrame(consistencies)


def plot_summaries(summary, output, consistency):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    palette = {'American': '#2878b5', 'Chinese': '#d55e00', 'French': '#009e73'}
    for index, persona in enumerate(sorted(summary.persona.unique())):
        palette.setdefault(persona, plt.get_cmap('tab20')(index % 20))
    plots = []
    for (model, filename, mode), group in summary.groupby(['model', 'question_file_name', 'mode']):
        topic = Path(filename).stem.removeprefix('questions_')
        baseline = consistency[(consistency.model == model)
                               & (consistency.question_file_name == filename)
                               & (consistency['mode'] == 'baseline')].set_index('turn').sort_index()
        last_turn = max(int(group.turn.max()), int(baseline.index.max()))
        fig, axis = plt.subplots(figsize=(10, 6), layout='constrained')
        for persona, curve in group.groupby('persona'):
            curve = curve.sort_values('turn')
            # Include absent integer turns as NaN so a line does not bridge missing data.
            curve = curve.set_index('turn').reindex(range(1, last_turn + 1))
            x = curve.index.to_numpy()
            y = curve.mean_similarity.to_numpy()
            sd = curve.std_similarity.to_numpy()
            color = palette[persona]
            axis.plot(x, y, marker='o', linewidth=1.8, markersize=4,
                      color=color, label=persona)
            axis.fill_between(x, np.clip(y - sd, -1, 1), np.clip(y + sd, -1, 1),
                              color=color, alpha=0.13, linewidth=0)
        baseline = baseline.reindex(range(1, last_turn + 1))
        axis.plot(baseline.index, baseline.mean_similarity, '--', color='#333333',
                  linewidth=1.5, label='Baseline consistency (distinct run pairs)')
        axis.axhline(1, color='#888888', linestyle=':', linewidth=1,
                     label='Perfect similarity (1)')
        method = group.method.iloc[0]
        method_name = 'Average baseline embedding' if method == 'average' else 'All persona–baseline response pairs'
        axis.set_title(f'{model.title()} · {topic.replace("_", " ").title()} · {mode}\n'
                       f'{method_name}', loc='left', fontsize=13, pad=14)
        axis.set_xlabel('Turn / question position within topic')
        axis.set_ylabel('Cosine similarity to matched baseline')
        axis.set_ylim(-1.02, 1.06)
        axis.set_xticks(range(1, last_turn + 1))
        axis.grid(axis='y', color='#e5e5e5', linewidth=0.7)
        axis.spines[['top', 'right']].set_visible(False)
        axis.legend(loc='lower left', fontsize=8, framealpha=0.92)
        fig.supxlabel('Shading: ±1 descriptive SD, clipped to cosine bounds; not a confidence interval.\n'
                      'Questions change with turn; similarity alone does not establish persona adherence or causation.',
                      fontsize=8, color='#555555')
        slug = re.sub(r'[^A-Za-z0-9_-]', '_', topic)
        suffix = hashlib.sha256(filename.encode()).hexdigest()[:8]
        relative = Path('plots') / f'{model}__{slug}-{suffix}__{mode}.png'
        (output / relative).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output / relative, dpi=160)
        plt.close(fig)
        plots.append(str(relative))
    return plots


def positive_integer(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return value


def arguments(method, argv=None):
    parser = argparse.ArgumentParser(
        description=f'Multi-turn baseline comparison using the {method} method.', allow_abbrev=False)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('--input-root', type=Path,
                        help='recursively discover CSVs here (default: repository results/)')
    inputs.add_argument('--csv', type=Path, nargs='+', help='explicit response CSVs instead of discovery')
    parser.add_argument('--output-folder', type=Path, default=ROOT / 'analysis_outputs',
                        help='parent of a new method-specific directory (default: analysis_outputs/)')
    parser.add_argument('--cache-folder', type=Path, default=ROOT / '.cache' / 'response_embeddings')
    parser.add_argument('--embedding-model', default='sentence-transformers/all-mpnet-base-v2')
    parser.add_argument('--revision', help='Hugging Face model revision; resolved commit is recorded')
    parser.add_argument('--batch-size', type=positive_integer, default=32)
    parser.add_argument('--device', default='cpu', help='encoder device (default: cpu; e.g. mps or cuda)')
    return parser.parse_args(argv)


def analyze(method, args):
    output = args.output_folder.resolve() / method
    input_root = None if args.csv else (args.input_root or ROOT / 'results').resolve()
    cache = args.cache_folder.resolve()
    if cache.is_relative_to(output) or output.is_relative_to(cache):
        raise ValueError('Output and embedding cache directories must not overlap.')
    if input_root and (output.is_relative_to(input_root) or cache.is_relative_to(input_root)):
        raise ValueError('Output and cache directories must be outside the recursive input root.')
    paths = [p.resolve() for p in args.csv] if args.csv else discover_inputs(input_root)
    if any(path.is_relative_to(output) or path.is_relative_to(cache) for path in paths):
        raise ValueError('Output/cache directories cannot contain source CSVs.')
    if output.exists():
        raise ValueError(f'Analysis output already exists: {output}; choose a fresh --output-folder.')
    input_records = [dict(path=str(path), sha256=embeddings.file_sha256(path)) for path in paths]
    frame = load_responses(paths)
    coverage = coverage_table(frame)
    print(f'Loaded {len(frame)} responses from {len(paths)} CSVs | method={method}', flush=True)
    incomplete = coverage[coverage.complete_run_count != coverage.run_count]
    if not incomplete.empty:
        print(f'Incomplete turn coverage in {len(incomplete)} condition/topic groups; '
              'retaining available rows; see coverage.csv.', flush=True)
    vectors, metadata = embeddings.embed_responses(
        frame.response.tolist(), cache, args.embedding_model, args.revision, args.batch_size, args.device,
        stored_vectors=frame.embedding_vector.tolist(), stored_metadata=frame.embedding_metadata.tolist())
    scores, summary, consistency = score_responses(frame, vectors, method)
    for record in input_records:
        if embeddings.file_sha256(record['path']) != record['sha256']:
            raise ValueError(f'Input changed during analysis: {record["path"]}; rerun on stable inputs.')
    input_records = persist_embeddings(frame, vectors, metadata,
                                       {record['path']: record['sha256'] for record in input_records})
    enriched_files = [record['path'] for record in input_records if record['enriched']]
    metadata.pop('token_counts')
    print(f'Updated embedding columns in {len(enriched_files)} source CSVs.', flush=True)
    manifest = dict(
        status='complete', method=method, created_utc=datetime.now(timezone.utc).isoformat(),
        inputs=input_records, response_count=len(frame), score_count=len(scores),
        summary_count=len(summary), consistency_count=len(consistency), enriched_files=enriched_files,
        embedding=metadata, python_version=platform.python_version(),
        packages={name: version(name) for name in ['pandas', 'numpy', 'matplotlib']},
        command=sys.argv, arguments={key: str(value) if isinstance(value, Path) else
                                    [str(p) for p in value] if isinstance(value, list) else value
                                    for key, value in vars(args).items()},
        analysis_code_sha256={name: embeddings.file_sha256(Path(__file__).parent / name)
                              for name in ['multi_turn_response_analysis.py', 'embeddings.py', 'csv_embeddings.py']},
        interpretation=dict(
            similarity='Higher means closer to the matched unpersonalized baseline.',
            standard_deviation='Sample SD (ddof=1) of descriptive scores; not a confidence interval.',
            pairing='All cross-run pairs are dependent; score_count is not an independent sample size.',
            consistency='Mean cosine over distinct run pairs; excludes self-pairs.',
            hypothesis='Personalization may weaken despite retained history; baseline similarity '
                       'does not establish ignored history, persona adherence, or learned social values.',
            confounding='Question content and order change with turn; curves do not isolate causation.',
            generation_provenance='CSV schema does not record generation settings or model versions.',
            method_relationship='pairing_mean = average_mean * baseline_concentration '
                                'for the same normalized response embeddings.',
        ),
    )
    plots = publish_outputs(output, scores, summary, consistency, coverage, manifest)
    print(f'Saved {len(scores)} scores, {len(summary)} summaries, and {len(plots)} plots to {output}',
          flush=True)
    return output


def publish_outputs(output, scores, summary, consistency, coverage, manifest):
    """Publish complete output only; a plotting/disk error leaves no blocked target."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=f'.{output.name}-') as temporary:
        stage = Path(temporary)
        for name, table in [('scores', scores), ('summary', summary),
                            ('consistency', consistency), ('coverage', coverage)]:
            table.to_csv(stage / f'{name}.csv', index=False)
        plots = plot_summaries(summary, stage, consistency)
        manifest.update(plots=plots, plot_count=len(plots))
        (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n',
                                              encoding='utf-8')
        # Claim the destination without replacing a directory created by someone else.
        output.mkdir(exist_ok=False)
        try:
            os.replace(stage, output)
        except BaseException:
            output.rmdir()
            raise
    return plots


def main(method, argv=None):
    args = arguments(method, argv)
    try:
        analyze(method, args)
    except KeyboardInterrupt:
        print('Analysis interrupted. Completed embedding cache entries can be reused.', file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f'Analysis failed: {exc}', file=sys.stderr)
        return 1
    return 0
