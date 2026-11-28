"""Checks for matching real experiment rows without conflating repetitions."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from analysis import multi_turn_response_analysis as analysis


def fixture_rows():
    rows = []
    for mode, persona in [('baseline', 'baseline'), ('prompt-prepend', 'American')]:
        for run in (1, 2):
            for turn in (1, 2):
                rows.append(dict(
                    model='gemma', mode=mode, persona=persona,
                    question_file_name='questions_books.txt', question_number=f'books_{turn}',
                    run_id=f'multi-turn/gemma/{mode}/books/{persona}/{run}',
                    run_number=run, turn=turn, question=f'Question {turn}',
                    user_message=f'Question {turn}', response=f'{persona} {run} {turn}',
                    timestamp='2026-11-28T00:00:00+00:00',
                ))
    return pd.DataFrame(rows)


class InputTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(callable(getattr(analysis, 'load_responses', None)),
                        'CSV loading has not been implemented')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, frame, name='responses.csv'):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        return path

    def test_recursive_discovery_uses_csv_columns_and_preserves_repeated_answers(self):
        frame = fixture_rows()
        frame['response'] = 'The same recommendation'
        self.write(frame[frame['mode'] == 'baseline'], 'nested/arbitrary.csv')
        self.write(frame[frame['mode'] != 'baseline'], 'uploaded once.csv')
        result = analysis.load_responses(analysis.discover_inputs(self.root))
        self.assertEqual(len(result), 8)
        self.assertEqual(result.response.nunique(), 1)
        self.assertEqual(result.source_file.nunique(), 2)

    def test_duplicate_file_and_identity_are_rejected(self):
        path = self.write(fixture_rows())
        with self.assertRaisesRegex(ValueError, 'Duplicate input'):
            analysis.load_responses([path, path])
        path = self.write(pd.concat([fixture_rows(), fixture_rows().iloc[:1]]))
        with self.assertRaisesRegex(ValueError, 'Duplicate response'):
            analysis.load_responses([path])

    def test_blank_response_invalid_mode_and_noninteger_run_are_rejected(self):
        for column, value, message in [('response', ' ', 'blank'),
                                       ('mode', 'surprise', 'mode'),
                                       ('run_number', 1.5, 'positive integer')]:
            with self.subTest(column=column):
                frame = fixture_rows().astype({column: object})
                frame.loc[0, column] = value
                with self.assertRaisesRegex(ValueError, message):
                    analysis.load_responses([self.write(frame)])

    def test_question_text_and_id_mismatches_and_missing_baselines_are_rejected(self):
        for column in ('question', 'question_number'):
            frame = fixture_rows()
            frame.loc[4, column] = 'changed'
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, 'Question'):
                analysis.load_responses([self.write(frame)])
        frame = fixture_rows().iloc[4:]
        with self.assertRaisesRegex(ValueError, 'baseline'):
            analysis.load_responses([self.write(frame)])

    def test_single_turn_and_unsafe_model_names_are_rejected(self):
        for column, value, error in [('run_id', 'single-turn/gemma/1', 'multi-turn'),
                                     ('model', '../../elsewhere', 'filename-safe')]:
            frame = fixture_rows()
            frame.loc[0, column] = value
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, error):
                analysis.load_responses([self.write(frame)])

    def test_embedded_nul_and_multiline_response_are_not_silently_truncated(self):
        frame = fixture_rows()
        frame.loc[0, 'response'] = 'first\x00second\nthird'
        result = analysis.load_responses([self.write(frame)])
        self.assertEqual(result.response.iloc[0], 'first\x00second\nthird')

    def test_incomplete_run_is_reported_without_discarding_rows(self):
        result = analysis.load_responses([self.write(fixture_rows().iloc[:-1])])
        coverage = analysis.coverage_table(result)
        row = coverage[coverage['mode'] == 'prompt-prepend'].iloc[0]
        self.assertEqual(row['response_count'], 3)
        self.assertEqual(row['complete_run_count'], 1)
        self.assertEqual(row['expected_turn_count'], 2)

    def test_uploaded_dataset_is_complete_and_aligns(self):
        root = Path(__file__).resolve().parents[1] / 'results'
        result = analysis.load_responses(analysis.discover_inputs(root))
        self.assertEqual(len(result), 4200)
        coverage = analysis.coverage_table(result)
        self.assertEqual(len(coverage), 84)
        self.assertTrue(coverage.complete_run_count.eq(5).all())
        self.assertTrue(coverage.expected_turn_count.eq(10).all())


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(callable(getattr(analysis, 'score_responses', None)),
                        'Response scoring has not been implemented')
        self.frame = fixture_rows()
        self.frame['source_file'] = 'fixture.csv'
        self.frame['source_row'] = np.arange(1, 9)
        # At both turns: baseline [1,0],[0,1]; persona [1,0],[1,0].
        self.vectors = np.array([[1, 0], [1, 0], [0, 1], [0, 1],
                                 [1, 0], [1, 0], [1, 0], [1, 0]], dtype=float)

    def test_average_and_pairing_scores_and_their_relationship(self):
        raw_a, avg, consistency = analysis.score_responses(self.frame, self.vectors, 'average')
        raw_p, pairs, _ = analysis.score_responses(self.frame, self.vectors, 'pairing')
        self.assertEqual(len(raw_a), 4)
        self.assertEqual(len(raw_p), 8)
        np.testing.assert_allclose(avg.mean_similarity, [2**-0.5, 2**-0.5])
        np.testing.assert_allclose(pairs.mean_similarity, [0.5, 0.5])
        np.testing.assert_allclose(avg.baseline_concentration, [2**-0.5, 2**-0.5])
        np.testing.assert_allclose(pairs.mean_similarity,
                                   avg.mean_similarity * avg.baseline_concentration)
        self.assertTrue(avg.std_similarity.eq(0).all())
        np.testing.assert_allclose(pairs.std_similarity, [3**-0.5, 3**-0.5])
        self.assertEqual(set(raw_p.baseline_run_id),
                         {'multi-turn/gemma/baseline/books/baseline/1',
                          'multi-turn/gemma/baseline/books/baseline/2'})
        self.assertTrue(consistency.loc[consistency['mode'] == 'baseline', 'mean_similarity'].eq(0).all())
        self.assertTrue(consistency.loc[consistency['mode'] != 'baseline', 'mean_similarity'].eq(1).all())
        self.assertTrue(consistency.pair_count.eq(1).all())

    def test_response_vectors_are_normalized_and_nonfinite_or_zero_rejected(self):
        _, summary, _ = analysis.score_responses(self.frame, self.vectors * 5, 'average')
        np.testing.assert_allclose(summary.mean_similarity, [2**-0.5, 2**-0.5])
        for value in (0, np.nan, np.inf):
            broken = self.vectors.copy()
            broken[0] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'vector'):
                analysis.score_responses(self.frame, broken, 'average')

    def test_single_run_has_undefined_spread_and_consistency(self):
        keep = self.frame.run_number.eq(1).to_numpy()
        for method in ('average', 'pairing'):
            _, summary, consistency = analysis.score_responses(
                self.frame[keep].reset_index(drop=True), self.vectors[keep], method)
            self.assertTrue(summary.std_similarity.isna().all())
            self.assertTrue(consistency.mean_similarity.isna().all())
            self.assertTrue(consistency.pair_count.eq(0).all())

    def test_cancelled_centroid_rejected_only_for_average(self):
        self.vectors[2:4] = [-1, 0]
        with self.assertRaisesRegex(ValueError, 'centroid'):
            analysis.score_responses(self.frame, self.vectors, 'average')
        _, summary, _ = analysis.score_responses(self.frame, self.vectors, 'pairing')
        self.assertTrue(summary.mean_similarity.eq(0).all())


class TinyTokenizer:
    def __call__(self, texts, **kwargs):
        return {'input_ids': [[0] * (len(text.split()) + 2) for text in texts]}


class TinyEncoder:
    """External encoder substitute; scoring, cache IO, and validation stay real."""
    tokenizer = TinyTokenizer()
    max_seq_length = 6
    prompts = {}
    default_prompt_name = None
    modality_config = {'text': {}}

    def __getitem__(self, index):
        return self

    def get_sentence_embedding_dimension(self):
        return 2

    def encode(self, texts, **kwargs):
        return np.array([[len(text), 1] for text in texts], dtype=float)


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        from analysis import embeddings
        self.embeddings = embeddings
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = Path(self.temp.name)

    def embed(self, texts, revision='abc', encoder=None):
        with patch.object(self.embeddings, 'load_encoder',
                          return_value=(encoder or TinyEncoder(), {'revision': revision, 'model': 'tiny'})):
            return self.embeddings.embed_responses(texts, self.cache, 'tiny', revision, 2, 'cpu')

    def test_cache_reuses_exact_text_and_preserves_input_order(self):
        first, meta = self.embed(['short', 'a bit', 'short'])
        np.testing.assert_allclose(first[0], [5 / np.sqrt(26), 1 / np.sqrt(26)])
        np.testing.assert_array_equal(first[0], first[2])
        self.assertEqual(meta['cache_misses'], 2)
        second, meta = self.embed(['a bit', 'short'])
        np.testing.assert_array_equal(second, first[[1, 0]])
        self.assertEqual(meta['cache_misses'], 0)
        _, changed = self.embed(['short'], revision='def')
        self.assertEqual(changed['cache_misses'], 1)

    def test_overlength_text_is_rejected_before_silent_truncation(self):
        with self.assertRaisesRegex(ValueError, 'token.*limit|limit.*token'):
            self.embed(['one two three four five'])

    def test_invalid_cached_vector_is_rejected(self):
        self.embed(['short'])
        path = next(self.cache.rglob('*.npy'))
        np.save(path, np.array([np.nan, 0]))
        with self.assertRaisesRegex(ValueError, 'cache|vector'):
            self.embed(['short'])

    def test_hub_revision_does_not_depend_on_private_transformer_config(self):
        encoder = TinyEncoder()
        with patch('sentence_transformers.SentenceTransformer', return_value=encoder), \
                patch('huggingface_hub.model_info', return_value=SimpleNamespace(sha='a' * 40)):
            loaded, metadata = self.embeddings.load_encoder('tiny', None, 'cpu')
        self.assertIs(loaded, encoder)
        self.assertEqual(metadata['revision'], 'a' * 40)

    def test_pinned_hub_revision_works_without_online_metadata_lookup(self):
        with patch('sentence_transformers.SentenceTransformer', return_value=TinyEncoder()), \
                patch('huggingface_hub.model_info', side_effect=OSError('offline')):
            _, metadata = self.embeddings.load_encoder('tiny', 'b' * 40, 'cpu')
        self.assertEqual(metadata['revision'], 'b' * 40)

    def test_local_revision_rejects_changed_weights(self):
        with patch('sentence_transformers.SentenceTransformer', return_value=TinyEncoder()):
            (self.cache / 'weights.bin').write_bytes(b'weights')
            with self.assertRaisesRegex(ValueError, 'Local model revision'):
                self.embeddings.load_encoder(str(self.cache), 'local-sha256:wrong', 'cpu')

    def test_chat_template_encoder_is_rejected_before_incomplete_length_check(self):
        encoder = TinyEncoder()
        encoder.modality_config = {'message': {}}
        with patch('sentence_transformers.SentenceTransformer', return_value=encoder), \
                self.assertRaisesRegex(ValueError, 'plain-text|chat'):
            self.embeddings.load_encoder('tiny', 'b' * 40, 'cpu')


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(callable(getattr(analysis, 'main', None)),
                        'Analysis command has not been implemented')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root / 'input'
        self.inputs.mkdir()
        self.source = self.inputs / 'responses.csv'
        fixture_rows().to_csv(self.source, index=False)

    def run_command(self, method, *extra, forbid_encoder=False):
        from analysis import embeddings
        with patch.object(embeddings, 'load_encoder',
                          return_value=(TinyEncoder(), {'revision': 'fixture', 'model': 'tiny'}),
                          side_effect=AssertionError('Stored vectors must avoid model loading')
                          if forbid_encoder else None):
            return analysis.main(method, [
                '--input-root', str(self.inputs), '--output-folder', str(self.root / 'output'),
                '--cache-folder', str(self.root / 'cache'), '--embedding-model', 'tiny', *extra])

    def test_both_methods_write_auditable_outputs_and_share_cache(self):
        before = pd.read_csv(self.source, dtype=str, keep_default_na=False)
        for method, expected_scores in [('average', 4), ('pairing', 8)]:
            self.assertEqual(self.run_command(method, forbid_encoder=method == 'pairing'), 0)
            folder = self.root / 'output' / method
            self.assertEqual(len(pd.read_csv(folder / 'scores.csv')), expected_scores)
            self.assertEqual(len(pd.read_csv(folder / 'summary.csv')), 2)
            self.assertEqual(len(pd.read_csv(folder / 'consistency.csv')), 4)
            self.assertEqual(len(pd.read_csv(folder / 'coverage.csv')), 2)
            manifest = json.loads((folder / 'manifest.json').read_text())
            self.assertEqual(manifest['method'], method)
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(len(manifest['inputs'][0]['sha256']), 64)
            png = next(folder.rglob('*.png'))
            self.assertEqual(png.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
            if method == 'pairing':
                self.assertEqual(manifest['embedding']['cache_misses'], 0)
                self.assertEqual(manifest['embedding']['stored_vector_hits'], 8)
        after = pd.read_csv(self.source, dtype=str, keep_default_na=False)
        pd.testing.assert_frame_equal(after[before.columns], before)
        self.assertEqual(len(json.loads(after.embedding_vector.iloc[0])), 2)
        self.assertEqual(json.loads(after.embedding_metadata.iloc[0])['configuration']['model'], 'tiny')

    def test_changed_response_invalidates_its_stored_vector_only(self):
        self.assertEqual(self.run_command('average'), 0)
        frame = pd.read_csv(self.source, keep_default_na=False)
        frame.loc[0, 'response'] = 'changed'
        frame.to_csv(self.source, index=False)
        self.assertEqual(self.run_command('pairing'), 0)
        manifest = json.loads((self.root / 'output/pairing/manifest.json').read_text())
        self.assertEqual(manifest['embedding']['stored_vector_hits'], 7)
        self.assertEqual(manifest['embedding']['cache_misses'], 1)

    def test_existing_embedding_configuration_does_not_override_requested_model(self):
        self.assertEqual(self.run_command('average'), 0)
        from analysis import embeddings
        with patch.object(embeddings, 'load_encoder',
                          return_value=(TinyEncoder(), {'revision': 'other', 'model': 'other'})):
            self.assertEqual(analysis.main('pairing', [
                '--input-root', str(self.inputs), '--output-folder', str(self.root / 'output'),
                '--cache-folder', str(self.root / 'cache'), '--embedding-model', 'other']), 0)
        manifest = json.loads((self.root / 'output/pairing/manifest.json').read_text())
        self.assertEqual(manifest['embedding']['stored_vector_hits'], 0)
        self.assertEqual(manifest['embedding']['model'], 'other')

    def test_swapped_vector_cells_are_recovered_from_cache(self):
        frame = fixture_rows()
        frame.loc[0, 'response'] = 'a'
        frame.loc[1, 'response'] = 'longer'
        frame.to_csv(self.source, index=False)
        self.assertEqual(self.run_command('average'), 0)
        frame = pd.read_csv(self.source, keep_default_na=False)
        frame.loc[[0, 1], 'embedding_vector'] = frame.loc[[1, 0], 'embedding_vector'].to_numpy()
        frame.to_csv(self.source, index=False)
        self.assertEqual(self.run_command('pairing'), 0)
        fixed = pd.read_csv(self.source, keep_default_na=False)
        np.testing.assert_allclose(json.loads(fixed.embedding_vector.iloc[0]), [2**-0.5, 2**-0.5])
        manifest = json.loads((self.root / 'output/pairing/manifest.json').read_text())
        self.assertEqual(manifest['embedding']['cache_misses'], 0)
        self.assertEqual(manifest['embedding']['cache_hits'], 2)

    def test_concurrent_source_edit_is_preserved_and_enrichment_aborts(self):
        from analysis import csv_embeddings
        original_save = pd.DataFrame.to_csv
        changed = False

        def save_then_external_edit(frame, *args, **kwargs):
            nonlocal changed
            result = original_save(frame, *args, **kwargs)
            if hasattr(args[0], 'name') and str(args[0].name).endswith('.tmp') and not changed:
                changed = True
                source = pd.read_csv(self.source, dtype=str, keep_default_na=False)
                source.loc[0, 'question'] = 'External edit'
                original_save(source, self.source, index=False)
            return result

        with patch.object(csv_embeddings.pd.DataFrame, 'to_csv', new=save_then_external_edit):
            self.assertEqual(self.run_command('average'), 1)
        saved = pd.read_csv(self.source, keep_default_na=False)
        self.assertEqual(saved.question.iloc[0], 'External edit')
        self.assertNotIn('embedding_vector', saved.columns)

    def test_existing_output_and_output_inside_inputs_are_rejected(self):
        target = self.root / 'output' / 'average'
        target.mkdir(parents=True)
        marker = target / 'keep.txt'
        marker.write_text('keep')
        self.assertEqual(self.run_command('average'), 1)
        self.assertEqual(marker.read_text(), 'keep')
        self.assertEqual(self.run_command('pairing', '--output-folder', str(self.inputs / 'analysis')), 1)
        self.assertFalse((self.inputs / 'analysis').exists())

    def test_output_cache_overlap_is_rejected_before_input_enrichment(self):
        before = self.source.read_bytes()
        self.assertEqual(self.run_command('average', '--cache-folder',
                                          str(self.root / 'output' / 'average')), 1)
        self.assertEqual(self.source.read_bytes(), before)

    def test_source_enrichment_preserves_extra_columns_and_nuls(self):
        frame = fixture_rows()
        frame['notes'] = 'quoted, value\nline two'
        frame.loc[0, 'response'] = 'a\x00b'
        frame.to_csv(self.source, index=False)
        before = pd.read_csv(self.source, dtype=str, keep_default_na=False, engine='python')
        self.assertEqual(self.run_command('average'), 0)
        after = pd.read_csv(self.source, dtype=str, keep_default_na=False, engine='python')
        pd.testing.assert_frame_equal(after[before.columns], before)

    def test_failed_plot_does_not_publish_partial_output(self):
        with patch.object(analysis, 'plot_summaries', side_effect=OSError('disk error')):
            self.assertEqual(self.run_command('average'), 1)
        self.assertFalse((self.root / 'output/average').exists())

    def test_plot_preserves_baseline_when_all_persona_rows_at_last_turn_are_missing(self):
        frame = fixture_rows()
        frame = frame[~((frame['mode'] != 'baseline') & frame.turn.eq(2))].reset_index(drop=True)
        frame['source_file'] = 'fixture'
        frame['source_row'] = np.arange(1, len(frame) + 1)
        vectors = np.tile([1., 0.], (len(frame), 1))
        _, summary, consistency = analysis.score_responses(frame, vectors, 'average')
        from matplotlib.figure import Figure
        saved = []
        original = Figure.savefig

        def capture(figure, *args, **kwargs):
            saved.append([(line.get_label(), list(line.get_xdata()), list(line.get_ydata()))
                          for line in figure.axes[0].lines])
            return original(figure, *args, **kwargs)

        with patch.object(Figure, 'savefig', new=capture):
            analysis.plot_summaries(summary, self.root / 'plots', consistency)
        baseline = next(line for line in saved[0] if line[0].startswith('Baseline consistency'))
        self.assertEqual(baseline[1], [1, 2])
        self.assertEqual(baseline[2], [1, 1])

    def test_one_model_topic_plot_contains_both_modes_and_all_personas(self):
        frame = fixture_rows()
        frame['source_file'] = 'fixture'
        frame['source_row'] = np.arange(1, len(frame) + 1)
        _, summary, consistency = analysis.score_responses(
            frame, np.tile([1., 0.], (len(frame), 1)), 'average')
        combined = pd.concat([
            summary.assign(mode=mode, persona=persona)
            for mode in ('prompt-prepend', 'prompt-prepend-once')
            for persona in ('American', 'Chinese', 'French')
        ], ignore_index=True)
        from matplotlib.figure import Figure
        saved = []
        original = Figure.savefig

        def capture(figure, *args, **kwargs):
            saved.append({line.get_label(): (line.get_color(), line.get_linestyle())
                          for line in figure.axes[0].lines})
            return original(figure, *args, **kwargs)

        with patch.object(Figure, 'savefig', new=capture):
            plots = analysis.plot_summaries(combined, self.root / 'combined', consistency)
        self.assertEqual(len(plots), 1)
        self.assertEqual(len(saved), 1)
        for persona in ('American', 'Chinese', 'French'):
            repeated = saved[0][f'{persona} · prompt-prepend']
            once = saved[0][f'{persona} · prompt-prepend-once']
            self.assertEqual(repeated[0], once[0])
            self.assertNotEqual(repeated[1], once[1])
        self.assertEqual(len(saved[0]), 8)  # six comparisons and two baseline references


if __name__ == '__main__':
    unittest.main()
