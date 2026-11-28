"""Persist audited embedding vectors while preserving all original CSV fields."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import pandas as pd
from filelock import FileLock

from analysis.embeddings import file_sha256, vector_sha256


def persist_embeddings(frame, vectors, metadata, expected_hashes):
    configuration = metadata['configuration']
    config_key = metadata['configuration_sha256']
    counts = metadata['token_counts']
    cells = [json.dumps(vector.tolist(), separators=(',', ':'), allow_nan=False) for vector in vectors]
    metadata_cells = [json.dumps(dict(
        configuration=configuration, configuration_sha256=config_key,
        response_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest(), token_count=int(count),
        vector_sha256=vector_sha256(vector),
    ), sort_keys=True, separators=(',', ':')) for text, count, vector in zip(frame.response, counts, vectors)]
    records = []
    for filename, rows in frame.groupby('source_file', sort=True):
        path = Path(filename)
        expected = expected_hashes[str(path)]
        with FileLock(str(path) + '.embedding.lock', timeout=30):
            if file_sha256(path) != expected:
                raise ValueError(f'CSV changed before embedding persistence: {path}; rerun.')
            original = pd.read_csv(path, dtype=str, keep_default_na=False, engine='python')
            if len(original) != len(rows):
                raise ValueError(f'CSV row count changed before embedding persistence: {path}')
            for column in ('embedding_vector', 'embedding_metadata'):
                if column not in original:
                    original[column] = ''
            updated = original.copy()
            for index, row in rows.iterrows():
                position = int(row.source_row) - 1
                if original.iloc[position]['response'] != row.response:
                    raise ValueError(f'CSV response changed before embedding persistence: {path}:{position + 1}')
                updated.loc[position, 'embedding_vector'] = cells[index]
                updated.loc[position, 'embedding_metadata'] = metadata_cells[index]
            changed = not updated.equals(original)
            after_hash = expected
            temporary = None
            try:
                if changed:
                    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='',
                                                     dir=path.parent, prefix=f'.{path.name}.',
                                                     suffix='.tmp', delete=False) as stream:
                        temporary = Path(stream.name)
                        updated.to_csv(stream, index=False)
                        stream.flush()
                        os.fsync(stream.fileno())
                    after_hash = file_sha256(temporary)
                    os.chmod(temporary, path.stat().st_mode & 0o777)
                # Check again after serializing the potentially large enriched CSV.
                if file_sha256(path) != expected:
                    raise ValueError(f'CSV changed during embedding persistence: {path}; rerun.')
                if changed:
                    os.replace(temporary, path)
                records.append(dict(path=str(path), sha256=after_hash,
                                    sha256_before_enrichment=expected, enriched=changed))
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
    return records
