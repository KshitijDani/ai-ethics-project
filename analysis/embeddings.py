"""Local response embeddings with explicit length checks and revision-aware caches."""
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def normalize_vectors(vectors):
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError('Embedding vectors must be a finite two-dimensional matrix.')
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms < 1e-12).any():
        raise ValueError('Embedding vector has zero or invalid norm.')
    return values / norms


def local_model_revision(path, expected=None):
    path = Path(path)
    digest = hashlib.sha256()
    for file in sorted(p for p in path.rglob('*') if p.is_file()):
        digest.update(str(file.relative_to(path)).encode())
        digest.update(file_sha256(file).encode())
    resolved = f'local-sha256:{digest.hexdigest()}'
    if expected is not None and expected != resolved:
        raise ValueError('Local model revision does not match the requested fingerprint.')
    return resolved


def vector_sha256(vector):
    return hashlib.sha256(np.asarray(vector, dtype='<f8').tobytes()).hexdigest()


def load_encoder(model_name, revision, device):
    # Lazy import keeps CSV/metric-only work independent of the model runtime.
    from sentence_transformers import SentenceTransformer

    local_path = Path(model_name)
    if local_path.is_dir():
        resolved = local_model_revision(local_path, revision)
    else:
        if revision and re.fullmatch(r'[a-fA-F0-9]{40}', revision):
            resolved = revision.lower()
        else:
            from huggingface_hub import model_info
            resolved = model_info(model_name, revision=revision).sha
        if not resolved:
            raise ValueError('Cannot determine encoder revision; use a local model directory.')
    encoder = SentenceTransformer(model_name, revision=None if local_path.is_dir() else resolved,
                                  device=device)
    if ('message' in getattr(encoder[0], 'modality_config', {})
            or getattr(encoder.tokenizer, 'chat_template', None)):
        raise ValueError('Use a plain-text embedding encoder without a chat template; '
                         'template-based token length validation is not supported.')
    return encoder, dict(model=model_name, revision=resolved,
                         sentence_transformers_version=version('sentence-transformers'),
                         transformers_version=version('transformers'), torch_version=version('torch'))


def stored_embeddings(texts, stored_vectors, stored_metadata, model_name, revision, device):
    """Return mutually compatible, response-matched vectors stored in CSV cells."""
    values, lengths, configurations = {}, {}, {}
    for text, vector_cell, metadata_cell in zip(texts, stored_vectors, stored_metadata):
        if not vector_cell or not metadata_cell:
            continue
        try:
            metadata = json.loads(metadata_cell)
            config = metadata['configuration']
            config_json = json.dumps(config, sort_keys=True)
            key = hashlib.sha256(config_json.encode()).hexdigest()
            if (config['model'] != model_name or config['device'] != device
                    or (revision is not None and config['revision'] != revision)
                    or metadata['response_sha256'] != hashlib.sha256(text.encode()).hexdigest()
                    or metadata['configuration_sha256'] != key
                    or config['normalization'] != 'unit-l2-float64'
                    or config['truncation'] != 'reject' or config['prompt'] != ''
                    or config['cache_format'] != 1):
                continue
            vector = np.asarray(json.loads(vector_cell), dtype=float)
            count = metadata['token_count']
            if (vector.shape != (config['dimension'],) or not np.isfinite(vector).all()
                    or not np.isclose(np.linalg.norm(vector), 1, atol=1e-7)
                    or metadata['vector_sha256'] != vector_sha256(vector)
                    or not isinstance(count, int) or not 0 < count <= config['max_sequence_length']):
                continue
        except (ValueError, TypeError, KeyError, AttributeError):
            continue
        if (key, text) in values and not np.array_equal(values[(key, text)], vector):
            raise ValueError('Conflicting stored embeddings for identical response text and configuration.')
        configurations[key] = config
        values[(key, text)] = vector
        lengths[(key, text)] = count
    if len(configurations) > 1:
        raise ValueError('CSV embeddings use multiple configurations; select a single --revision '
                         'or clear embedding columns before intentionally regenerating them.')
    if not configurations:
        return {}, {}, None
    key, config = next(iter(configurations.items()))
    return ({text: vector for (stored_key, text), vector in values.items() if stored_key == key},
            {text: count for (stored_key, text), count in lengths.items() if stored_key == key}, config)


def embed_responses(texts, cache_dir, model_name, revision=None, batch_size=32, device='cpu',
                    stored_vectors=None, stored_metadata=None):
    if batch_size < 1:
        raise ValueError('Embedding batch size must be positive.')
    unique_texts = list(dict.fromkeys(texts))
    if not unique_texts:
        raise ValueError('No response texts supplied for embedding.')
    if Path(model_name).is_dir():
        # Also check local weights when all vectors can be reused without loading a model.
        revision = local_model_revision(model_name, revision)
    stored, stored_lengths, stored_config = stored_embeddings(
        texts, stored_vectors if stored_vectors is not None else [''] * len(texts),
        stored_metadata if stored_metadata is not None else [''] * len(texts),
        model_name, revision, device)
    if all(text in stored for text in unique_texts):
        config_key = hashlib.sha256(json.dumps(stored_config, sort_keys=True).encode()).hexdigest()
        metadata = stored_config | dict(
            configuration=stored_config, configuration_sha256=config_key,
            cache_directory=None, unique_response_count=len(unique_texts), cache_misses=0,
            cache_hits=0, stored_vector_hits=len(stored),
            maximum_observed_tokens=max(stored_lengths.values()),
            token_counts=[stored_lengths[text] for text in texts],
        )
        print(f'Reused {len(stored)} unique embeddings from CSVs; no encoder loading needed.', flush=True)
        return np.stack([stored[text] for text in texts]), metadata
    pinned = revision or (stored_config['revision'] if stored_config else None)
    encoder, configuration = load_encoder(model_name, pinned, device)
    limit = int(encoder.max_seq_length)
    dimension_method = (getattr(encoder, 'get_embedding_dimension', None)
                        or encoder.get_sentence_embedding_dimension)
    dimension = int(dimension_method())
    configuration = configuration | dict(max_sequence_length=limit, dimension=dimension,
                                         normalization='unit-l2-float64', prompt='',
                                         truncation='reject', device=device, cache_format=1)
    config_json = json.dumps(configuration, sort_keys=True)
    key = hashlib.sha256(config_json.encode()).hexdigest()
    folder = Path(cache_dir) / key
    if stored_config != configuration:
        stored, stored_lengths = {}, {}
    token_lengths = []
    for start in range(0, len(unique_texts), batch_size):
        batch = unique_texts[start:start + batch_size]
        tokens = encoder.tokenizer(batch, add_special_tokens=True, truncation=False, padding=False)
        lengths = [len(ids) for ids in tokens['input_ids']]
        token_lengths.extend(lengths)
        for text, length in zip(batch, lengths):
            if length > limit:
                raise ValueError(f'Response has {length} tokens, exceeding encoder token limit {limit}; '
                                 f'no silent truncation is allowed. Text starts: {text[:100]!r}')
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'configuration.json').write_text(config_json + '\n', encoding='utf-8')
    values = dict(stored)
    missing = []
    for text in unique_texts:
        if text in values:
            continue
        path = folder / (hashlib.sha256(text.encode('utf-8')).hexdigest() + '.npy')
        if path.exists():
            try:
                vector = np.load(path, allow_pickle=False)
            except (OSError, ValueError) as exc:
                raise ValueError(f'Invalid embedding cache: {path}') from exc
            if (vector.shape != (dimension,) or not np.isfinite(vector).all()
                    or not np.isclose(np.linalg.norm(vector), 1, atol=1e-7)):
                raise ValueError(f'Invalid cached vector: {path}; remove it and rerun.')
            values[text] = vector
        else:
            missing.append((text, path))
    print(f'Embedding {len(unique_texts)} unique responses | cache hits={len(values)} | '
          f'new={len(missing)} | maximum tokens={max(token_lengths)}/{limit}', flush=True)
    for start in range(0, len(missing), batch_size):
        batch = missing[start:start + batch_size]
        vectors = normalize_vectors(encoder.encode(
            [text for text, _ in batch], batch_size=batch_size, prompt='',
            show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False,
        ))
        if vectors.shape != (len(batch), dimension):
            raise ValueError('Encoder returned an unexpected vector matrix shape.')
        for (text, path), vector in zip(batch, vectors):
            # Publish complete cache entries atomically, including on interruption.
            with tempfile.NamedTemporaryFile(dir=folder, suffix='.npy', delete=False) as stream:
                temporary = Path(stream.name)
                np.save(stream, vector, allow_pickle=False)
            os.replace(temporary, path)
            values[text] = vector
        print(f'Encoded {min(start + batch_size, len(missing))}/{len(missing)} new responses', flush=True)
    length_by_text = dict(zip(unique_texts, token_lengths))
    metadata = configuration | dict(configuration_sha256=key, cache_directory=str(folder.resolve()),
                                    configuration=configuration, stored_vector_hits=len(stored),
                                    unique_response_count=len(unique_texts), cache_misses=len(missing),
                                    cache_hits=len(unique_texts) - len(missing) - len(stored),
                                    maximum_observed_tokens=max(token_lengths),
                                    token_counts=[length_by_text[text] for text in texts])
    return np.stack([values[text] for text in texts]), metadata
