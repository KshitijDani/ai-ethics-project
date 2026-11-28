# Multi-turn Response Analysis Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Deliver two reproducible analysis commands and run both on all uploaded responses.

**Architecture:** A shared module loads and validates response CSVs, caches normalized local embeddings, computes method-specific scores, and exports consistent tables and plots. Two small entry points select average-baseline or all-pairs scoring.

**Tech Stack:** Python 3.10+, pandas, NumPy, Matplotlib, Sentence Transformers.

**Spec:** `docs/superpowers/specs/2026-11-28-multi-turn-response-analysis-design.md` (approved in chat).

## Global Constraints

- Work on the user's existing `scripts` branch and preserve their local edits.
- Embed assistant responses only, using the same encoder/configuration for both methods.
- Default inputs: `results/`. Preserve original input fields while enriching CSVs with approved embedding columns; never overwrite an existing completed analysis.
- Unit-normalize response embeddings; reject zero centroid for the average method.
- Show descriptive variation, never claim confidence intervals or independent pair counts.
- Preserve identical text across distinct runs and inspect token lengths before encoding.
- Report departure from baseline separately from within-condition consistency; retain the agreed hypothesis and interpretation boundaries.

## Review Focus

- Duplicate input files or response identities must fail before embedding.
- Single-run and incomplete conditions must report coverage and omit undefined spread.
- Invalid vectors, stale/incompatible caches, and oversized texts must fail clearly.
- Question text/ID mismatches must never silently match by turn alone.
- Unsafe output locations must never alter source data or mix generated CSVs into inputs.

### Task 1: Load and validate experiment inputs

**Files:** `analysis/__init__.py`, `analysis/multi_turn_response_analysis.py`, `tests/test_response_analysis.py`.

**Interfaces:** `load_responses(paths: list[Path]) -> pandas.DataFrame`; `discover_inputs(root: Path) -> list[Path]`; `coverage_table(frame) -> pandas.DataFrame`.

- [x] Write tests for recursive layouts, actual data counts, duplicate identities, blank data, question alignment, mode validation, and incomplete coverage.
- [x] Run `python -m unittest discover -s tests -p test_response_analysis.py -v`; confirm missing implementation fails.
- [x] Implement schema/identity checks and preserve source/run provenance without deduplicating responses.
- [x] Run the new tests and verify they pass.

### Task 2: Metrics and reproducible local embeddings

**Files:** shared module, `analysis/embeddings.py`, `requirements-analysis.txt`, analysis tests.

**Interfaces:** `score_responses(frame, vectors, method) -> (scores, summary, consistency)`; `embed_responses(texts, cache_dir, model_name, revision, batch_size, device) -> (ndarray, metadata)`.

- [x] Add known-vector tests: normalized centroid score, all-pairs counts, baseline/within-persona consistency, method relationship, zero vectors, single-run spread, and centroid cancellation.
- [x] Add cache/length validation tests using an injected small encoder at the external model boundary.
- [x] Watch those tests fail; implement scoring and embedding persistence keyed by exact text, encoder revision, and length/normalization policy.
- [x] Verify tests pass, add separately installable analysis dependencies, and record actual resolved model revision/configuration.

### Task 3: Commands, plots, and real-data validation

**Files:** two entry scripts, shared module, README, `.gitignore`, analysis tests.

**Interfaces:** `main(method, argv=None) -> int`; command flags `--input-root`, `--csv`, `--output-folder`, `--cache-folder`, `--embedding-model`, `--revision`, `--batch-size`, `--device`.

- [x] Write an end-to-end fixture test with injected encoder: run each method, read actual output tables/manifest/PNG, reject collisions and unsafe paths.
- [x] Watch it fail; implement common CLI, method-specific fresh output directories, summary/raw/consistency/coverage tables, provenance manifest, and 24 plots per method on current inputs.
- [x] Use consistent persona colors and fixed cosine axes [-1, 1]; label bands as descriptive SD. Export baseline and persona pairwise consistency separately.
- [x] Update README with install/run/output instructions and interpretation notes; ignore caches and generated analysis output.
- [x] Run full unittest suite and both actual commands; inspect a generated plot and validate counts and the score relationship from exported files.
- [x] Request a fresh independent review, address material findings with reproducing tests, and rerun relevant checks.

## Execution record

- User approved implementation of the design; proceed inline without another approval round.
- Keep changes reviewable and uncommitted alongside the existing script reorganization; do not stage unrelated persona/question edits.
- Task 1 completed: seven CSV validation tests RED→GREEN; baseline suite 34/34 passed.
- Task 2 completed: known-vector scoring and embedding cache tests RED→GREEN.
- Task 3: both commands completed initial full-data runs; 51/51 tests passed before the persistence amendment.
- Ruling: execute on the existing user-requested `scripts` branch and retain uncommitted work rather than introduce another checkout/approval round; this keeps the existing reorganization and local edits together.
- Ruling: user requested source CSV embedding persistence during implementation; this explicitly supersedes the initial no-source-mutation rule for embedding columns only.
- Independent review identified source vector binding, concurrent edit preservation, incomplete baseline plotting, offline pinned revisions, local fingerprints, and output-directory overlap. Added reproducing tests and fixed them. Plain-text encoders are supported; chat-template encoders are rejected so length checks cannot silently miss template tokens.
- Persistence amendment: add `analysis/csv_embeddings.py`; tests cover encoder-free reuse, response/model invalidation, vector checksum recovery, preservation of NUL/multiline/extra data, concurrent edit rejection, and atomic output publication. 29 analysis tests pass after this fix pass.

- Final verification: full unittest suite 63/63 passed; pip dependency check and git diff whitespace check passed.
- Final real-data outputs: `analysis_outputs/final/average/` (3,600 scores, 720 summaries, 24 PNGs) and `analysis_outputs/final/pairing/` (18,000 scores, 720 summaries, 24 PNGs). Both have 840 consistency and 84 coverage rows.
- Verified all 4,200 stored 768-dimensional vectors and all nine enriched CSVs: original fields exactly match Git HEAD. Pairing reused 2,595 unique CSV embeddings without encoder loading or new inference.
- Verified manifests against final source hashes; maximum error in the mathematical relationship between the two mean metrics was 9.992e-16. Inspected matching average and pairing plots visually.
- Independent reviewer findings fixed with reproducing tests; no material findings or deferred minor items remain. Work remains uncommitted on `scripts`, as intended for review.
