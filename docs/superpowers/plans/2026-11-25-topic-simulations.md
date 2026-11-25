# Topic-based simulations implementation plan

Approved in chat on November 25, 2026. Analysis and question authoring are deferred.

## Design

Both entrypoints share discovery, orchestration, CSV formatting, and progress logging
in `simulate.py`. Each retains its own request generator so multi-turn history and
stateless single-turn requests remain distinct. Preserve the existing workspace's
managed-server and configuration changes.

## Tasks

- [x] Add regression tests for the input, execution, output, and logging contracts;
  run them before implementation to verify the missing functionality.
- [x] Replace `--questions` with `--questions-folder`, defaulting to repository
  `data/`. Discover sorted direct-child `question_<topic>.txt` files, skipping
  directories and unrelated files. Read UTF-8 with optional BOM, strip lines,
  ignore blank lines, and preserve order and duplicates. Fail before server
  startup for no matching files, empty topic names, or empty question files.
- [x] Run model → mode → topic → persona → repetition → question. Default to five
  repetitions; retain `--runs`. Default mode order is baseline, prompt-prepend,
  prompt-prepend-once. Finish all personas and repetitions within each topic.
- [x] Keep multi-turn history only within one topic/persona/repetition. Prepend
  persona every turn or first turn according to mode. Single-turn supports all
  three labels, with persona prepended on every stateless personalized request.
- [x] Separate outputs and server logs under `multi-turn/` and `single-turn/`.
  Keep overwrite preflight, partial saves, and server cleanup.
- [x] Write the exact CSV schema:
  `model,mode,persona,question_file_name,question_number,run_id,run_number,turn,question,user_message,response,timestamp`.
  Use `<topic>_<turn>` question IDs and
  `<experiment>/<model>/<mode>/<topic>/<persona>/<run_number>` run IDs. Baseline's
  persona is `baseline`; single-turn run IDs group independent requests.
- [x] Print flushed extraction counts and ID ranges, total inputs, experiment
  summary and output path, model/mode/topic starts, and run starts/completions.
  Add `--verbose` for extracted question IDs/text and per-request progress.
  Keep assistant answer bodies and persona prompt text out of normal progress
  logs, and do not print success for failed/interrupted runs.
- [x] Update README and CLI help. Leave existing input files untouched; the topic
  files will be supplied later. Do not add analysis dependencies or migrate old
  outputs.

## Verification

Run `python -m unittest discover -s tests -v` with the project virtual environment.
Use mock HTTP servers and temporary input/output folders, not real inference.
Verify discovery and invalid-input preflight, exact condition order, history
boundaries, stateless behavior under all labels, CSV round-trips and IDs,
experiment isolation, default/verbose logs, partial saves, collisions, and server
cleanup. The planned four-topic, ten-question, three-model, three-persona,
five-repetition matrix must yield nine CSVs and 4,200 responses per script.

## Execution notes

- Baseline: 27 tests passed before implementation.
- Regression tests failed first on the missing directory CLI/discovery interface.
- After implementation: 33 tests passed, including both full-matrix checks.
- Independent review identified raw response bodies in error logs. Added a
  failing regression for both entrypoints, removed raw HTTP/malformed-response
  dumps while retaining error status/location, and verified the regression passes.
  Also clarified README row counts per personalized mode.
- Final verification: all 34 tests passed; both CLI help commands and
  `git diff --check` passed. No real model inference was performed.
- Work continued in place on `ksdani/topic-simulation-outputs` to preserve the
  existing uncommitted simulator changes. No commits or model runs were made.
