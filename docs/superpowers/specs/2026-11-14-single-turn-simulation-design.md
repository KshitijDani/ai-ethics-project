# Single-Turn Simulation Design

## Intent and scope

Add a separate `simulate_single_turn.py` runner for experiments where each question in `data/questions.txt` is asked in its own fresh model request. This supports measuring isolated answers without conversation history while retaining the existing experiment's model, persona, mode, and run conventions.

## Goals and constraints

- Read every non-empty question line in order, including duplicates.
- Support `baseline` and `prompt-prepend`. Do not offer `prompt-prepend-once`, because with no conversation history it would be identical to `prompt-prepend`.
- Reuse the existing configuration, selected-model validation, generation settings, `--runs`, endpoint check, and output conventions where practical.
- Save one CSV row per question request, with fields consistent with the existing simulator and a distinct run identifier.
- Do not change `simulate.py` conversation behavior.

## Approaches

- Add a single-turn flag to `simulate.py`: rejected because the user wants another script and separate commands make the history behavior explicit.
- Create `simulate_single_turn.py` with baseline and persona-prepended modes, using the existing shared conventions: selected because it preserves current workflows while isolating each question request and avoids a redundant once-only mode.

## Builder calls

Keep command-line options and output layout aligned with `simulate.py`; give the new script its own default timestamped output folder, while allowing an explicit shared output parent subject to the same no-overwrite checks. No server switching, retries, or resume behavior is added.
