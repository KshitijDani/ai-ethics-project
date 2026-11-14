# Local simulation implementation plan

Goal: implement the approved file-driven experiment with three modes and mode-specific CSV outputs.

Architecture: one Python script reads questions, persona files and local JSON settings. It calls running llama.cpp chat endpoints sequentially, retains messages within each conversation, and exports pandas DataFrames.

Decision from the implementation request: defer executable discovery, server process management, and model downloads. Use already running endpoints now. No collaborator edits Python source. Exact server model aliases are checked before inference to avoid mislabeled experiments.

Spec: ../specs/2026-11-13-local-multimodel-simulation-design.md

- [x] Add configuration example, empty input files, dependencies, Git ignore rules.
- [x] Implement optional mode, model filter, server override, output folder, run count and input overrides.
- [x] Implement ordered input parsing, fresh conversation state, user-level persona injection and sequential requests.
- [x] Export baseline/model CSVs by mode, protect existing files, retain completed turns on failure.
- [x] Verify with a local fake HTTP server: >10 questions, all modes, history isolation, counts, CSV text fidelity, folder reuse, collisions, model mismatch and HTTP failure.
- [x] Update README with real installation and usage commands; review implementation.

No real inference validation is possible without installed models and running servers. Fake-server checks verify transport and experiment orchestration only.

Validation: six unittest integration checks passed against a fake local HTTP server; CLI help checked. Independent review found no concrete correctness issues. Real inference was not run.

## Per-model execution update

- [x] Require --model and contact only its endpoint.
- [x] Store baseline per model, matching the personalized mode structure.
- [x] Permit sequential model invocations into a shared parent while protecting existing target files.
- [x] Update README/design and cover isolated endpoints, shared folder preservation and required selector in tests.
