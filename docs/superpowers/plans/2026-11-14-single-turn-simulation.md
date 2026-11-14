# Single-Turn Simulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `simulate_single_turn.py` to ask each input question in an independent request.

**Architecture:** Follow `simulate.py` conventions for arguments, settings, endpoint validation, persona conditions, timestamps, and CSV persistence. Reuse its helpers where practical; each completion request contains the optional system prompt and exactly one user question, with no prior assistant or user history.

**Tech Stack:** Python 3.10+, urllib, argparse, pandas.

**Spec:** [../specs/2026-11-14-single-turn-simulation-design.md](../specs/2026-11-14-single-turn-simulation-design.md)

## Global Constraints

- Ask every non-empty question line in order, preserving duplicate entries.
- Support only `baseline` and `prompt-prepend`; once-only persona mode is redundant without conversation history.
- Keep the existing `simulate.py` behavior unchanged.
- Protect existing CSV targets and retain completed rows if a later request fails.

## Review Focus

- Empty question files must fail before inference, matching the existing runner.
- Baseline mode must not require persona files.
- Every per-question request must contain no history from another question.
- `prompt-prepend` must prefix the persona on each question.
- CSV output paths must not overwrite files from another run.

---

### Task 1: Implement and document the single-turn runner

**Files:**
- Create: `simulate_single_turn.py`
- Modify: `README.md`
- Reference: `simulate.py`

**Interfaces:**
- CLI follows `simulate.py`, with required `--model`, optional `--mode` (`baseline` or `prompt-prepend`), and the same config, question, persona, output, server URL, and run-count options.
- CSV follows existing output conventions under `baseline/<model>.csv` and `prompt-prepend/<model>.csv`.

- [x] Implement the standalone CLI and execution loop, reusing shared helpers from `simulate.py` where practical. Each request sends only optional system prompt and current question; persona is prefixed to each user message in `prompt-prepend` mode.
- [x] Preserve run IDs and metadata conventions, using separate run records for each question, and save accumulated rows after each completion and on failure.
- [x] Document when to use this script, its modes, invocation examples, and per-question CSV row counts in `README.md`.
- [x] Review the implementation against the review focus and the design spec.

Implementation review completed by reading the runner and README changes. Tests were not added or run.
