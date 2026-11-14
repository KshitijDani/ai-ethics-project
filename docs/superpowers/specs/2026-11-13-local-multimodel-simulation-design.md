# Local Multi-Model Conversation Simulation Design

## Purpose and scope

Study model responses across user personas and conversation turns using locally served GGUF models. Simulation and CSV collection are implemented; analysis and automatic server switching are outside this phase.

## Inputs

- data/questions.txt: read every non-empty UTF-8 line into a Python list, preserving order and duplicate questions. No fixed question count. Empty input is an error.
- Personas/American.txt, Personas/French.txt, Personas/Chinese.txt: user-authored complete persona prompts. Baseline does not require these files.
- config.local.json: ignored machine-specific configuration, copied from config.example.json. Each model has base_url and api_model. The launcher additionally uses model_path, server_executable and context_size. Relative filesystem paths resolve beside the configuration file; absolute paths and home-directory paths are supported by the launcher.

## One model per invocation

--model is required. Only the selected model is validated and contacted; other configured servers need not be running. The user starts one server, conducts its experiment, stops it, and repeats for the next model.

The optional --mode argument selects baseline, prompt-prepend, or prompt-prepend-once. Omission runs all three modes for the selected model.

```bash
python simulate.py --model gemma --output-folder "Output Experiment 1"
python simulate.py --mode baseline --model qwen --output-folder "Output Experiment 1"
```

## Conversation behavior

- Default five runs per condition; --runs overrides this.
- Each run asks every loaded question sequentially and retains user/assistant history. History resets between conversations, personas and modes.
- Baseline: question only, no persona.
- prompt-prepend: prepend persona to every user message.
- prompt-prepend-once: prepend persona only to the first user question; that message remains in history.
- Store the response before asking the next question. No inference retries.
- Explicit generation settings and consecutive run seeds are recorded in the output.

## Output

--output-folder chooses a parent experiment directory. Without it, create Output_<timestamp> once per invocation and print its path. Reuse a chosen parent across separate model invocations to collect a complete experiment.

```text
Output_<timestamp>/
  baseline/
    gemma.csv
    qwen.csv
    ministral.csv
  prompt-prepend/
    gemma.csv
    qwen.csv
    ministral.csv
  prompt-prepend-once/
    gemma.csv
    qwen.csv
    ministral.csv
```

Only the selected model and modes produce files. Each model's baseline file includes its baseline runs; each personalized file includes all three personas and all runs. One row per question-answer pair, including model, persona, mode, run and turn identifiers, original question, actual user message, response, finish reason and generation settings. Export via pandas without an index.

At five runs, each model produces 5 baseline conversations plus 15 per personalized mode: 35 total. All three models together produce 105 conversations, nine CSVs and 105 × N rows for N questions. Selected target paths must not already exist; unrelated model/mode files do not block an invocation. Legacy baseline.csv files are preserved, not migrated or merged. Save completed responses on ordinary errors; partial experiments must be rerun into a fresh folder.

## Server setup

start_server.py MODEL starts a single foreground llama.cpp server from configured local executable/weight paths and alias/port. --dry-run validates paths and prints the command. Existing-server users need only base_url and api_model and can skip the launcher. simulate.py never starts or stops servers. It checks the selected endpoint's advertised model IDs before inference. Weights and generated CSVs are excluded from Git.

Model downloads are Gemma 3 12B IT, Qwen3 14B and Ministral 3 14B Instruct 2512 in Q4_K_M GGUF format. Exact paths vary by collaborator. The README documents setup for macOS and Linux.

## Configurable system instruction

The top-level system_prompt setting defaults to: "You are a recommendation engine. Give exactly one recommendation per question. Do not provide multiple recommendations or lists of alternatives." Both example and local configuration include it. A non-empty prompt becomes one initial system message in every conversation and remains in history across all turns and modes. It does not count as a turn. Baseline has this system instruction and no persona. An empty string disables it; non-string values are rejected. Every CSV row records the exact configured prompt. The instruction does not guarantee model compliance.
