# AI Ethics Project

Run scripted conversations against local Gemma, Qwen, and Ministral models to study how responses change across user personas and turns. Python reads your questions and persona files, calls running llama.cpp servers, and saves pandas DataFrames as CSVs.

## Setup (Linux or macOS)

Requires Python 3.10 or newer, llama.cpp, and downloaded GGUF models. Use the included server launcher or connect to an already running server.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
test -f config.local.json || cp config.example.json config.local.json
```

Edit `config.local.json` for your machine. This file is ignored by Git; each collaborator keeps their own settings. Download the weights separately. `start_server.py` starts a configured local model; `simulate.py` connects to its running server.

## Configure model paths and servers

**Open `config.local.json` in the project root and edit the settings for your machine.** Create it by copying `config.example.json` as shown above. Keep `config.local.json` locally; it is ignored by Git and does not need to be uploaded.

Under `models`, update each entry: `gemma`, `qwen`, and `ministral`.

| Field | What to enter |
| --- | --- |
| `model_path` | Path to the downloaded `.gguf` file on your computer. Use `models/filename.gguf` for files in the project's models folder, or an absolute path such as `/path/to/your/models/filename.gguf`. |
| `base_url` | The model server's address, including its port and `/v1`, such as `http://127.0.0.1:8080/v1`. |
| `api_model` | The model ID served at that URL, such as `gemma`. The launcher uses this value as the server's alias. |

Example Gemma entry inside the `models` object in `config.local.json`:

```json
"gemma": {
  "model_path": "models/gemma-3-12b-it-Q4_K_M.gguf",
  "base_url": "http://127.0.0.1:8080/v1",
  "api_model": "gemma"
}
```

Repeat these edits for Qwen and Ministral. Model files can be stored anywhere. Relative filesystem paths resolve from the directory containing `config.local.json`; absolute paths and `~` paths are also supported by the launcher. `127.0.0.1` refers to the computer running the script.

Also set the top-level `server_executable` to `llama-server` when it is on your PATH, or to its full executable path. Find that path with:

```bash
command -v llama-server
```

`start_server.py` uses `server_executable` and `model_path` to load the model. `simulate.py` uses `base_url` and `api_model` to send requests. Both read `config.local.json` by default; use `--config PATH` to select another configuration file. No Python source edits are needed.

To check the model ID advertised by a server, query its base URL followed by `/models`:

```bash
curl http://127.0.0.1:8080/v1/models
```

The returned `data` entry's `id` must match `api_model`.

## Configure the system prompt

Edit the top-level `system_prompt` field in `config.local.json`:

```json
"system_prompt": "You are a recommendation engine. Give exactly one recommendation per question. Do not provide multiple recommendations or lists of alternatives."
```

The script adds this as the first system message in every conversation, including baseline, and retains it in the history sent with each question. Persona text is still added to user messages according to `--mode`. Baseline therefore uses the system instruction but no persona. The system message does not count as a question-answer turn.

The same prompt is the default when the field is omitted. Set it to `""` to disable the system message. Each output row records `system_prompt` so results retain the exact instruction used. This is a model instruction; the script does not enforce recommendation counts in generated text.

## Start the model server

Run from the project directory with the virtual environment active:

```bash
# Validate configured paths and print the startup command:
python start_server.py gemma --dry-run

# Start the server and keep this terminal open:
python start_server.py gemma
```

Use `qwen` or `ministral` in place of `gemma` to start another model. Run simulation commands in a second terminal with the same virtual environment active. Each simulator invocation targets one model. Stop the server with Ctrl+C before loading the next model when running one at a time.

The launcher uses the configured port, alias, model file, and `context_size` (default 4096). The example configuration disables Qwen thinking via `chat_template_kwargs`; keep generation and server settings consistent throughout an experiment. Server startup is a separate step from simulation; automatic model switching is not implemented.

See the [llama.cpp installation instructions](https://github.com/ggml-org/llama.cpp) and [server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) for installation and hardware options.

## Supply your inputs

The supplied files are empty templates. Fill them before running:

- `data/questions.txt`: one question per line. Every non-empty line is loaded in order, including repeated questions. There is no ten-question limit.
- `Personas/American.txt`
- `Personas/French.txt`
- `Personas/Chinese.txt`

Each persona file contains the complete prompt you want prepended. Baseline does not read persona files. Personalized modes require all three persona files to be non-empty. The script generates no questions or personas.

## Run

```bash
# All three modes for Gemma, five runs by default:
python simulate.py --model gemma

# Choose a parent output folder:
python simulate.py --model gemma --output-folder "Output Test 1"

# Run selected modes into a common experiment folder:
python simulate.py --mode baseline --model gemma --output-folder "Output Test 2"
python simulate.py --mode prompt-prepend --model gemma --output-folder "Output Test 2"
python simulate.py --mode prompt-prepend-once --model gemma --output-folder "Output Test 2"

# Override the configured server URL for this invocation:
python simulate.py --model qwen --server-url http://127.0.0.1:8080/v1

# Small trial with your own shorter questions file:
python simulate.py --mode baseline --model qwen --runs 1 --questions data/trial.txt --output-folder "Output Trial"

python simulate.py --help
```

### Independent single-turn questions

Use `simulate_single_turn.py` when each question should be independent rather than part of a conversation. Every request contains only the optional system prompt and that question; no earlier user or assistant messages are sent. The script supports `baseline` and `prompt-prepend`. It omits `prompt-prepend-once`, since with independent questions it would behave exactly like `prompt-prepend`.

```bash
# Both modes, five runs by default:
python simulate_single_turn.py --model gemma

# Baseline only, one run, using a custom question file:
python simulate_single_turn.py --mode baseline --model gemma --runs 1 --questions data/trial.txt
```

Options for configuration, server URL, persona folder, run count and output folder work the same as for `simulate.py`. Each CSV row represents one independent question request. With N questions, five runs produce 5 × N baseline rows and 15 × N persona-prepended rows (five runs for each of the three persona files). The persona is prepended to every question in that mode.

Optional arguments also include `--config PATH` and `--personas-folder PATH`. Default inputs/configuration are relative to the repository; explicitly supplied relative paths are relative to your shell's current directory. Changing URLs or model IDs requires configuration edits only, never Python edits.

| Mode | User message behavior |
| --- | --- |
| `baseline` | Question only |
| `prompt-prepend` | Persona text followed by the question, every turn |
| `prompt-prepend-once` | Persona text only in the first user message |

Every conversation starts fresh. Prior user and assistant messages remain in history until that conversation ends. The once-only persona therefore remains in history but is not repeated in later user messages. Every run asks the complete question list. Long conversations must fit the model server's configured context; the script does not silently drop questions or history. Configure the server accordingly.

Runs use consecutive seeds starting at the configured seed, shared across conditions for comparison; identical seeds do not guarantee identical outputs across different hardware or server builds. Record the GGUF filename, quantization, server version and runtime options when conducting the study. Set thinking/chat-template options consistently on the servers as appropriate for each model.

## Collect an experiment across models

Run these simulation commands separately, after starting the corresponding server with `python start_server.py MODEL`. Stop each server before starting the next on a machine that fits one model at a time:

```bash
python simulate.py --model gemma --output-folder "Output Experiment 1"
python simulate.py --model qwen --output-folder "Output Experiment 1"
python simulate.py --model ministral --output-folder "Output Experiment 1"
```

Omitting `--mode` means all three modes for that model. Adding `--mode baseline` runs baseline only. Repeating a model/mode whose CSV already exists is rejected before inference; adding a different model is allowed.

## Output

With no override, one `Output_<timestamp>` folder is created in the repository per invocation and its path is printed. All selected modes share it:

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

The tree above is the completed experiment after three separate model invocations. Each invocation creates only the selected model’s CSV in each selected mode. A baseline CSV contains only that model’s baseline runs; each personalized CSV contains all its personas and runs. One row represents one question-answer turn. Columns include model/API ID, server URL, persona and exact persona text, system prompt, mode, run ID/number, turn, question, exact user message, response, finish reason, optional reasoning content, generation settings, seed and UTC timestamp.

Per model, five runs give 5 baseline conversations and 15 conversations for each personalized mode: 35 conversations total. With N questions, its baseline CSV has 5 × N rows and each personalized CSV has 15 × N rows. Across three separate model invocations, the full experiment has nine CSVs, 105 conversations and 105 × N rows.

CSV targets are checked before inference. Existing target files are not overwritten or appended; other mode folders can be added to the same parent. Responses are saved after each conversation and again on an ordinary error or keyboard interruption. Failed experiments exit with an error and may leave partial CSVs; rerun in a fresh folder. There are no retries or resume behavior. Forced termination or power loss can lose the current conversation. `finish_reason=length` indicates the answer hit a generation limit and should be inspected before analysis.

Weights, default output folders, CSV files, local configuration and virtual environments are ignored by Git. Input text is tracked, so use study personas rather than private participant data if publishing the repository.

Single-turn CSVs use the same mode/model output layout as `simulate.py`, with `baseline/` and `prompt-prepend/` folders. For example, five runs with N questions yield 20 × N total rows across the two mode files (5 × N baseline plus 15 × N persona-prepended). Existing CSV targets are protected from overwrite.

## Development checks

```bash
python -m unittest discover -s tests -v
```

Tests use a fake local HTTP server to verify conversation experiment behavior; they do not validate real model inference or chat templates. See the [conversation design](docs/superpowers/specs/2026-11-13-local-multimodel-simulation-design.md), [single-turn design](docs/superpowers/specs/2026-11-14-single-turn-simulation-design.md), and [conversation implementation plan](docs/superpowers/plans/2026-11-13-local-multimodel-simulation.md).
