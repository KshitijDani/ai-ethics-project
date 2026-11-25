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

Edit `config.local.json` for your machine. This file is ignored by Git; each collaborator keeps their own settings. Download the weights separately. `start_server.py` starts a configured local model. Both simulation scripts can connect to existing servers or start and stop them automatically with `--manage-servers`.

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

The same prompt is the default when the field is omitted. Set it to `""` to disable the system message. The analysis CSV omits the system prompt and generation settings; keep your configuration with your study records. This is a model instruction; the script does not enforce recommendation counts in generated text.

## Start the model server

Run from the project directory with the virtual environment active:

```bash
# Validate configured paths and print the startup command:
python start_server.py gemma --dry-run

# Start the server and keep this terminal open:
python start_server.py gemma
```

Use `qwen` or `ministral` in place of `gemma` to start another model. Run simulation commands in a second terminal with the same virtual environment active. When managing servers manually, stop the server with Ctrl+C before loading the next model. Use `--model all --manage-servers` on either simulator to handle the full sequence automatically in one terminal.

The launcher uses the configured port, alias, model file, and `context_size` (default 4096). The example configuration disables Qwen thinking via `chat_template_kwargs`; keep generation and server settings consistent throughout an experiment. With `--manage-servers`, each simulator uses the same launcher configuration to load, wait for, and stop one model at a time.

See the [llama.cpp installation instructions](https://github.com/ggml-org/llama.cpp) and [server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) for installation and hardware options.

## Supply your inputs

Supply newline-separated questions in topic files under `data/`, for example:

```text
data/
  questions_books.txt
  questions_food.txt
  questions_hobby.txt
  questions_movies.txt
```

Both simulators load every direct-child file matching `questions_<topic>.txt`, sorted by filename. Each non-empty line is one predefined question. Surrounding whitespace and an optional UTF-8 BOM are removed; question order and repeated questions are preserved. Unrelated files (including the legacy `data/questions.txt`) and nested directories are ignored. No matching files, an empty topic name, or a matching file with no questions is an error before any server starts.

The included four topic files each contain ten questions, using the same files for both experiments. The scripts support any number of topics and questions for smaller trials. They generate no questions. The legacy mixed-topic file is ignored. Use `--questions-folder PATH` for another input directory; this replaces the old `--questions FILE` option.

Personalized modes read these non-empty files:

- `Personas/American.txt`
- `Personas/French.txt`
- `Personas/Chinese.txt`

Each contains the complete persona prompt. Baseline does not need persona files.

## Run

Run the entire multi-turn experiment across all configured models and all three modes with one command:

```bash
python simulate.py --model all --manage-servers
```

The runner starts and stops each server automatically and saves all results in one timestamped output folder. Five runs per condition are used by default. Execution order is model → mode → topic → persona → repetition → question. For each model, baseline finishes across all topics before prompt-prepend, then prompt-prepend-once. Within a mode, all personas and repetitions for a topic finish before the next topic. Conversation history is retained within a topic/persona/repetition and reset between topics, runs, personas, modes, and models. Fill all three persona files before running all modes. For baseline only:

```bash
python simulate.py --mode baseline --model all --manage-servers
```

Both simulators accept `--runs`, `--output-folder`, and `--startup-timeout` with managed servers. Each server gets a log under `multi-turn/server-logs/` or `single-turn/server-logs/` in the output folder. Stop any manually running servers on the configured ports before using managed startup. Errors or Ctrl+C stop the owned server and preserve completed responses.

For manually managed servers, the existing per-model commands remain available:

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

# Small trial with a directory of shorter topic files:
python simulate.py --mode baseline --model qwen --runs 1 --questions-folder trial-data --output-folder "Output Trial"

python simulate.py --help
```

### Independent single-turn questions

Use `simulate_single_turn.py` when each question should be independent rather than part of a conversation. Every request contains only the optional system prompt and that question; no earlier user or assistant messages are sent. It reads the same topic files and supports all three mode labels. In both personalized modes, persona text is prepended to every independent request. `prompt-prepend-once` is therefore an equivalent condition to `prompt-prepend` in the single-turn experiment; it does not personalize only the first question.

```bash
# All three mode labels, five runs by default:
python simulate_single_turn.py --model gemma

# Baseline only, one run, using a custom topic directory:
python simulate_single_turn.py --mode baseline --model gemma --runs 1 --questions-folder trial-data
```

To start and stop the local servers automatically and run baseline questions across all configured models with one command:

```bash
python simulate_single_turn.py --mode baseline --model all --manage-servers
```

This loads one model at a time in configuration order, waits for its `/health` endpoint, runs its questions, and stops its server before loading the next. It uses five runs by default and creates one shared timestamped output folder containing `single-turn/baseline/<model>.csv` and `single-turn/server-logs/<model>.log`. Add `--runs 1` for a short trial or `--output-folder "Output Baseline Single Turn"` to name the folder. Omit `--mode` to run all three single-turn mode labels, or replace `all` with one model key.

Managed startup requires valid `server_executable`, GGUF paths, and local server URLs in your configuration. Stop any manually running server on the configured ports first. The runner never stops unrelated servers. Startup has a 600-second limit per model, adjustable with `--startup-timeout SECONDS`. Errors or Ctrl+C stop the owned server and preserve completed responses; rerun into a fresh output folder. `--server-url` is only supported for a single externally managed model. Without `--manage-servers`, servers must already be running.

Options for configuration, question folder, server URL, persona folder, run count and output folder work the same as for `simulate.py`. Each CSV row represents one independent question request. With N questions, five runs produce 5 × N baseline rows and 15 × N rows per personalized mode (five runs for each of the three persona files), totaling 35 × N rows per model when all three modes run. The persona is prepended to every question in either personalized mode.

Optional arguments also include `--config PATH` and `--personas-folder PATH`. Default inputs/configuration are relative to the repository; explicitly supplied relative paths are relative to your shell's current directory. Changing URLs or model IDs requires configuration edits only, never Python edits.

| Mode | User message behavior |
| --- | --- |
| `baseline` | Question only |
| `prompt-prepend` | Persona text followed by the question, every turn |
| `prompt-prepend-once` | Persona text only in the first user message |

The table above describes multi-turn behavior. Every multi-turn conversation starts fresh. Prior user and assistant messages remain in history until that conversation ends. The once-only persona therefore remains in history but is not repeated in later user messages. Every repetition asks the complete question list for its topic. Long conversations must fit the model server's configured context; the script does not silently drop questions or history. Configure the server accordingly.

Runs use consecutive seeds starting at the configured seed, shared across conditions for comparison; identical seeds do not guarantee identical outputs across different hardware or server builds. Record the GGUF filename, quantization, server version and runtime options when conducting the study. Set thinking/chat-template options consistently on the servers as appropriate for each model.

## Collect an experiment across models

Alternatively, run these simulation commands separately with manually managed servers, after starting the corresponding server with `python start_server.py MODEL`. Stop each server before starting the next on a machine that fits one model at a time:

```bash
python simulate.py --model gemma --output-folder "Output Experiment 1"
python simulate.py --model qwen --output-folder "Output Experiment 1"
python simulate.py --model ministral --output-folder "Output Experiment 1"
```

Omitting `--mode` means all three modes for that model. Adding `--mode baseline` runs baseline only. Repeating a model/mode whose CSV already exists is rejected before inference; adding a different model is allowed.

## Output

With no override, one `Output_<timestamp>` folder is created per invocation. An explicit `--output-folder` lets separate model/mode invocations and both experiment types share a parent without mixing their CSVs:

```text
Output_<timestamp>/
  multi-turn/
    baseline/<model>.csv
    prompt-prepend/<model>.csv
    prompt-prepend-once/<model>.csv
    server-logs/<model>.log
  single-turn/
    baseline/<model>.csv
    prompt-prepend/<model>.csv
    prompt-prepend-once/<model>.csv
    server-logs/<model>.log
```

Only the selected experiment, models, and modes are created. Old output folders are not migrated. Each CSV includes all topics for one model/mode and all relevant personas. One row represents one question-answer request. Both scripts use these columns in order:

```text
model,mode,persona,question_file_name,question_number,run_id,run_number,turn,question,user_message,response,timestamp
```

`question_file_name` is the source basename. `question_number` combines the filename's topic suffix with its one-based non-empty-line position: `questions_books.txt` produces `books_1`, `books_2`, etc. `turn` is the numeric position within that topic, starting at 1. `question` stores the original stripped question; `user_message` includes any persona prefix; `response` is the assistant answer; `timestamp` is UTC.

`run_id` is `<experiment>/<model>/<mode>/<topic>/<persona>/<run_number>`, for example `multi-turn/gemma/prompt-prepend/books/American/1`. Baseline uses `baseline` as its persona. In single-turn output, this ID groups a repetition batch of independent requests; it does not imply shared history. `run_number` distinguishes repetitions. IDs are scoped to an output experiment; question text should be checked when combining results from different input versions.

The CSV omits finish reason, reasoning content, API model, server URL, persona prompt, system prompt, seed, temperature, `top_p`, and `max_tokens`. Generation settings still come from the configuration and existing CLI overrides.

With four topics, ten questions each, three models, three personas, and five repetitions, each script produces nine CSVs and 4,200 response rows: 600 baseline rows plus 1,800 rows for each personalized mode. Multi-turn and single-turn results remain separate. Analysis, embedding models, baseline matching, and plots will be added later.

CSV targets are checked before inference. Existing files are not overwritten or appended; different models or modes can be added to the same parent. Multi-turn responses are saved after each conversation; single-turn responses are saved after each request. Both also save completed responses on an ordinary error or keyboard interruption. Failed experiments may leave partial CSVs; rerun in a fresh folder. There are no retries or resume behavior. Forced termination or power loss can lose unsaved responses.

Weights, default output folders, CSV files, local configuration and virtual environments are ignored by Git. Input text is tracked, so use study personas rather than private participant data if publishing the repository.

### Progress logs

Both scripts always print file extraction counts, question ID ranges, the total input count, output location, experiment summary, model/mode/topic starts, and each repetition's start and completion. Messages are flushed immediately:

```text
Extracted 10 questions from questions_books.txt | topic=books | IDs=books_1..books_10
Extracted 10 questions from questions_food.txt | topic=food | IDs=food_1..food_10
Loaded 40 questions from 4 files.
Starting model=gemma | experiment=multi-turn
Starting mode=prompt-prepend | model=gemma
Starting topic=books | file=questions_books.txt | questions=10
Starting run | model=gemma | mode=prompt-prepend | topic=books | persona=American | run=1/5
Completed run | model=gemma | mode=prompt-prepend | topic=books | persona=American | run=1/5 | responses=10
```

Add `--verbose` to list every extracted question's filename, ID, and full text once during loading, plus each request's start and completion with its condition, repetition, and question ID:

```bash
python simulate.py --mode prompt-prepend --model all --manage-servers --verbose
python simulate_single_turn.py --mode prompt-prepend --model all --manage-servers --verbose
```

Verbose console logs do not include assistant answers or persona prompt text. Server diagnostics stay in the server log files. Failed or interrupted runs report the error without a successful run or experiment completion message.

## Development checks

```bash
python -m unittest discover -s tests -v
```

Tests use a fake local HTTP server to verify conversation experiment behavior; they do not validate real model inference or chat templates. See the [conversation design](docs/superpowers/specs/2026-11-13-local-multimodel-simulation-design.md), [single-turn design](docs/superpowers/specs/2026-11-14-single-turn-simulation-design.md), and [conversation implementation plan](docs/superpowers/plans/2026-11-13-local-multimodel-simulation.md).
