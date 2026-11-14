"""Run file-driven conversations against existing llama.cpp servers."""
import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parent
MODES = ("baseline", "prompt-prepend", "prompt-prepend-once")
PERSONAS = ("American", "French", "Chinese")
DEFAULT_SYSTEM_PROMPT = 'You are a recommendation engine. Give exactly one recommendation per question. Do not provide multiple recommendations or lists of alternatives.'


def read_questions(path):
    questions = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not questions:
        raise ValueError(f"No questions found in {path}; add one question per line.")
    return questions


def request_json(url, timeout, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        with exc:
            detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Server returned HTTP {exc.code} at {url}: {detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"Cannot reach {url}: {exc}") from exc


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, help="omit to run all three modes")
    parser.add_argument("--config", type=Path, default=ROOT / "config.local.json")
    parser.add_argument("--questions", type=Path, default=ROOT / "data/questions.txt")
    parser.add_argument("--personas-folder", type=Path, default=ROOT / "Personas")
    parser.add_argument("--output-folder", type=Path)
    parser.add_argument("--model", required=True, help="model key to run (exactly one model per invocation)")
    parser.add_argument("--server-url", help="override the endpoint for --model, e.g. http://127.0.0.1:8080/v1")
    parser.add_argument("--runs", type=positive_int, help="override runs per model/persona (default 5)")
    return parser.parse_args(argv)


def load_settings(args):
    if not args.config.is_file():
        raise ValueError(f"Configuration missing: {args.config}. Copy config.example.json to config.local.json and edit server URLs and model aliases.")
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError("Configuration must be a JSON object.")
    defaults = dict(runs=5, temperature=0.7, top_p=0.9, max_tokens=512, seed=42, timeout_seconds=600, system_prompt=DEFAULT_SYSTEM_PROMPT)
    settings = defaults | settings
    if not isinstance(settings["system_prompt"], str):
        raise ValueError("system_prompt must be a string; use an empty string to disable it.")
    if args.runs is not None:
        settings["runs"] = args.runs
    for field in ("runs", "max_tokens", "seed"):
        value = settings[field]
        if type(value) is not int or value < (0 if field == "seed" else 1):
            raise ValueError(f"Invalid {field}: {value}")
    for field in ("temperature", "top_p", "timeout_seconds"):
        value = settings[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Invalid {field}: {value}")
    if settings["temperature"] < 0 or not 0 < settings["top_p"] <= 1 or settings["timeout_seconds"] <= 0:
        raise ValueError("Require temperature >= 0, 0 < top_p <= 1 and timeout_seconds > 0.")
    models = settings.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("Configure at least one model under 'models'.")
    if args.model not in models:
        raise ValueError(f"Unknown model {args.model!r}; choose from {', '.join(models)}.")
    models = {args.model: models[args.model]}
    for name, model in models.items():
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
            raise ValueError(f"Model keys must be simple filename-safe names: {name!r}")
        if not isinstance(model, dict):
            raise ValueError(f"Model {name} must specify base_url and api_model.")
        if args.server_url:
            model["base_url"] = args.server_url
        url = model.get("base_url", "")
        if not isinstance(url, str):
            raise ValueError(f"Invalid base_url for {name}")
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError(f"Invalid base_url for {name}: {url}")
        model["base_url"] = url.rstrip("/")
        if not model["base_url"].endswith("/v1"):
            model["base_url"] += "/v1"
        if not isinstance(model.get("api_model"), str) or not model["api_model"].strip():
            raise ValueError(f"Set api_model for {name} to the model ID advertised by /v1/models.")
    return settings, models


def conversation(questions, persona, mode, model, settings, seed):
    messages = []
    if settings["system_prompt"].strip():
        messages.append({"role": "system", "content": settings["system_prompt"]})
    for turn, question in enumerate(questions, 1):
        prefix = persona if mode == "prompt-prepend" or (mode == "prompt-prepend-once" and turn == 1) else ""
        user_message = f"{prefix}\n\n{question}" if prefix else question
        messages.append({"role": "user", "content": user_message})
        payload = {
            "model": model["api_model"], "messages": messages, "stream": False,
            "temperature": settings["temperature"], "top_p": settings["top_p"],
            "max_tokens": settings["max_tokens"], "seed": seed,
        }
        response = request_json(model["base_url"] + "/chat/completions", settings["timeout_seconds"], payload)
        try:
            choice = response["choices"][0]
            answer = choice["message"]["content"]
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("Missing or empty assistant content")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid chat response at turn {turn}: {response}") from exc
        messages.append({"role": "assistant", "content": answer})
        yield dict(turn=turn, question=question, user_message=user_message, response=answer,
                   finish_reason=choice.get("finish_reason"),
                   reasoning_content=choice["message"].get("reasoning_content", ""),
                   timestamp=datetime.now(timezone.utc).isoformat())


def run(args):
    settings, models = load_settings(args)
    questions = read_questions(args.questions)
    modes = (args.mode,) if args.mode else MODES
    personas = {}
    if any(mode != "baseline" for mode in modes):
        for name in PERSONAS:
            path = args.personas_folder / f"{name}.txt"
            text = path.read_text(encoding="utf-8-sig").strip()
            if not text:
                raise ValueError(f"Persona file is empty: {path}")
            personas[name] = text
    output = (args.output_folder or ROOT / datetime.now().strftime("Output_%Y-%m-%d_%H-%M-%S_%f")).resolve()
    targets = {(mode, name): output / mode / f"{name}.csv"
               for mode in modes for name in models}
    for target in set(targets.values()):
        if target.exists():
            raise ValueError(f"Output already exists: {target}. Choose another --output-folder.")
    if args.output_folder is None and output.exists():
        raise ValueError(f"Generated output folder already exists: {output}")
    # Check only the selected endpoint; other configured models need not be running.
    for name, model in models.items():
        available = request_json(model["base_url"] + "/models", settings["timeout_seconds"])
        ids = [entry.get("id") for entry in available.get("data", [])]
        if model["api_model"] not in ids:
            raise ValueError(f"{name}: configured api_model {model['api_model']!r} is not served at {model['base_url']}; available: {ids}")
    output.mkdir(parents=True, exist_ok=args.output_folder is not None)
    for target in set(targets.values()):
        target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {output}", flush=True)
    records = {target: [] for target in targets.values()}
    try:
        for name, model in models.items():
            for mode in modes:
                selected_personas = {"baseline": ""} if mode == "baseline" else personas
                target = targets[mode, name]
                for persona_id, persona in selected_personas.items():
                    for run_number in range(1, settings["runs"] + 1):
                        seed = settings["seed"] + run_number - 1
                        print(f"{name} | {mode} | {persona_id} | run {run_number}/{settings['runs']}", flush=True)
                        for row in conversation(questions, persona, mode, model, settings, seed):
                            row.update(model=name, api_model=model["api_model"], server_url=model["base_url"],
                                       persona=persona_id, persona_prompt=persona, system_prompt=settings["system_prompt"], mode=mode,
                                       run_id=f"{name}/{mode}/{persona_id}/{run_number}", run_number=run_number,
                                       seed=seed, temperature=settings["temperature"], top_p=settings["top_p"],
                                       max_tokens=settings["max_tokens"])
                            records[target].append(row)
                        pd.DataFrame(records[target]).to_csv(target, index=False)
    finally:
        # Keep completed responses even if a later turn fails; errors still stop the run.
        for target, rows in records.items():
            if rows:
                pd.DataFrame(rows).to_csv(target, index=False)
    total = sum(len(rows) for rows in records.values())
    print(f"Completed: {total} turns in {len(records)} CSV files.", flush=True)
    return output


def main(argv=None):
    args = arguments(argv)
    try:
        run(args)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}\nExperiment did not complete. Any saved CSVs contain partial results; use a fresh folder when rerunning.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
