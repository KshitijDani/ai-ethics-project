"""Run each file-driven question as an independent request to a llama.cpp server."""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import simulate

ROOT = Path(__file__).resolve().parent
MODES = ("baseline", "prompt-prepend")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, help="omit to run both modes")
    parser.add_argument("--config", type=Path, default=ROOT / "config.local.json")
    parser.add_argument("--questions", type=Path, default=ROOT / "data/questions.txt")
    parser.add_argument("--personas-folder", type=Path, default=ROOT / "Personas")
    parser.add_argument("--output-folder", type=Path)
    parser.add_argument("--model", required=True, help="model key to run (exactly one model per invocation)")
    parser.add_argument("--server-url", help="override the endpoint for --model, e.g. http://127.0.0.1:8080/v1")
    parser.add_argument("--runs", type=simulate.positive_int, help="override runs per model/persona (default 5)")
    return parser.parse_args(argv)


def request_answer(question, persona, mode, model, settings, seed, question_number):
    prefix = persona if mode == "prompt-prepend" else ""
    user_message = f"{prefix}\n\n{question}" if prefix else question
    messages = []
    if settings["system_prompt"].strip():
        messages.append({"role": "system", "content": settings["system_prompt"]})
    messages.append({"role": "user", "content": user_message})
    payload = {
        "model": model["api_model"], "messages": messages, "stream": False,
        "temperature": settings["temperature"], "top_p": settings["top_p"],
        "max_tokens": settings["max_tokens"], "seed": seed,
    }
    response = simulate.request_json(
        model["base_url"] + "/chat/completions", settings["timeout_seconds"], payload
    )
    try:
        choice = response["choices"][0]
        message = choice["message"]
        answer = message["content"]
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Missing or empty assistant content")
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid chat response for question {question_number}: {response}") from exc
    return dict(
        turn=question_number,
        question=question,
        user_message=user_message,
        response=answer,
        finish_reason=choice.get("finish_reason"),
        reasoning_content=message.get("reasoning_content", ""),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def run(args):
    settings, models = simulate.load_settings(args)
    questions = simulate.read_questions(args.questions)
    modes = (args.mode,) if args.mode else MODES
    personas = {}
    if "prompt-prepend" in modes:
        for name in simulate.PERSONAS:
            path = args.personas_folder / f"{name}.txt"
            text = path.read_text(encoding="utf-8-sig").strip()
            if not text:
                raise ValueError(f"Persona file is empty: {path}")
            personas[name] = text

    output = (args.output_folder or ROOT / datetime.now().strftime("Output_%Y-%m-%d_%H-%M-%S_%f")).resolve()
    targets = {(mode, name): output / mode / f"{name}.csv" for mode in modes for name in models}
    for target in set(targets.values()):
        if target.exists():
            raise ValueError(f"Output already exists: {target}. Choose another --output-folder.")
    if args.output_folder is None and output.exists():
        raise ValueError(f"Generated output folder already exists: {output}")

    for name, model in models.items():
        available = simulate.request_json(model["base_url"] + "/models", settings["timeout_seconds"])
        ids = [entry.get("id") for entry in available.get("data", [])]
        if model["api_model"] not in ids:
            raise ValueError(
                f"{name}: configured api_model {model['api_model']!r} is not served at "
                f"{model['base_url']}; available: {ids}"
            )

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
                        run_id = f"{name}/{mode}/{persona_id}/{run_number}"
                        for question_number, question in enumerate(questions, 1):
                            print(
                                f"{name} | {mode} | {persona_id} | run {run_number}/"
                                f"{settings['runs']} | question {question_number}/{len(questions)}",
                                flush=True,
                            )
                            row = request_answer(
                                question, persona, mode, model, settings, seed, question_number
                            )
                            row.update(
                                model=name,
                                api_model=model["api_model"],
                                server_url=model["base_url"],
                                persona=persona_id,
                                persona_prompt=persona,
                                system_prompt=settings["system_prompt"],
                                mode=mode,
                                run_id=f"{run_id}/question-{question_number}",
                                run_number=run_number,
                                seed=seed,
                                temperature=settings["temperature"],
                                top_p=settings["top_p"],
                                max_tokens=settings["max_tokens"],
                            )
                            records[target].append(row)
                            pd.DataFrame(records[target]).to_csv(target, index=False)
    finally:
        for target, rows in records.items():
            if rows:
                pd.DataFrame(rows).to_csv(target, index=False)

    total = sum(len(rows) for rows in records.values())
    print(f"Completed: {total} independent question requests in {len(records)} CSV files.", flush=True)
    return output


def main(argv=None):
    args = arguments(argv)
    try:
        run(args)
    except (ValueError, OSError, RuntimeError) as exc:
        print(
            f"Error: {exc}\nExperiment did not complete. Any saved CSVs contain partial results; "
            "use a fresh folder when rerunning.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
