"""Run each file-driven question as an independent request to a llama.cpp server."""
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import simulate

MODES = simulate.MODES


def arguments(argv=None):
    return simulate.arguments(argv, description=__doc__)


def request_answer(question, persona, mode, model, settings, seed, question_number):
    prefix = persona if mode != "baseline" else ""
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
        raise RuntimeError(f"Invalid chat response for question {question_number}: expected choices[0].message.content to be a non-empty string.") from exc
    return dict(
        turn=question_number,
        question=question,
        user_message=user_message,
        response=answer,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def independent_requests(questions, persona, mode, model, settings, seed, on_request=None):
    for turn, question in enumerate(questions, 1):
        if on_request is not None:
            on_request(turn)
        yield request_answer(question, persona, mode, model, settings, seed, turn)


def run(args):
    return simulate.run_experiment(args, "single-turn", independent_requests)


def main(argv=None):
    args = arguments(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        print("Interrupted; server stopped and completed responses saved.", file=sys.stderr)
        return 130
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
