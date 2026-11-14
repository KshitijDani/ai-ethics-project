"""Start one llama.cpp server using the simulator's local configuration."""
import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent


def local_path(value, config_dir):
    path = Path(value).expanduser()
    return path if path.is_absolute() else config_dir / path


def server_command(config_path, model_name):
    config_path = config_path.resolve()
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError("Configuration must be a JSON object.")
    models = settings.get("models", {})
    if not isinstance(models, dict) or model_name not in models:
        raise ValueError(f"Model {model_name!r} is not configured.")
    model = models[model_name]
    if not isinstance(model, dict):
        raise ValueError("Model configuration must be an object.")
    executable = settings.get("server_executable", "llama-server")
    if not isinstance(executable, str) or not executable.strip():
        raise ValueError("server_executable must be a program name or executable path.")
    if "/" in executable or executable.startswith("~"):
        executable = str(local_path(executable, config_path.parent))
    executable = shutil.which(executable)
    if not executable:
        raise ValueError("Cannot find llama-server. Set server_executable in your local configuration.")
    model_file = model.get("model_path")
    if not isinstance(model_file, str) or not model_file.strip():
        raise ValueError(f"Set models.{model_name}.model_path to your downloaded GGUF file.")
    model_path = local_path(model_file, config_path.parent).resolve()
    if not model_path.is_file():
        raise ValueError(f"Model file not found: {model_path}")
    alias = model.get("api_model")
    if not isinstance(alias, str) or not alias.strip():
        raise ValueError(f"Set models.{model_name}.api_model to a model alias.")
    base_url = model.get("base_url", "")
    if not isinstance(base_url, str):
        raise ValueError("base_url must be a URL string.")
    url = urlparse(base_url)
    if (url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost")
            or url.username or url.password or url.query or url.fragment
            or url.path.rstrip("/") not in ("", "/v1")):
        raise ValueError("The launcher needs a local http://127.0.0.1:PORT/v1 or http://localhost:PORT/v1 URL. For an existing server, run simulate.py directly.")
    if url.port is None:
        raise ValueError("Include an explicit port in base_url, such as :8080.")
    context_size = settings.get("context_size", 4096)
    if type(context_size) is not int or context_size <= 0:
        raise ValueError("context_size must be a positive integer.")
    command = [executable, "--model", str(model_path), "--alias", alias,
               "--host", "127.0.0.1", "--port", str(url.port),
               "--ctx-size", str(context_size), "--parallel", "1"]
    template_kwargs = model.get("chat_template_kwargs")
    if template_kwargs is not None:
        if not isinstance(template_kwargs, dict):
            raise ValueError("chat_template_kwargs must be a JSON object.")
        command += ["--chat-template-kwargs", json.dumps(template_kwargs)]
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="model key in config.local.json")
    parser.add_argument("--config", type=Path, default=ROOT / "config.local.json")
    parser.add_argument("--dry-run", action="store_true", help="validate paths and print the command without loading a model")
    args = parser.parse_args(argv)
    try:
        command = server_command(args.config, args.model)
        print(shlex.join(command), flush=True)
        if args.dry_run:
            return 0
        # No shell: paths containing spaces remain single arguments.
        return subprocess.run(command).returncode
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
