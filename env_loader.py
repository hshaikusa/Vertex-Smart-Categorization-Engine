"""Load OpenAI / Vertex settings from a local .env into os.environ.

`.env` wins over a pre-existing process environment so the file you edit
is what `run.py` actually uses. Call `load_env()` before constructing the
OpenAI client.
"""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"

# Keys this project reads. Values live in `.env`, not in this file.
MANAGED_KEYS = ("OPENAI_API_KEY", "VERTEX_MODEL", "VERTEX_CACHE_DIR")


# Purpose: read a simple KEY=VALUE .env file into os.environ - lets the project's API
#   key / model / cache-dir settings live in an untracked .env file instead of real shell
#   environment variables. Blank lines and lines starting with "#" are skipped; quoted
#   values have their surrounding quotes stripped. Called once automatically at import
#   time (see the module-level load_env() call at the bottom of this file), so simply
#   `import env_loader` anywhere before building the OpenAI client is enough.
# Input: path - which .env file to read (default: ENV_PATH, a ".env" next to this file);
#   override - if True (default), values in the .env file replace any value already in
#   os.environ for the same key; if False, an existing process env var wins.
# Output: dict[str, str] - the keys that ended up set (either freshly loaded, or already
#   present when override=False), for callers that want to see what was picked up. Also
#   has the side effect of mutating os.environ. Returns {} if the .env file doesn't exist.
def load_env(path: Path | None = None, override: bool = True) -> dict[str, str]:
    path = Path(path) if path is not None else ENV_PATH
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip("'").strip('"')
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = val
        loaded[key] = os.environ[key]
    return loaded


load_env()
