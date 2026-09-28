"""Thin OpenAI wrapper: structured outputs, retries, disk cache, token accounting."""
import hashlib, json, os, time, threading
from pathlib import Path

CACHE_DIR = Path(os.environ.get("VERTEX_CACHE_DIR", ".llm_cache"))


class LLMError(Exception):
    pass


# Purpose: build the raw request body for one chat.completions call - shared by the live
#   OpenAIJSONClient path and the OpenAI Batch API path (selector/batch.py), so both
#   submit byte-identical requests. Drops the "temperature" field for reasoning models
#   (gpt-5/o1/o3/o4), which reject it.
# Input: model - the model name; temperature - sampling temperature; system/user - the
#   two prompt strings; schema - the strict JSON response_schema dict.
# Output: dict - the request body ready to pass to client.chat.completions.create(**body).
def build_request(model: str, temperature: float, system: str, user: str, schema: dict) -> dict:
    """Body of a chat.completions request; shared by the live client and the Batch API path."""
    body = dict(model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={"type": "json_schema", "json_schema": schema})
    if not model.startswith(("gpt-5", "o1", "o3", "o4")):   # reasoning models reject temperature
        body["temperature"] = temperature
    return body


class OpenAIJSONClient:
    # Purpose: construct the OpenAI client wrapper for one run - picks the model (arg,
    #   then VERTEX_MODEL env var, then a hardcoded default), builds a real openai.OpenAI
    #   client unless one was injected (tests inject a fake one), sets up the usage
    #   counters, and creates the on-disk cache directory when caching is enabled.
    # Input: model - model name override, or None to use the env var/default; temperature
    #   - sampling temperature (0.0 = deterministic); max_retries - retry attempts on
    #   failure; timeout - per-call HTTP timeout in seconds; use_cache - whether to read/
    #   write the on-disk response cache; client - an existing OpenAI client to reuse
    #   (mainly for tests), or None to build a real one.
    # Output: None (constructs self.model/.temperature/.max_retries/.use_cache/.client/
    #   .usage/._lock, and the cache directory on disk).
    def __init__(self, model: str | None = None, temperature: float = 0.0, max_retries: int = 4,
                 timeout: float = 60.0, use_cache: bool = True, client=None):
        self.model = model or os.environ.get("VERTEX_MODEL", "gpt-4.1-mini")
        self.temperature = temperature
        self.max_retries = max_retries
        self.use_cache = use_cache
        if client is None:
            from openai import OpenAI  # imported lazily so offline mode needs no key
            client = OpenAI(timeout=timeout)
        self.client = client
        self.usage = {"prompt": 0, "completion": 0, "cached_prompt": 0, "calls": 0, "cache_hits": 0, "retries": 0}   # cached_prompt = tokens served from OpenAI prompt caching (discounted)
        self._lock = threading.Lock()
        if use_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Purpose: compute the on-disk cache filename for one exact request - a SHA-256 hash
    #   of the model, temperature, both prompt strings, the schema, and (when repeat > 1)
    #   the sample index. Identical requests always land on the same file; any change to
    #   any of these produces a completely different filename. Sample 0 is left out of
    #   the hashed parts so pre-existing single-call cache entries stay valid.
    # Input: system/user - the two prompt strings; schema - the response_schema dict;
    #   sample - which independent sample this is (0 for a normal single call).
    # Output: a pathlib.Path - CACHE_DIR / "<64 hex chars>.json" (may or may not exist yet).
    def _key(self, system, user, schema, sample=0):
        parts = [self.model, self.temperature, system, user, schema] + ([sample] if sample else [])   # sample 0 keeps old cache keys valid
        h = hashlib.sha256(json.dumps(parts, sort_keys=True).encode())
        return CACHE_DIR / (h.hexdigest() + ".json")

    # Purpose: get one structured JSON response for a (system, user, schema) request -
    #   serves it from the on-disk cache when available (zero API cost), otherwise calls
    #   the live API with exponential retry backoff (1s/2s/4s/8s/16s, capped) on any
    #   exception (rate limit, timeout, bad JSON, a model refusal), and writes a
    #   successful response to the cache for next time.
    # Input: system/user - the two prompt strings; schema - the strict response_schema
    #   dict; sample - which independent sample this is (folded into the cache key).
    # Output: dict - the parsed JSON response matching schema. Raises LLMError if every
    #   retry attempt fails.
    def complete_json(self, system: str, user: str, schema: dict, sample: int = 0) -> dict:
        path = self._key(system, user, schema, sample) if self.use_cache else None
        if path and path.exists():
            with self._lock:
                self.usage["cache_hits"] += 1
            return json.loads(path.read_text(encoding="utf-8"))
        kwargs = build_request(self.model, self.temperature, system, user, schema)
        last = None
        for attempt in range(self.max_retries + 1):
            try:
                r = self.client.chat.completions.create(**kwargs)
                msg = r.choices[0].message
                if getattr(msg, "refusal", None):
                    raise LLMError(f"model refusal: {msg.refusal}")
                out = json.loads(msg.content)
                with self._lock:
                    self.usage["calls"] += 1
                    self.usage["prompt"] += getattr(r.usage, "prompt_tokens", 0)
                    self.usage["completion"] += getattr(r.usage, "completion_tokens", 0)
                    self.usage["cached_prompt"] += getattr(getattr(r.usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
                if path:
                    path.write_text(json.dumps(out), encoding="utf-8")
                return out
            except Exception as e:  # rate limits, timeouts, bad JSON
                last = e
                with self._lock:
                    self.usage["retries"] += 1
                time.sleep(min(2 ** attempt, 20))
        raise LLMError(f"LLM call failed after retries: {last}")
