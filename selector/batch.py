"""OpenAI Batch API path for big uploads: asynchronous (results within 24 h), billed at the Batch rate
(typically about half the live price - check the current pricing page), no per-minute rate-limit babysitting.

  python run.py big.json --batch-submit              -> uploads requests, prints a batch id, exits
  python run.py big.json --batch-collect <batch_id>  -> when the batch is done: same outputs + metrics as a live run

Same prompt, schema, validation and decision rule as the live path (selector.pipeline.prepare / finalize).
One sample per product (--repeat is ignored here).
"""
import json, os
from . import pipeline as P
from .llm import build_request
from .prompts import SYSTEM_PROMPT, PROMPT_VERSION, response_schema


# Purpose: resolve which model name to use for a batch job - same precedence as
#   OpenAIJSONClient (explicit cfg.model, then VERTEX_MODEL env var, then the hardcoded
#   default) so the batch path and the live path always pick the same model.
# Input: cfg - the run Config (only cfg.model is read).
# Output: str - the model name to submit requests with.
def _model(cfg):
    return cfg.model or os.environ.get("VERTEX_MODEL", "gpt-4.1-mini")


# Purpose: turn every product needing an LLM call into one Batch API request line -
#   mirrors the live path's selector.pipeline.prepare() to build the same prompt, and
#   skips any product prepare() marks "early" (empty/no results - decided locally, no
#   API call needed either live or batched).
# Input: data - the input JSON (product_key -> product dict); cfg - the run Config.
# Output: tuple(reqs, cid_to_key) - reqs is the list of Batch API request dicts (each with
#   a custom_id, method, url, and body from build_request()); cid_to_key maps each
#   custom_id back to its original product key, needed later to line up results in collect().
def build_requests(data: dict, cfg) -> tuple[list[dict], dict]:
    reqs, cid_to_key = [], {}
    for n, (k, v) in enumerate(data.items()):
        ctx = P.prepare(v)
        if ctx["early"]:
            continue                                   # empty / no results: decided locally, no API call
        cid = f"p{n}"
        cid_to_key[cid] = k
        reqs.append({"custom_id": cid, "method": "POST", "url": "/v1/chat/completions",
                     "body": build_request(_model(cfg), 0.0, SYSTEM_PROMPT, ctx["user"], response_schema(ctx["idx"]))})
    return reqs, cid_to_key


# Purpose: kick off an OpenAI Batch API job for a whole input file in one call - builds
#   every request, writes them as a JSONL file, uploads that file, creates the batch (24h
#   completion window), and saves a small sidecar JSON (batch id + cid_to_key + model +
#   prompt version) on disk so a later `--batch-collect` call can find and decode the
#   result without needing to recompute anything.
# Input: data - the input JSON; cfg - the run Config; client - an OpenAI client instance;
#   workdir - directory to write batch_input.jsonl and batch_<id>.json into (default ".").
# Output: str - the new batch's id (also printed by run.py for the user to save).
def submit(data: dict, cfg, client, workdir: str = ".") -> str:
    reqs, c2k = build_requests(data, cfg)
    path = os.path.join(workdir, "batch_input.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in reqs)
    with open(path, "rb") as fh:
        up = client.files.create(file=fh, purpose="batch")
    b = client.batches.create(input_file_id=up.id, endpoint="/v1/chat/completions", completion_window="24h")
    with open(os.path.join(workdir, f"batch_{b.id}.json"), "w", encoding="utf-8") as f:
        json.dump({"batch_id": b.id, "cid_to_key": c2k, "model": _model(cfg), "prompt_version": PROMPT_VERSION}, f)
    return b.id


# Purpose: fetch a previously submitted batch job and, if OpenAI has finished it, turn
#   the raw batch output rows back into the same `judged` shape the live pipeline produces
#   - so run.py can hand it to assemble() and get identical *.json/*.audit.json/metrics
#   output regardless of which path (live or batch) produced it. Re-derives early-abstain
#   products locally (they were never sent to the batch) and fails a product closed
#   (empty trust list, "llm_error" status) if its batch row errored or is missing, rather
#   than crashing the whole collect.
# Input: batch_id - the id returned by submit(); data - the original input JSON (must
#   match what was submitted); cfg - the run Config; client - an OpenAI client instance;
#   workdir - directory holding batch_<id>.json from submit() (default ".").
# Output: tuple(judged, usage) when the batch has completed - judged is {product_key:
#   (trusted_indices, audit_dict)} matching the live pipeline's shape; usage is the
#   aggregate token/call counters. Returns (None, status_text) if the batch is not
#   finished yet (status_text describes progress, e.g. completed/total/failed counts).
def collect(batch_id: str, data: dict, cfg, client, workdir: str = "."):
    """-> (judged, usage) when finished, else (None, status_text)."""
    with open(os.path.join(workdir, f"batch_{batch_id}.json"), encoding="utf-8") as f:
        c2k = json.load(f)["cid_to_key"]
    b = client.batches.retrieve(batch_id)
    if b.status != "completed":
        rc = getattr(b, "request_counts", None)
        return None, f"status={b.status}" + (f" completed={rc.completed}/{rc.total} failed={rc.failed}" if rc else "")
    rows = {}
    if getattr(b, "output_file_id", None):
        for line in client.files.content(b.output_file_id).text.splitlines():
            if line.strip():
                r = json.loads(line); rows[r["custom_id"]] = r
    usage = {"prompt": 0, "completion": 0, "calls": 0, "cache_hits": 0, "retries": 0}
    judged = {}
    for k, v in data.items():
        judged[k] = P._judge_once(v, cfg, None)          # early-abstain products (no API call needed)
    for cid, k in c2k.items():
        ctx = P.prepare(data[k])
        r = rows.get(cid)
        try:
            body = r["response"]["body"]
            if r.get("error") or r["response"].get("status_code") != 200:
                raise RuntimeError("batch request failed")
            out = json.loads(body["choices"][0]["message"]["content"])
            usage["calls"] += 1
            usage["prompt"] += body.get("usage", {}).get("prompt_tokens", 0)
            usage["completion"] += body.get("usage", {}).get("completion_tokens", 0)
            judged[k] = P.finalize(out, ctx, cfg)
        except Exception as e:                            # fail closed, same as the live path
            judged[k] = ([], {"status": f"llm_error: {type(e).__name__}", "product_is_specific": None, "results": {}})
    return judged, usage
