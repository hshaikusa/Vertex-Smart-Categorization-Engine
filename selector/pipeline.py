"""Search-result selection pipeline.

input  : {product_key: {product_title, product_description, search_results{"1": "...", ...}}}
output : same dict + `llm_trusted_search_results: [int, ...]` per product (+ separate audit trail)

Failure policy is FAIL CLOSED: on any error / invalid output / unspecific product we trust nothing.
An empty list costs us enrichment coverage; a wrong link poisons the tax classification.
"""
from __future__ import annotations
import concurrent.futures as cf
import collections, copy, re, time
from urllib.parse import urlparse
from dataclasses import dataclass, field

from . import heuristic
from .prompts import SYSTEM_PROMPT, USER_TEMPLATE, response_schema

GT_KEYS = ("trusted_search_results", "ground_truth")
MAX_RESULTS = 15                # results beyond this many are dropped (recorded in audit as "results_dropped")
MAX_SNIPPET_CHARS = 600
MAX_TITLE_CHARS = 300           # caps a single malformed/adversarial title from inflating cost
MAX_RESULT_TITLE_CHARS = 200    # same, for one search result's own Title field (was uncapped)
INJECTION_PAT = re.compile(r"(ignore (all|previous|the above)|system prompt|you are now|disregard)", re.I)


@dataclass
class Config:
    backend: str = "openai"            # "openai" | "heuristic"
    model: str | None = None
    conf_threshold: float = 0.70       # tuned on train, see run.py --sweep
    heuristic_threshold: float = 0.75
    workers: int = 8
    use_cache: bool = True
    require_variant_confirmed: bool = False   # if True, 'not_stated' matches are not trusted (stricter)
    repeat: int = 1                    # independent LLM samples per product; >1 enables majority voting
    vote_min: int | None = None        # votes needed to trust a link (default: strict majority, e.g. 2 of 3)


# Purpose: normalize a raw text field before it's used anywhere downstream - strips stray
#   non-breaking spaces, neutralises anything that looks like an HTML/XML tag (defends
#   against a prompt-injection attempt using our own <data>/<search_results> tag syntax),
#   and collapses repeated whitespace.
# Input: s - any value from a JSON field (expected to be a string, but may not be).
# Output: a str - the cleaned text, or "" if s wasn't a string to begin with.
def _clean(s) -> str:
    if not isinstance(s, str):        # non-string field (number, list, dict, bool...) -> treated as empty, not fatal
        return ""
    s = s.replace("\xa0", " ")
    s = re.sub(r"<[^>]{0,40}>", " ", s)              # neutralise anything that looks like our prompt tags
    return re.sub(r"\s+", " ", s).strip()


# Purpose: parse a value as an int without ever raising - used to tell a real numeric
#   result key ("3") apart from a garbage/non-numeric one ("abc") while building the
#   search-result index list.
# Input: x - any value (typically a dict key, so usually a str).
# Output: int(x) on success, or None if x can't be parsed as an integer.
def _safe_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


# Purpose: reduce a URL down to "the same page" for duplicate detection - strips the
#   scheme, a leading "www.", the query string (which is where ad-click tracking params
#   like srsltid live) and any trailing slash/fragment, so two byte-different URLs that
#   point at the identical product page collapse to one key.
# Input: link - a raw URL string (or None).
# Output: a str - "netloc/path", lower-cased, with no scheme/www/query/fragment.
def _norm_url(link: str) -> str:
    """domain + path, lower-case, no scheme / www / query (tracking params like srsltid) / fragment."""
    u = urlparse((link or "").strip().lower())
    return (u.netloc.replace("www.", "") + u.path.rstrip("/"))


# Purpose: turn one product's raw search_results dict into the numbered prompt text sent
#   to the model, while deduping same-page results (sent once, verdict copied to the
#   duplicate afterwards) and dropping/reporting anything beyond MAX_RESULTS or with a
#   non-numeric key - all issues are recorded, never silently discarded.
# Input: results - dict of {result_key: raw "Title: ...\nLink: ...\nSnippet: ..." string}.
# Output: tuple of (prompt text block, idx list actually sent to the model,
#   dup_of {duplicate_index: representative_index}, dropped index list, invalid key list).
def _format_results(results: dict) -> tuple[str, list[int], dict[int, int], list[int], list[str]]:
    """-> (prompt text, indices sent to the model, {duplicate_index: representative_index}, dropped indices).
    Results pointing at the same page are sent ONCE; the verdict is copied to the duplicates afterwards
    (saves tokens and makes duplicate handling consistent). Results beyond MAX_RESULTS are dropped and
    reported (not silently discarded) so an unusually long result list is visible in the audit."""
    invalid_keys = sorted(str(k) for k in results if _safe_int(k) is None)   # non-numeric/garbage keys: skipped, reported, not fatal
    all_keys = [k for k in results if _safe_int(k) is not None]
    kept_keys = sorted(all_keys, key=lambda x: int(x))[:MAX_RESULTS]
    dropped = sorted(int(k) for k in set(all_keys) - set(kept_keys))    # numeric but beyond MAX_RESULTS
    lines, idx, dup_of, seen = [], [], {}, {}
    for k in kept_keys:
        raw = results[k] if isinstance(results[k], str) else ""      # non-string result value -> treated as empty, not fatal
        parts = {p.split(":", 1)[0]: p.split(":", 1)[1].strip() for p in raw.split("\n") if ":" in p}
        key = _norm_url(parts.get("Link", ""))
        if key and key in seen:
            dup_of[int(k)] = seen[key]
            continue
        if key:
            seen[key] = int(k)
        snippet = _clean(parts.get("Snippet", ""))[:MAX_SNIPPET_CHARS]
        title = _clean(parts.get("Title", ""))[:MAX_RESULT_TITLE_CHARS]
        lines.append(f"[{int(k)}] Title: {title}\n    URL: {key[:160]}\n    Snippet: {snippet}")
        idx.append(int(k))
    return "\n".join(lines), idx, dup_of, dropped, invalid_keys


# Purpose: build everything one call to the model needs for a single product - decode
#   the title/description, format the search results (see _format_results), and detect
#   the early-exit cases (malformed product, no search results, or too little text to
#   judge) BEFORE any model call or exception can happen.
# Input: p - one product's value from the input JSON (expected to be a dict; may not be).
# Output: dict ctx with at least "early" (None or a reason string); when "early" is None
#   it also carries "body"/"idx"/"dup_of"/"dropped"/"invalid_keys"/"user" (the literal
#   prompt text to send to the model).
def prepare(p) -> dict:
    """Everything needed to call the model for one product (also used by the Batch API path).
    p is expected to be a dict (one JSON object per product); a malformed input (string, list, null,
    number in that slot) is caught HERE with a clear status rather than raising deep inside the pipeline."""
    if not isinstance(p, dict):
        return {"early": "invalid_product_format", "title": "", "desc": "", "results": {}}
    title = _clean(p.get("product_title"))[:MAX_TITLE_CHARS]
    desc = _clean(p.get("product_description"))[:MAX_TITLE_CHARS]
    results = p.get("search_results")
    if not isinstance(results, dict):
        results = {}
    ctx = {"title": title, "desc": desc, "results": results, "early": None}
    ctx["upload_flag"] = "possible_prompt_injection_in_upload_fields" if INJECTION_PAT.search(f"{title} {desc}") else None
    if not results:
        ctx["early"] = "no_results"
    elif len(title) < 3 and len(desc) < 3:
        ctx["early"] = "abstain_empty_input"
    else:
        ctx["body"], ctx["idx"], ctx["dup_of"], ctx["dropped"], ctx["invalid_keys"] = _format_results(results)
        ctx["user"] = USER_TEMPLATE.format(title=title, description=desc, results=ctx["body"])
    return ctx


# Purpose: validate one model response against the product's own context and apply the
#   trust decision - builds the per-index audit (clamping confidence to [0,1], truncating
#   the reason, flagging a result whose raw snippet matches the injection pattern),
#   copies the verdict onto any duplicate index, then calls decide().
# Input: out - the raw JSON dict returned by the model; ctx - this product's prepare()
#   output; cfg - the run's Config (thresholds, etc).
# Output: tuple of (trusted index list from decide(), the built audit dict).
def finalize(out: dict, ctx: dict, cfg: Config) -> tuple[list[int], dict]:
    """Validate the model output for one product and apply the decision rule."""
    idx, results = ctx["idx"], ctx["results"]
    audit = {"status": "ok", "product_is_specific": bool(out.get("product_is_specific")),
             "normalized_product": out.get("normalized_product"), "results": {}}
    seen = set()
    for r in out.get("results", []):
        i = r.get("index")
        if i not in idx or i in seen:                 # guardrail: hallucinated / duplicate indices ignored
            continue
        seen.add(i)
        conf = min(max(float(r.get("confidence", 0)), 0.0), 1.0)
        audit["results"][i] = {"verdict": r["verdict"], "variant_evidence": r.get("variant_evidence"),
                               "confidence": conf, "reason": r["reason"][:200]}
        raw = results.get(str(i)) if isinstance(results, dict) else None
        if INJECTION_PAT.search(raw if isinstance(raw, str) else ""):
            audit["results"][i]["flag"] = "possible_prompt_injection_in_snippet"
    if len(seen) < len(idx):
        audit["status"] = "incomplete_model_output"   # missing indices are simply not trusted
    for d, rep in ctx["dup_of"].items():              # same page -> same verdict
        if rep in audit["results"]:
            audit["results"][d] = {**audit["results"][rep], "dup_of": rep}
    _add_data_quality_flags(audit, ctx)
    return decide(audit, cfg), audit


# Purpose: attach non-fatal, informational notes about the INPUT data's quality to the
#   audit (results dropped for exceeding MAX_RESULTS, non-numeric result keys, a possible
#   injection attempt in the product's own title/description) - purely for visibility;
#   none of these ever change the trust decision itself.
# Input: audit - the audit dict being built for this product (mutated in place);
#   ctx - this product's prepare() output.
# Output: None (mutates audit in place).
def _add_data_quality_flags(audit: dict, ctx: dict) -> None:
    """Attach non-fatal input-quality notes to the audit (never changes the trust decision itself)."""
    if ctx.get("dropped"):
        audit["results_dropped"] = ctx["dropped"]              # numeric indices beyond MAX_RESULTS, never sent to the model
    if ctx.get("invalid_keys"):
        audit["invalid_result_keys"] = ctx["invalid_keys"]     # non-numeric/garbage result keys, skipped
    if ctx.get("upload_flag"):
        audit["upload_flag"] = ctx["upload_flag"]               # possible injection in product_title/description itself


# Purpose: run exactly ONE judging pass for one product - call prepare(), handle the
#   early-exit statuses, dispatch to either the offline heuristic scorer or a real model
#   call (catching any API exception so it fails closed rather than crashing), then hand
#   a real model response to finalize().
# Input: p - one product's raw value; cfg - the run's Config; llm - the OpenAIJSONClient
#   (unused for the heuristic backend); sample - which independent sample this is (0 for
#   a normal single call, 1+ when cfg.repeat > 1).
# Output: tuple of (trusted index list, the audit dict for this one pass).
def _judge_once(p, cfg: Config, llm=None, sample: int = 0) -> tuple[list[int], dict]:
    ctx = prepare(p)
    audit = {"status": "ok", "product_is_specific": None, "results": {}}
    if ctx["early"]:
        audit["status"] = ctx["early"]; return [], audit
    _add_data_quality_flags(audit, ctx)
    if cfg.backend == "heuristic":
        allidx = sorted(ctx["idx"] + list(ctx["dup_of"]))
        blobs = {str(k): (ctx["results"].get(str(k)) if isinstance(ctx["results"].get(str(k)), str) else "") for k in allidx}
        sc = heuristic.score_results(ctx["title"], ctx["desc"], blobs)
        audit["results"] = {i: {"score": round(sc[i], 3)} for i in allidx}
        return [i for i in allidx if sc[i] >= cfg.heuristic_threshold], audit
    try:
        out = llm.complete_json(SYSTEM_PROMPT, ctx["user"], response_schema(ctx["idx"]), sample=sample)
    except Exception as e:
        audit["status"] = f"llm_error: {type(e).__name__}"; return [], audit
    return finalize(out, ctx, cfg)


# Purpose: roll up N independent per-sample audits (from repeated calls with cfg.repeat
#   > 1) into one combined audit - per result index, takes the modal verdict, the mean
#   confidence, and a vote count, so decide() can apply the majority-vote threshold to it
#   exactly like a single-sample audit.
# Input: samples - list of per-sample audit dicts (each from one _judge_once() call);
#   cfg - the run's Config (used only to recompute per-sample decide() for vote counting).
# Output: dict - a single combined audit with "samples", "votes", "unstable", and a
#   "results" map shaped like a normal single-call audit.
def _summarise_samples(samples: list[dict], cfg: Config) -> dict:
    """Readable roll-up of N independent samples (votes, modal verdict, mean confidence)."""
    n = len(samples)
    votes = collections.Counter(i for s in samples for i in decide(s, cfg))
    spec = sum(bool(s.get("product_is_specific")) for s in samples)
    results = {}
    for i in sorted({i for s in samples for i in s["results"]}):
        rs = [s["results"][i] for s in samples if i in s["results"]]
        verdict = collections.Counter(r["verdict"] for r in rs).most_common(1)[0][0]
        first = next(r for r in rs if r["verdict"] == verdict)
        results[i] = {"verdict": verdict, "variant_evidence": first.get("variant_evidence"),
                      "confidence": round(sum(r["confidence"] for r in rs) / len(rs), 3),
                      "reason": first["reason"], "votes": f"{votes.get(i, 0)}/{n}"}
    bad = [s["status"] for s in samples if s["status"] != "ok"]
    return {"status": "ok" if not bad else bad[0], "n_samples": n, "samples": samples,
            "product_is_specific": spec * 2 > n, "votes": dict(votes), "results": results,
            "unstable": sorted(i for i, v in votes.items() if v < n) }


# Purpose: the per-product entry point submitted to the thread pool - decides how many
#   independent samples to take (1 normally, cfg.repeat when the backend is "openai" and
#   repeat > 1), skips the repeat/majority-vote path entirely for an early-exit product
#   (nothing to gain by repeating a no_results/abstain_empty_input result), and otherwise
#   gathers all samples and summarises + re-decides on the combined audit.
# Input: p - one product's raw value from the input JSON; cfg - the run's Config;
#   llm - the OpenAIJSONClient to call (the heuristic backend uses none).
# Output: tuple of (trusted index list, the audit dict for this product).
def select_for_product(p: dict, cfg: Config, llm=None) -> tuple[list[int], dict]:
    n = max(1, cfg.repeat) if cfg.backend == "openai" else 1
    trusted, first = _judge_once(p, cfg, llm, 0)
    if n == 1 or first["status"] in ("no_results", "abstain_empty_input"):
        return trusted, first
    samples = [first] + [_judge_once(p, cfg, llm, s)[1] for s in range(1, n)]
    audit = _summarise_samples(samples, cfg)
    return decide(audit, cfg), audit


# Purpose: the pure decision rule - given an already-saved audit (real or replayed from
#   disk) and a Config, decide which indices are trusted. Runs with zero API calls, which
#   is what lets --redecide re-threshold a saved run offline. Recurses into the
#   majority-vote branch when the audit holds multiple "samples"; otherwise applies the
#   per-result gate (verdict, confidence, injection flag, variant-evidence checks).
# Input: audit - one product's audit dict (single-call shape, or the "samples" roll-up
#   shape from _summarise_samples); cfg - the run's Config (conf_threshold, vote_min,
#   require_variant_confirmed).
# Output: a sorted list[int] of the trusted result indices (possibly empty).
def decide(audit: dict, cfg: Config) -> list[int]:
    """Pure decision rule over a saved audit entry -> lets us re-threshold offline with zero API calls."""
    if "samples" in audit:                               # majority vote over independent samples
        n = len(audit["samples"]); need = cfg.vote_min or (n // 2 + 1)
        votes = collections.Counter(i for s in audit["samples"] for i in decide(s, cfg))
        return sorted(i for i, v in votes.items() if v >= need)
    if not audit.get("product_is_specific"):
        return []
    out = []
    for i, r in audit["results"].items():
        if r.get("verdict") != "match" or r.get("confidence", 0) < cfg.conf_threshold or "flag" in r:
            continue
        if cfg.require_variant_confirmed and r.get("variant_evidence") != "confirmed":
            continue
        if r.get("variant_evidence") == "contradicted":
            continue
        out.append(int(i))
    return sorted(out)


# Purpose: turn the per-product (trusted, audit) results into the three things run.py
#   writes out - the output JSON (original data + llm_trusted_search_results per
#   product, wrapping any malformed original product value instead of crashing), the
#   full audit-by-product dict, and the run's ops stats (product/error/duplicate counts,
#   token usage).
# Input: data - the original input JSON; judged - {product_key: (trusted, audit)} as
#   returned by select_for_product() for every product; cfg - the run's Config;
#   seconds - elapsed wall-clock time for the whole run; llm - the OpenAIJSONClient (for
#   its usage counters, or None for the heuristic backend); extra_usage - a token-usage
#   dict supplied by the Batch API path instead of a live llm object.
# Output: tuple of (output_json dict, audits dict, stats dict).
def assemble(data: dict, judged: dict, cfg: Config, seconds: float, llm=None, extra_usage: dict | None = None):
    """judged: {product_key: (trusted, audit)} -> (output_json, audits, stats)"""
    out, audits = copy.deepcopy(data), {}
    for k, (trusted, audit) in judged.items():
        if isinstance(out.get(k), dict):
            out[k]["llm_trusted_search_results"] = trusted
        else:
            # malformed product value (string/list/null/number instead of an object): wrap it rather than
            # crashing the whole batch with a TypeError on item assignment. The original value is preserved
            # under "original_value" so nothing is silently lost, and every OTHER product still gets written.
            out[k] = {"original_value": out.get(k), "llm_trusted_search_results": trusted}
        audits[k] = audit
    stats = {"products": len(data), "seconds": round(seconds, 2),
             "errors": sum(a["status"] not in ("ok", "no_results", "abstain_empty_input") for a in audits.values())}
    multi = [a for a in audits.values() if "samples" in a]
    if multi:
        stats["repeat"] = cfg.repeat
        stats["products_with_disagreement"] = sum(bool(a["unstable"]) for a in multi)
        stats["unstable_results"] = sum(len(a["unstable"]) for a in multi)
    dups = sum(len(prepare(v).get("dup_of", {})) for v in data.values()) if cfg.backend == "openai" else 0
    stats["duplicate_urls_not_sent"] = dups
    if llm is not None:
        stats["llm_usage"] = dict(llm.usage)
    if extra_usage:
        stats["llm_usage"] = extra_usage
    return out, audits, stats


class SelectorPipeline:
    # Purpose: build the pipeline for one run - keep the given Config (or the default),
    #   and construct a real OpenAIJSONClient unless one was already supplied or the
    #   backend is "heuristic" (which needs no client at all).
    # Input: cfg - a Config, or None to use Config()'s defaults; llm - an existing
    #   OpenAIJSONClient to reuse (e.g. across multiple calls), or None to build one.
    # Output: None (constructs self.cfg and self.llm).
    def __init__(self, cfg: Config | None = None, llm=None):
        self.cfg = cfg or Config()
        if self.cfg.backend == "openai" and llm is None:
            from .llm import OpenAIJSONClient
            llm = OpenAIJSONClient(model=self.cfg.model, use_cache=self.cfg.use_cache)
        self.llm = llm

    # Purpose: run every product in `data` through select_for_product() on a thread
    #   pool, catching any exception per-product so one bad product can never take down
    #   the batch (the "never let one product crash the batch" safety net), then hand the
    #   collected results to assemble() for the final output/audit/stats.
    # Input: data - the full input JSON, {product_key: product_value, ...}.
    # Output: tuple of (output_json dict, audits dict, stats dict) - assemble()'s return.
    def run(self, data: dict) -> tuple[dict, dict, dict]:
        """returns (output_json, audit_by_product, ops_stats)"""
        t0, judged = time.time(), {}
        with cf.ThreadPoolExecutor(self.cfg.workers) as ex:
            futs = {ex.submit(select_for_product, v, self.cfg, self.llm): k for k, v in data.items()}
            for f in cf.as_completed(futs):
                k = futs[f]
                try:
                    judged[k] = f.result()
                except Exception as e:                # never let one product crash the batch
                    judged[k] = ([], {"status": f"pipeline_error: {type(e).__name__}: {e}"})
        return assemble(data, judged, self.cfg, time.time() - t0, self.llm)
