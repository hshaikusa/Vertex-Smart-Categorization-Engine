#!/usr/bin/env python
"""Run the selector on any json with the case's structure.

  OPENAI_API_KEY=... python run.py test.json                        # production path (prompt v2)
  python run.py samples/train_50.json --backend heuristic           # offline baseline
  python run.py samples/train_50.json --redecide samples/train_50.output.audit.json --conf 0.9
                                                                    # re-apply decision rule to a saved audit: NO API calls
  python run.py samples/train_50.json --subset holdout              # only the ~30% never-tuned-on products

Writes <out>.json (input + llm_trusted_search_results), <out>.audit.json (verdicts, reasons),
<out>.errors.json (every false/missed link with the model's reason). Prints metrics when the input has
ground truth (`trusted_search_results`), overall and split into dev (70%) / holdout (30%).
Split is a stable hash of the product key: tune prompts/thresholds looking at DEV errors only, then report HOLDOUT.
"""
import argparse, hashlib, json, sys
import env_loader  # noqa: F401  — loads OPENAI_API_KEY (and VERTEX_MODEL / VERTEX_CACHE_DIR) from .env
from selector.pipeline import Config, SelectorPipeline, decide, assemble, GT_KEYS
from selector.metrics import evaluate
from selector.prompts import PROMPT_VERSION


# Purpose: deterministically assign a product to the "dev" (70%) or "holdout" (30%)
#   split, purely from its key - so the split is stable across runs/machines with no
#   stored state (same key always lands in the same split). Used so prompts/thresholds
#   are tuned only against dev metrics while holdout stays a held-out check.
# Input: key - the product's key string (the JSON object key from the input file).
# Output: str - "dev" or "holdout".
def split_of(key: str) -> str:
    return "holdout" if int(hashlib.md5(key.encode()).hexdigest(), 16) % 10 < 3 else "dev"


# Purpose: pull a product's human ground-truth set straight out of the raw input dict
#   (run.py's own lightweight version of selector.metrics._gt, used for the CLI's error
#   report and repeat report - does not swallow malformed values the way metrics._gt does).
# Input: v - one product's dict value from the input JSON.
# Output: a set of trusted indices if a GT_KEYS field is present, else None.
def gt_of(v):
    for k in GT_KEYS:
        if k in v: return set(v[k])
    return None


# Purpose: build the human-readable *.errors.json report - for every product that HAS
#   ground truth and where the model's prediction differs from it, lists each mismatched
#   index as FALSE_LINK (model trusted it, human didn't) or MISSED_LINK (human trusted it,
#   model didn't), with the model's own verdict/reason pulled back out of the audit trail.
# Input: out - the final output dict (product -> {..., "llm_trusted_search_results": [...]});
#   audit - the parallel audit dict with per-result verdicts/reasons/confidence.
# Output: dict - {product_key.strip(): {split, human, model, product_is_specific, errors:
#   [...]}}, containing only products that both have ground truth and have at least one
#   mismatch. Products with no ground truth, or a perfect match, are omitted.
def error_report(out, audit):
    rep = {}
    for k, v in out.items():
        g = gt_of(v)
        if g is None: continue
        p = set(v["llm_trusted_search_results"])
        if p == g: continue
        items = []
        for i in sorted(p ^ g):
            res = audit.get(k, {}).get("results", {})
            a = res.get(str(i)) or res.get(i) or {}
            items.append({"index": i, "error": "FALSE_LINK" if i in p else "MISSED_LINK",
                          "title": v["search_results"][str(i)].split("\n")[0][7:90],
                          "verdict": a.get("verdict"), "variant_evidence": a.get("variant_evidence"),
                          "confidence": a.get("confidence"), "reason": a.get("reason")})
        rep[k.strip()] = {"split": split_of(k), "human": sorted(g), "model": sorted(p),
                          "product_is_specific": audit.get(k, {}).get("product_is_specific"), "errors": items}
    return rep


# Purpose: print the overall metrics line plus a per-split (dev/holdout) breakdown to
#   stdout after a run - the main way a human sanity-checks a run's quality. Silently does
#   nothing (just a note) when the input has no ground truth to score against.
# Input: data - the input JSON (may or may not carry ground truth); out - the pipeline's
#   final output dict.
# Output: None (prints to stdout only).
def print_metrics(data, out):
    m = evaluate(data, out)
    if not m.get("scored", True):
        print("(no ground truth in input - skipping metrics)"); return
    print("metrics (all):", json.dumps({k: v for k, v in m.items() if k != "products_with_false_links"}, indent=1))
    for s in ("dev", "holdout"):
        sub = {k: v for k, v in data.items() if split_of(k) == s}
        if sub:
            r = evaluate(sub, {k: out[k] for k in sub})
            print(f"  {s:8s} n={r['products']:3d} P={r['result_precision']} R={r['result_recall']} F0.5={r['result_F0.5']} "
                  f"clean={r['clean_product_rate']} abstain_ok={r['correct_abstain_rate']}")


# Purpose: print the multi-sample voting diagnostics used only when --repeat > 1 - shows
#   metrics for each individual sample and for the final majority-voted decision (so you
#   can see whether voting actually helps), then breaks down, for every result that had a
#   split vote, whether unanimous-trust / unanimous-reject / split-vote results tend to be
#   correct - i.e. whether sample disagreement is a useful signal for flagging likely errors.
# Input: data - the input JSON; audit - the audit dict (must contain "samples"/"votes"/
#   "unstable" keys per product, only present when repeat > 1); cfg - the run Config (for
#   vote_min).
# Output: None (prints to stdout only). No-ops immediately if no product in audit has
#   multiple samples, and skips the per-bucket accuracy breakdown if the input is unlabeled.
def print_repeat_report(data, audit, cfg):
    """Run-to-run noise + does disagreement between samples flag the errors? (only for --repeat > 1)"""
    multi = {k for k, a in audit.items() if "samples" in a}
    if not multi:
        return
    n = len(audit[next(iter(multi))]["samples"])
    need = cfg.vote_min or n // 2 + 1
    dis = sum(bool(audit[k]["unstable"]) for k in multi)
    print(f"\nrepeat report: {n} samples/product, link trusted when >= {need} agree; "
          f"{dis}/{len(multi)} products had at least one split result")
    labelled = any(gt_of(v) is not None for v in data.values())
    if not labelled:
        return
    def show(name, pred):
        m = evaluate(data, pred)
        print(f"  {name:12s} P={m['result_precision']} R={m['result_recall']} F0.5={m['result_F0.5']} "
              f"clean={m['clean_product_rate']} abstain_ok={m['correct_abstain_rate']} FP={m['counts']['FP']} FN={m['counts']['FN']}")
    for s in range(n):
        show(f"sample {s+1}", {k: decide(audit[k]["samples"][s], cfg) if k in multi else decide(audit[k], cfg) for k in data})
    show(f"VOTED {need}/{n}", {k: decide(audit[k], cfg) for k in data})
    b = {"unanimous trust": [0, 0], "split vote": [0, 0], "unanimous reject": [0, 0]}
    for k in multi:
        g = gt_of(data[k])
        if g is None: continue
        votes = {int(i): v for i, v in audit[k]["votes"].items()}
        for i in (int(x) for x in audit[k]["results"]):
            v = votes.get(i, 0)
            name = "unanimous trust" if v == n else ("unanimous reject" if v == 0 else "split vote")
            b[name][int(i in g)] += 1
    for name, (neg, pos) in b.items():
        tot = neg + pos
        if tot:
            what = "human-trusted (= missed links)" if name == "unanimous reject" else "human-trusted (= correct links)"
            print(f"  {name:16s} {tot:4d} results, {pos} {what} = {pos/tot:.2f}")


# Purpose: CLI entry point - parses arguments, loads and validates the input JSON, routes
#   to one of three execution modes (normal live/heuristic run via SelectorPipeline.run,
#   OpenAI Batch API submit/collect via selector.batch, or a free "redecide" pass that
#   re-applies decide() to a previously saved audit.json with zero new API calls), then
#   writes the three output files (*.json/*.audit.json/*.errors.json), prints ops stats
#   and metrics, and prints a summary line of what was written.
# Input: none directly - reads sys.argv via argparse (see the module docstring at the top
#   of this file for the full flag list: input path, --out, --backend, --model, --conf,
#   --heur, --require-confirmed, --redecide, --repeat, --vote-min, --batch-submit,
#   --batch-collect, --subset, --workers, --no-cache).
# Output: None. Exits early (via sys.exit or a bare return) for a bad/missing input file,
#   a --batch-submit request (prints the batch id and returns), or an unfinished
#   --batch-collect (prints status and returns). Otherwise writes <out>.json,
#   <out>.audit.json, and (if there were any errors) <out>.errors.json to disk.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input"); ap.add_argument("--out", default=None)
    ap.add_argument("--backend", default="openai", choices=["openai", "heuristic"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--conf", type=float, default=None, help="LLM confidence threshold (default 0.70)")
    ap.add_argument("--heur", type=float, default=None)
    ap.add_argument("--require-confirmed", action="store_true", help="only trust matches whose variant_evidence=confirmed")
    ap.add_argument("--redecide", default=None, help="saved audit.json: re-apply decision rule offline, no API calls")
    ap.add_argument("--repeat", type=int, default=1, help="independent LLM samples per product; >1 = majority vote (costs N x)")
    ap.add_argument("--vote-min", type=int, default=None, help="votes needed to trust a link (default: strict majority)")
    ap.add_argument("--batch-submit", action="store_true", help="OpenAI Batch API: upload requests, print batch id, exit")
    ap.add_argument("--batch-collect", default=None, metavar="BATCH_ID", help="fetch a finished batch and produce the normal outputs + metrics")
    ap.add_argument("--subset", default="all", choices=["all", "dev", "holdout"])
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()

    try:
        with open(a.input, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        sys.exit(f"error: input file not found: {a.input}")
    except json.JSONDecodeError as e:
        sys.exit(f"error: {a.input} is not valid JSON ({e.msg} at line {e.lineno}, col {e.colno}). "
                  "Nothing was run - check the file for a trailing comma, unescaped quote, or truncated copy/paste.")
    if not isinstance(data, dict):
        sys.exit(f"error: {a.input} must be a JSON OBJECT of {{product_key: product}} (got a top-level "
                  f"{type(data).__name__}). Nothing was run.")
    bad_products = [k for k, v in data.items() if not isinstance(v, dict)]
    if bad_products:
        print(f"warning: {len(bad_products)} product(s) have a non-object value and will be recorded as "
              f"'invalid_product_format' with no trusted results (first few: {bad_products[:5]})")
    if a.subset != "all":
        data = {k: v for k, v in data.items() if split_of(k) == a.subset}
    cfg = Config(backend=a.backend, model=a.model, workers=a.workers, use_cache=not a.no_cache,
                 require_variant_confirmed=a.require_confirmed, repeat=a.repeat, vote_min=a.vote_min)
    if a.conf is not None: cfg.conf_threshold = a.conf
    if a.heur is not None: cfg.heuristic_threshold = a.heur
    base = a.out or a.input.rsplit(".", 1)[0] + (".redecided" if a.redecide else ".output")

    if a.batch_submit:
        from openai import OpenAI
        from selector import batch
        print("batch id:", batch.submit(data, cfg, OpenAI()), "-> later: python run.py", a.input, "--batch-collect <id>")
        return
    if a.batch_collect:
        from openai import OpenAI
        from selector import batch
        judged, usage = batch.collect(a.batch_collect, data, cfg, OpenAI())
        if judged is None:
            print("batch not finished yet:", usage); return
        out, audit, stats = assemble(data, judged, cfg, 0.0, extra_usage=usage)
        stats["prompt_version"] = PROMPT_VERSION; stats["mode"] = "batch"
    elif a.redecide:
        saved = json.load(open(a.redecide, encoding="utf-8"))
        audit = {k: saved[k] for k in data}
        out = json.loads(json.dumps(data))
        for k in out: out[k]["llm_trusted_search_results"] = decide(audit[k], cfg)
        stats = {"mode": "redecide (no API calls)", "conf": cfg.conf_threshold, "require_confirmed": cfg.require_variant_confirmed}
    else:
        out, audit, stats = SelectorPipeline(cfg).run(data)
        stats["prompt_version"] = PROMPT_VERSION
    json.dump(out, open(base + ".json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump(audit, open(base + ".audit.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=str)
    rep = error_report(out, audit)
    if rep: json.dump(rep, open(base + ".errors.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("ops:", json.dumps(stats))
    print_metrics(data, out)
    print_repeat_report(data, audit, cfg)
    print(f"wrote {base}.json, {base}.audit.json" + (f", {base}.errors.json" if rep else ""))


if __name__ == "__main__":
    main()
