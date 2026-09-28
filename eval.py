#!/usr/bin/env python
"""Eval harness for the search-result selector.

Scores any labeled subset of a case-format JSON. Unlabeled files (the live
meeting test) still run the pipeline and report scored=false instead of failing.

  python eval.py samples/train_50.json
  python eval.py samples/train_50.json --backend heuristic
  python eval.py test.json
  python eval.py samples/train_50.json --predictions samples/train_50.output.json
  python eval.py samples/synthetic_1.json --backend heuristic --out-report eval_report.json
"""
import argparse, json, os, sys
import env_loader  # noqa: F401  — loads OPENAI_API_KEY from .env
from selector.pipeline import Config, SelectorPipeline
from selector.metrics import evaluate
from run import KEY_HELP, resolve_input


# Purpose: format one evaluate()-style metrics dict into a human-readable multi-line
#   report for the terminal - a short "not scored" message when the input carried no
#   ground truth, otherwise headline/coverage/abstain/counts lines plus a list of any
#   products with a false (wrongly-trusted) link.
# Input: m - the dict returned by selector.metrics.evaluate().
# Output: str - the formatted report, ready to print.
def _report(m: dict) -> str:
    if not m.get("scored"):
        return (
            "EVAL: not scored (no trusted_search_results / ground_truth on any product).\n"
            f"  products={m.get('products')}  {m.get('reason')}"
        )
    c = m["counts"]
    lines = [
        f"EVAL  n={m['products']} unlabeled={m.get('products_unlabeled', 0)}",
        f"  headline   F0.5={m['result_F0.5']}  precision={m['result_precision']}  clean_product={m['clean_product_rate']}",
        f"  coverage   recall={m['result_recall']}  F1={m['result_F1']}  exact={m['exact_match_rate']}",
        f"  abstain    correct_abstain={m['correct_abstain_rate']}  coverage_when_gt={m['coverage_when_gt_exists']}",
        f"  counts     TP={c['TP']} FP={c['FP']} FN={c['FN']}  "
        f"gt+={c['products_with_gt']} gt0={c['products_no_gt']} pred+={c['products_with_any_pred']}",
    ]
    dirty = m.get("products_with_false_links") or []
    if dirty:
        lines.append(f"  false-link products ({len(dirty)}):")
        lines.extend(f"    - {k}" for k in dirty)
    return "\n".join(lines)


# Purpose: CLI entry point for the standalone eval harness - either scores an existing
#   predictions file against an input's ground truth with no model call
#   (--predictions), or runs the full SelectorPipeline first and then scores its own
#   output; either way prints the ops stats and the _report() summary, and optionally
#   writes a combined {ops, metrics} JSON report to disk.
# Input: none directly - reads sys.argv via argparse (input path, --predictions,
#   --backend, --model, --conf, --heur, --workers, --no-cache, --out, --out-report).
# Output: None. When not scoring an existing predictions file, also writes
#   <out>.json/<out>.audit.json (the pipeline's own outputs) to disk, and writes
#   --out-report's path if given.
def main():
    ap = argparse.ArgumentParser(description="Run / score the selector eval harness")
    ap.add_argument("input", nargs="?", default=None)
    ap.add_argument("--train", action="store_true", help="samples/train_50.json (or INPUT)")
    ap.add_argument("--test", action="store_true", help="held-out test JSON (or INPUT); score the whole file")
    ap.add_argument("--predictions", default=None, help="score an existing output JSON; skip the model")
    ap.add_argument("--backend", default="openai", choices=["openai", "heuristic"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--heur", type=float, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--out", default=None, help="pipeline output prefix (ignored with --predictions)")
    ap.add_argument("--out-report", default=None, help="write the metrics JSON here")
    a = ap.parse_args()
    a.input, eval_set, _report_split = resolve_input(a.train, a.test, a.input)
    data = json.load(open(a.input, encoding="utf-8"))

    if a.predictions:
        preds = json.load(open(a.predictions, encoding="utf-8"))
        stats = {"mode": "score_only", "predictions": a.predictions, "eval_set": eval_set}
    else:
        if a.backend == "openai" and not os.environ.get("OPENAI_API_KEY"):
            print(KEY_HELP, file=sys.stderr)
        cfg = Config(backend=a.backend, model=a.model, workers=a.workers, use_cache=not a.no_cache)
        if a.conf is not None: cfg.conf_threshold = a.conf
        if a.heur is not None: cfg.heuristic_threshold = a.heur
        out, audit, stats = SelectorPipeline(cfg).run(data)
        stats = {"mode": "run_and_score", "eval_set": eval_set, **stats}
        base = a.out or a.input.rsplit(".", 1)[0] + ".output"
        json.dump(out, open(base + ".json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        json.dump(audit, open(base + ".audit.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=str)
        preds = out
        print(f"wrote {base}.json and {base}.audit.json")

    m = evaluate(data, preds)
    print("ops:", json.dumps(stats))
    print(_report(m))
    if a.out_report:
        json.dump({"ops": stats, "metrics": m}, open(a.out_report, "w", encoding="utf-8"), indent=2)
        print(f"wrote {a.out_report}")
    if not a.predictions and a.backend == "openai" and not os.environ.get("OPENAI_API_KEY") and stats.get("errors"):
        print("Uncached products failed closed because OPENAI_API_KEY is unset.", file=sys.stderr)


if __name__ == "__main__":
    main()
