"""Metrics. Business framing: a wrongly trusted link poisons enrichment (bad tax category) and
destroys user trust; a missed link only reduces enrichment coverage. So we lead with precision-type
metrics and report recall as the cost we pay.

Products without a ground-truth key are skipped (not an error). A fully unlabeled
file — the live meeting test — still produces pipeline output; scoring reports
`scored: false` instead of failing.
"""
from .pipeline import GT_KEYS


# Purpose: extract a product's human ground-truth set, if it has one - checks both
#   accepted field names (GT_KEYS = "trusted_search_results", "ground_truth") and treats
#   a malformed product value or a non-list GT field as "no ground truth" rather than an
#   error, so a bad row never crashes scoring.
# Input: v - one product's value from the input JSON (expected to be a dict; may not be).
# Output: a set[int] of trusted indices (possibly empty), or None if this product has no
#   usable ground truth at all (treated as unlabeled, not scored).
def _gt(v):
    if not isinstance(v, dict):          # malformed product value (string/list/null/number) -> no GT, not fatal
        return None
    for k in GT_KEYS:
        if k in v and isinstance(v[k], (list, tuple, set)):
            try:
                return set(int(x) for x in v[k])
            except (TypeError, ValueError):
                return None               # ground-truth field present but not a list of numbers -> treat as unlabeled
    return None


# Purpose: score a set of predictions against ground truth across a labeled subset -
#   computes TP/FP/FN and derives precision/recall/F0.5/F1/clean_product_rate/
#   correct_abstain_rate/coverage_when_gt_exists/exact_match_rate. Products without
#   ground truth are silently skipped (not scored, not an error); if NO product in the
#   input has ground truth, returns scored:false instead of dividing by zero.
# Input: data - the input JSON (product_key -> product dict, some/all carrying ground
#   truth); preds - either the full output JSON (dicts with "llm_trusted_search_results")
#   or a plain {product_key: [int, ...]} mapping.
# Output: dict - either {"scored": False, "reason": ..., "products", "products_unlabeled"}
#   when no product has ground truth, or the full scored dict (scored, products,
#   products_unlabeled, result_precision, result_recall, result_F0.5, result_F1,
#   clean_product_rate, coverage_when_gt_exists, correct_abstain_rate, exact_match_rate,
#   counts, products_with_false_links).
def evaluate(data: dict, preds: dict) -> dict:
    """Score any labeled subset. `data` is the input json; `preds` is output json or {key: [ints]}."""
    tp = fp = fn = 0
    n = n_gt_pos = n_pred_pos = clean = exact = any_hit = abst_ok = n_gt_neg = unlabeled = 0
    dirty_products = []
    for k, v in data.items():
        g = _gt(v)
        if g is None:
            unlabeled += 1
            continue
        p = preds[k]
        p = set(p.get("llm_trusted_search_results", [])) if isinstance(p, dict) else set(p)
        n += 1
        tp += len(p & g); fp += len(p - g); fn += len(g - p)
        if p - g:
            dirty_products.append(k)
        exact += p == g
        if g:
            n_gt_pos += 1; any_hit += bool(p & g)
        else:
            n_gt_neg += 1; abst_ok += not p
        n_pred_pos += bool(p)
        clean += not (p - g)
    if n == 0:
        return {
            "scored": False,
            "reason": "no ground truth in input (trusted_search_results / ground_truth); pipeline output was still written",
            "products": unlabeled,
            "products_unlabeled": unlabeled,
        }
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    f = lambda b: (1 + b * b) * prec * rec / (b * b * prec + rec) if prec + rec else 0.0
    return {
        "scored": True,
        "products": n,
        "products_unlabeled": unlabeled,
        "result_precision": round(prec, 3),                 # of links we show, share the humans also trusted
        "result_recall": round(rec, 3),                     # of links humans trusted, share we found
        "result_F0.5": round(f(0.5), 3),                    # headline: precision-weighted (beta=0.5)
        "result_F1": round(f(1), 3),
        "clean_product_rate": round(clean / n, 3),          # products with ZERO wrongly trusted links (user-visible trust)
        "coverage_when_gt_exists": round(any_hit / n_gt_pos, 3) if n_gt_pos else None,  # >=1 correct link found
        "correct_abstain_rate": round(abst_ok / n_gt_neg, 3) if n_gt_neg else None,     # empty when humans trusted none
        "exact_match_rate": round(exact / n, 3),
        "counts": {"TP": tp, "FP": fp, "FN": fn, "products_with_gt": n_gt_pos, "products_no_gt": n_gt_neg,
                   "products_with_any_pred": n_pred_pos},
        "products_with_false_links": dirty_products,
    }
