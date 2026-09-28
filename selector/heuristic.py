"""Cheap lexical baseline (no API needed).

Serves three purposes: (1) a baseline the LLM has to beat, (2) an offline backend so the
pipeline/tests/metrics run without a key, (3) a sanity cross-check signal in production
(large LLM-vs-lexical disagreement gets sampled for human review).
"""
import re
from urllib.parse import urlparse

STOP = set("""a an and the of for with in on to by ea ct pk pack oz fl lb lbs g kg ml l zbtl btl cn can bottle
bottles case count each size new original classic""".split())
UNIT = re.compile(r"^\d+(\.\d+)?(oz|z|ml|l|lb|lbs|g|kg|ct|pk|ea|fl|x)?$")
ABBR = {"unswet": "unsweetened", "blk": "black", "sel": "select", "prtn": "protein", "pwdr": "powder",
        "brk": "", "org": "organic", "frsh": "fresh", "bkd": "baked", "btr": "butter", "rst": "roasted"}


# Purpose: tokenize a piece of text for lexical matching - lowercases, splits into
#   alphanumeric words, expands known abbreviations (ABBR), and drops stopwords/units
#   (STOP, UNIT) so pack-size and filler words never count as a signal.
# Input: text - any raw string (product title/description, or a search-result blob); may
#   be None or empty.
# Output: list[str] - the cleaned, ABBR-expanded, stopword/unit-filtered tokens, in order
#   (duplicates kept; caller dedupes if needed).
def toks(text: str) -> list[str]:
    out = []
    for t in re.findall(r"[a-z0-9]+", (text or "").lower()):
        t = ABBR.get(t, t)
        if not t or t in STOP or UNIT.match(t):
            continue
        out.append(t)
    return out


# Purpose: fuzzy membership test for one product token against a result's token set -
#   exact match first, then a tolerant prefix/plural/abbreviation check (first 4 chars
#   match and one token is a prefix of the other) so e.g. "bottle"/"bottles" or
#   "chocolate"/"choc" still count as a hit.
# Input: tok - one token from the product title/description; haystack - the set of tokens
#   from one search result; text - unused positionally-named parameter kept for call-site
#   symmetry (not read by this function's logic).
# Output: bool - True if tok is considered present in haystack (exact or fuzzy match).
def _has(tok: str, haystack: set[str], text: str) -> bool:
    if tok in haystack:
        return True
    # prefix / plural / abbreviation tolerance
    if len(tok) >= 4 and any(h.startswith(tok[:4]) and (h.startswith(tok) or tok.startswith(h)) for h in haystack if len(h) >= 4):
        return True
    return False


# Purpose: cheap lexical baseline score for every search result against one product -
#   no API call. Tokenizes the product (title+description) and each result, scores by
#   overlap of product tokens found anywhere in the result (half weight) plus overlap
#   found in just the result's first line/title (half weight, since snippets often
#   mention sibling flavors that would falsely inflate a whole-blob-only score).
# Input: title/description - the product's text fields; results - {index_str: blob} where
#   blob is the "Title: ...\nLink: ...\nSnippet: ..." formatted search result text.
# Output: dict[int, float] - result index (as int) -> score in [0, 1], one entry per input
#   result. All-zero scores if the product itself has no usable tokens.
def score_results(title: str, description: str, results: dict[str, str]) -> dict[int, float]:
    ptoks = list(dict.fromkeys(toks(title) + toks(description)))
    scores: dict[int, float] = {}
    for k, blob in results.items():
        slug = urlparse(next((l[6:].strip() for l in blob.split("\n") if l.startswith("Link:")), "")).path
        hay_text = blob + " " + slug
        hay = set(toks(hay_text))
        if not ptoks:
            scores[int(k)] = 0.0
            continue
        hit = sum(1 for t in ptoks if _has(t, hay, hay_text))
        # title-line match counts double: snippets often mention sibling flavors
        first = set(toks(blob.split("\n")[0]))
        hit_title = sum(1 for t in ptoks if _has(t, first, ""))
        scores[int(k)] = 0.5 * hit / len(ptoks) + 0.5 * hit_title / len(ptoks)
    return scores
