# Vertex Smart Categorization – Search Result Selector

GenAI step that decides which **already-retrieved** search results are about the
uploaded product. It does not run a live web search. Wrong links poison
downstream enrichment (ingredients → tax category) and are shown in the product
UI, so the pipeline **fails closed** and optimizes **precision / F0.5 / clean-product
rate** over raw recall.

## Setup

```
pip install -r requirements.txt
copy .env.example .env          # Windows
# then set OPENAI_API_KEY=... and optionally VERTEX_MODEL=gpt-4.1-mini
```

`env_loader.py` loads `.env` on import (`run.py`, `eval.py`). **`.env` overrides** a
shell `OPENAI_API_KEY` so the file you edit is what the OpenAI client uses.
`.env` is gitignored; never commit it.

Windows:

```
python run.py samples/train_50.json
```

Unix (optional; `.env` is enough):

```
export OPENAI_API_KEY=...
python run.py samples/train_50.json
```

## Run

```
python run.py --train
python run.py --test
python run.py --test samples/search_results_ground_truth_test.json
python run.py --train --subset holdout
python run.py samples/train_50.json --backend heuristic
python run.py samples/train_50.json --redecide samples/train_50.output.audit.json --conf 0.9
python run.py samples/train_50.json --require-confirmed
python run.py samples/train_50.json --no-cache --repeat 3
python run.py samples/train_50.json --repeat 3 --vote-min 3
python run.py big.json --batch-submit
python run.py big.json --batch-collect <batch_id>
python eval.py --train
python eval.py --test --predictions samples/search_results_ground_truth_test.output.json
python tests/test_pipeline.py
```

`--train` / `--test` pick the default JSON (`samples/train_50.json` vs
`samples/search_results_ground_truth_test.json`). Test files also carry
`trusted_search_results`, so without `--test` the CLI would print a meaningless
in-file hash **dev/holdout**. `--test` scores the **whole file** as one set.
`--subset` is train-only.

JSON reads/writes use **UTF-8** (Windows `cp1252` cannot store characters such as `⅔`).
Malformed top-level JSON or a top-level array exits with a one-line error (no traceback).
A product whose value is not an object is recorded as `invalid_product_format` and
**does not crash the rest of the batch**.

Unlabeled meeting files still write predictions; metrics print `scored: false`
instead of failing.

`python run.py -h` documents every flag.

## Input / output

Input: `{product_key: {product_title, product_description, search_results{"1": "Title:..\nLink:..\nSnippet:.."}}}`.

| File | Contents |
|---|---|
| `*.output.json` | Input + `llm_trusted_search_results: [int]` (the deliverable) |
| `*.output.audit.json` | Per product: status, `product_is_specific`, verdict / `variant_evidence` / confidence / reason; data-quality notes (`results_dropped`, `invalid_result_keys`, `upload_flag`) |
| `*.output.errors.json` | Labelled input only: every false / missed link with the model's reason |

Metrics (when `trusted_search_results` is present): overall always. **`--train`** also
prints the in-file hash split **dev (~70%) / holdout (~30%)** for prompt tuning.
**`--test`** prints overall only.

## Layout

| Path | Role |
|---|---|
| `run.py` | CLI: live / heuristic / `--redecide` / OpenAI Batch |
| `eval.py` | Eval harness (run or score existing output; unlabeled-safe) |
| `env_loader.py` | Loads `OPENAI_API_KEY`, `VERTEX_MODEL`, `VERTEX_CACHE_DIR` from `.env` |
| `selector/prompts.py` | Prompt **v3.1** + strict JSON schema |
| `selector/pipeline.py` | Sanitize, parallel `select_for_product`, `decide`, `assemble`, fail-closed |
| `selector/llm.py` | OpenAI structured outputs, retries, disk cache, token accounting |
| `selector/heuristic.py` | Offline lexical baseline |
| `selector/metrics.py` | F0.5, precision, clean-product, abstain; skip unlabeled products |
| `selector/batch.py` | `--batch-submit` / `--batch-collect` (same prepare / finalize / decide) |
| `samples/` | `train_50.json`, synthetics, `search_results_ground_truth_test.json`, `adversarial/` |
| `docs/product-owner-brief.md` | Trade-offs, production requirements, roadmap, clarifying questions |
| `docs/Vertex_Selector_Approach_Deck.pptx` | Stakeholder deck (`python scripts/build_deck.py`) |
| `tests/test_pipeline.py` | 24 offline tests (scripted fake LLM + robustness) |

Default model: `VERTEX_MODEL` in `.env`, else `gpt-4.1-mini`. CLI `--model` overrides.

## How a run works

`SelectorPipeline.run` fans products out on a thread pool (`--workers`, default 8).
Each worker runs `select_for_product` (prepare → LLM or heuristic → validate →
`decide`). `assemble` writes `llm_trusted_search_results` back onto a copy of the
input and builds ops stats. One product cannot crash the batch.

Duplicate URLs (same domain+path, ignoring tracking query params) are sent **once**;
the verdict is copied to the other copies. Results beyond 15 are dropped and listed
in the audit as `results_dropped`.

## Headline metric

A false link is worse than a missed link. Optimize **F0.5** and **clean-product rate**
(share of products with zero false links). Recall is reported as coverage cost.

## Formulas

Implemented in `selector/metrics.py` (`evaluate`). Per labeled product, `P` is the set
of predicted indices (`llm_trusted_search_results`) and `G` is the human set
(`trusted_search_results`). Products with **no** GT key are skipped (unlabeled), not
scored. An **empty** gold list `G = ∅` *is* labeled: humans trusted nothing.

### Link-level counts (pooled across all labeled products)

| Symbol | Meaning |
|---|---|
| TP | `\|P ∩ G\|` — links we trusted that humans also trusted |
| FP | `\|P − G\|` — links we trusted that humans did not (false links) |
| FN | `\|G − P\|` — links humans trusted that we missed |

If `TP + FP = 0`, precision is defined as `1`. If `TP + FN = 0`, recall is defined as `1`.

### Result metrics (the `metrics (all)` block)

| Field | Formula | What it means |
|---|---|---|
| `result_precision` | `TP / (TP + FP)` | Of links we would show, share the labelers also trusted |
| `result_recall` | `TP / (TP + FN)` | Of links humans trusted, share we found |
| `result_F0.5` | `F_β` with `β = 0.5` | Headline: precision weighted **2×** recall |
| `result_F1` | `F_β` with `β = 1` | Balanced harmonic mean (reported, not optimized) |
| `clean_product_rate` | `\|{ products with FP = 0 }\| / n` | Share of products with **zero** wrongly trusted links |
| `coverage_when_gt_exists` | `\|{ products with G ≠ ∅ and P ∩ G ≠ ∅ }\| / n_gt_pos` | Share of products that had a gold link where we found **at least one** correct link |
| `correct_abstain_rate` | `\|{ products with G = ∅ and P = ∅ }\| / n_gt_neg` | Share of “trust nothing” gold products where we also returned `[]` |
| `exact_match_rate` | `\|{ products with P = G }\| / n` | Share of products whose index set matches gold exactly |

`n` = labeled products (GT key present, including empty lists).
`n_gt_pos` = products with `G ≠ ∅`. `n_gt_neg` = products with `G = ∅`.

**F-beta** (same function for 0.5 and 1):

```
F_β = (1 + β²) · P · R / (β² · P + R)
```

If `P + R = 0`, `F_β = 0`. Values are rounded to 3 decimals.

### Worked example (held-out test, prompt v3.1)

Counts from the run: **TP = 211, FP = 36, FN = 30**, `n = 50`, `n_gt_pos = 46`, `n_gt_neg = 4`.

```
precision = 211 / (211 + 36) = 211 / 247 = 0.854
recall    = 211 / (211 + 30) = 211 / 241 = 0.876
F0.5      = 1.25 · 0.854 · 0.876 / (0.25 · 0.854 + 0.876) = 0.858
F1        = 2 · 0.854 · 0.876 / (0.854 + 0.876) = 0.865
clean_product_rate       = 28 / 50  = 0.56     (zero false links)
coverage_when_gt_exists  = 45 / 46  = 0.978    (≥1 correct link when gold exists)
correct_abstain_rate     =  2 /  4  = 0.50     (empty when humans trusted none)
exact_match_rate         = 13 / 50  = 0.26     (P equals G exactly)
```

### Other formulas in the repo

**Heuristic score** (`selector/heuristic.py`), then trust if `score ≥ --heur` (default 0.75):

```
score(i) = 0.5 · (hits_anywhere / |product_tokens|)
         + 0.5 · (hits_in_result_title / |product_tokens|)
```

**Majority vote** (`--repeat N`, `--vote-min`): a link is trusted when at least
`vote_min` samples trust it. Default `vote_min = ⌊N/2⌋ + 1` (strict majority).

**Dev / holdout split** (`run.py` `split_of`, printed only with `--train`): product key
hashed with MD5; `int(hex, 16) % 10 < 3` → holdout (~30%), else dev (~70%). `--test`
does not apply this split.

**LLM trust rule** (`decide`): index `i` is kept only if
`verdict = match` AND `confidence ≥ --conf` (default 0.70) AND no injection flag
AND `variant_evidence ≠ contradicted` (and, with `--require-confirmed`,
`variant_evidence = confirmed`).

## Results

### train_50 (in-sample; the same 50 informed the prompt)

| | lexical | LLM v1 | LLM v2 | LLM v3 / v3.1 |
|---|---|---|---|---|
| precision / recall / F0.5 | 0.67 / 0.80 / 0.70 | 0.80 / 0.91 / 0.82 | 0.86 / 0.92 / 0.87 | ~0.85 / ~0.93 / ~0.86 |
| clean-product / correct-abstain | 0.40 / 0.17 | 0.64 / 0.67 | 0.68 / 1.00 | (see run logs) |
| false / missed links | 93 / 48 | 56 / 22 | 35 / 20 | ~39 / ~18 |

v2: dev F0.5 0.90 (38 products), holdout 0.80 (12 products — too small to treat as a
generalization claim). Remaining errors mix label ambiguity (near-identical pages
labelled differently) with a few real model misses. Confidence barely separates
right from wrong, so `--conf` barely helps. `--require-confirmed` changed nothing
on v2 because `variant_evidence` was `confirmed` on every trusted link.

v3 adds: pack/count/jumbo cannot make a different variant. v3.1 is the same rules
with a compact `URL:` line and duplicate-URL dedupe (−15% input tokens on train_50).
v2 vs v3 F0.5 difference is inside run-to-run noise (~0.01).

### `--repeat 3` (train_50, prompt v3)

| | P | R | F0.5 | false / missed |
|---|---|---|---|---|
| sample 1 / 2 / 3 | .854 / .850 / .846 | .925 / .921 / .912 | .867 / .863 / .858 | 38/18, 39/19, 40/21 |
| voted 2 of 3 | .850 | .921 | .863 | 39 / 19 |
| voted 3 of 3 | .861 | .904 | .869 | 35 / 23 |

Only ~15 of 455 links split across samples; unanimous wrongs are systematic / label
issues that repeating the same prompt cannot fix. **Default remains `--repeat 1`.**

### Held-out test file (`samples/search_results_ground_truth_test.json`)

Prompt **v3.1**, 50 products, 8 workers, 16.5 s, 50 live calls, 57 duplicate URLs not sent:

| Metric | Value | Formula (this run) |
|---|---:|---|
| `result_precision` | 0.854 | 211 / 247 |
| `result_recall` | 0.876 | 211 / 241 |
| `result_F0.5` | 0.858 | F₀.₅(0.854, 0.876) |
| `result_F1` | 0.865 | F₁(0.854, 0.876) |
| `clean_product_rate` | 0.56 | 28 / 50 |
| `coverage_when_gt_exists` | 0.978 | 45 / 46 |
| `correct_abstain_rate` | 0.50 | 2 / 4 |
| `exact_match_rate` | 0.26 | 13 / 50 |

Counts: TP 211 / FP 36 / FN 30 (46 products with a non-empty gold list, 4 with none). Treat this as the unseen-data check, not another in-sample number.

### Cost / speed (v3.1)

On train_50 (single sample): ~1.9k input + ~0.3k output tokens/product; tens of seconds
at 8 workers. `--repeat N` multiplies both. `--redecide` retunes `--conf` /
`--require-confirmed` from a saved audit with **no API calls**.

OpenAI prompt-cache (`cached_prompt` in ops) is recorded when the prefix hits the
cacheable threshold; disk `.llm_cache` is a separate local replay cache.

## Robustness (adversarial samples)

`samples/adversarial/` plus unit tests cover: malformed JSON, top-level array,
non-object product values, null/wrong-type fields, non-numeric result keys, 25+
results, huge titles, injection in snippets **and** upload fields. Fail closed;
never wipe a successful product because a sibling row is garbage.

## Meeting artifacts

- Deck: `docs/Vertex_Selector_Approach_Deck.pptx`
- Leave-behind: `docs/product-owner-brief.md`
- Hidden-test command: `python run.py --test`
