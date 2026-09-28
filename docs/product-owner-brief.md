# Search Result Selector — product-owner brief

Stakeholder leave-behind for the AI Product Engineer case. use this note for
questions, trade-offs, production requirements, and the roadmap.

## What we own

Tax departments upload a title (and sometimes a description). Vertex must
enrich that row from the web so O-Series can map it to a tax category. A
wrong page means wrong ingredients, a wrong category, and a customer who
stops trusting the links we show them.

This case is **only the selection step**: given already-retrieved search
results, decide which ones are about the uploaded product. Live web search,
scraping, and tax mapping are the next stages of the enrichment pipeline —
not this deliverable.

**Contract**

- In: `{product_title, product_description, search_results}`
- Out: same JSON + `llm_trusted_search_results: [int, …]`
- Empty list = abstain (no enrichment from the web for that row)

## Metric we optimize

A false link is worse than a missed link. False links poison classification
and are user-visible. Missed links only reduce coverage.

| Metric | Role |
|---|---|
| **F0.5** | Headline. Precision weighted 2× recall. |
| **Result precision** | Of the links we would show, share the labelers also trusted. |
| **Clean-product rate** | Share of products with *zero* false links. This is the trust metric. |
| Recall / coverage-when-GT | Cost of being strict. |
| Correct-abstain rate | Did we stay silent on generic / unlinkable items? |

Launch recommendation (to confirm with the business): **do not ship a
threshold that drops clean-product rate to chase recall.** Prefer fewer,
correct links.

## Measured results (train_50, in-sample)

Same 50 products informed the prompt rules. Treat these as development
numbers, not a generalization claim.

| Backend | Precision | Recall | F0.5 | Clean-product |
|---|---:|---:|---:|---:|
| Heuristic baseline | 0.674 | 0.800 | 0.696 | 0.400 |
| LLM (`gpt-4.1-mini`, conf ≥ 0.70) | 0.796 | 0.908 | 0.816 | 0.640 |

LLM also abstains more correctly on generic items (correct-abstain 0.667 vs
0.167 lexical). Coverage when a labeled link exists: 0.974.

The ceiling is below 1.0 even with a perfect model: 12 of 50 products have an
empty gold list, and 7 of 82 duplicate-URL groups have one copy trusted and
the other not (Theraworx, V8, Sparkling Ice, …). Error analysis has to
separate model misses from label noise.

## Trade-offs we are making

1. **Precision over recall.** We will leave enrichment holes rather than
   attach a sibling flavor, a listing page, or a spiked/diet variant. Ask
   the business to accept lower coverage on messy ERP titles in exchange
   for not showing a wrong URL.
2. **Snippet-only judgment.** We do not fetch the page. Cheap and fast;
   we will miss cases where the title is right and the snippet is a
   site-nav junk drawer. Page-fetch is the first roadmap item after this
   step is accepted.
3. **Fail closed.** API errors, refusals, schema misses, prompt-injection
   flags, and unspecific products all yield `[]`. One bad product cannot
   crash a 100k batch; it also cannot silently trust a guess.
4. **LLM default, heuristic fallback.** The lexical baseline exists so the
   meeting and CI can run without a key, and so we can sample large
   LLM-vs-lexical disagreements for human review. It is not the production
   path.
5. **One call per product, all results together.** The model can see that
   3 of 10 hits are the exact flavor and the rest are siblings. Cost is
   ~1 request × N products, not N × R. We cap at 15 results.
6. **Size is ignored; flavor/form is not.** Matches the brief. This will
   disagree with any labeler who treated “12 oz vs 20 oz” as a miss, or
   who treated “decaf vs regular” as a match.

## Production requirements (what must be true to serve hundreds of clients)

- **Pinned model + prompt version** in the audit log (`VERTEX_MODEL`,
  prompt hash). A prompt edit is a release.
- **Structured outputs** (JSON schema with an enum of legal indices) so
  the model cannot invent result #99.
- **Confidence threshold** tunable without a re-call (`--conf`, or
  re-threshold from `*.audit.json`).
- **Disk cache** for idempotent replays; `--no-cache` for a clean live run.
- **Concurrency** (`--workers`, default 8) with per-product isolation.
- **Audit trail** written next to the output: verdict, confidence, reason,
  injection flag. This is what tax analysts and we debug from.
- **Key required for uncached LLM calls.** Cached replays (train_50) work
  without a key. A new test file needs `OPENAI_API_KEY` — `run.py` now loads
  it automatically from a local `.env` (via `env_loader.py`, the same
  mechanism `eval.py` already used), so it no longer has to be typed into
  the shell during the meeting; an explicit `$env:OPENAI_API_KEY=...` still
  overrides the `.env` value if you need to swap keys for one run.
- **Human-review sample** of (a) false-link products and (b) LLM vs
  heuristic disagreements, every release.
- **Launch bar (proposed, not agreed):** clean-product ≥ 0.80 and
  precision ≥ 0.85 on a held-out set the labelers have not seen. Train_50
  is not that set.
- **Override path:** an analyst must be able to add/remove a trusted
  index; we should log it as future gold.
- **Cost envelope (order of magnitude):** 100k products × 1 call ×
  gpt-4.1-mini. Confirm budget and latency SLO (see questions).
- **No silent backend swap** in production. If OpenAI is down, fail
  closed or pause the job — do not quietly switch to lexical.

## Roadmap

**Now (this case)**
Selector pipeline, precision-first metrics, three synthetic smoke tests,
guardrails, heuristic baseline, this brief + deck. Ready to run the
hidden test JSON in the meeting.

**Next 2 weeks**
- Second labeler pass on the 18 train_50 false-link products; measure
  inter-annotator agreement on the “grey” items the brief warned about.
- Re-threshold from the audit file; do not re-prompt until labels settle.
- Optional: fetch the page for `match` candidates below 0.85 confidence
  and re-judge. Only if the business wants coverage back.

**Next quarter**
- Stage 2: scrape trusted URLs into an enriched description (ingredients,
  carbonation, juice %, sweetener, alcohol).
- Stage 3: classify the enriched description to a Vertex tax category,
  with the same fail-closed / audit pattern.
- Online monitors: clean-product proxy via analyst overrides, cost per
  1k SKUs, p95 latency, injection-flag rate.
- Client-specific abbreviation tables (the ERP junk is not universal).

**Later**
Per-jurisdiction policy if taxability (not just components) starts to
change what “same product” means. Active learning from analyst overrides.

## Clarifying questions

These are the questions I would ask as the owner of this step.

1. **Gold labels.** The brief says team members disagree. Whose
   `trusted_search_results` is authoritative, and do we have a second
   annotator on the 50? What is the expected agreement rate?
2. **Hidden test.** Will the meeting file include `trusted_search_results`?
   The harness scores a labeled subset and still writes output if labels
   are omitted — I need to know which print you want on the screen.
3. **Form changes.** Ready-to-drink vs mix vs tea bags vs concentrate of
   the same brand/flavor: match or different product? Components differ.
4. **Alcohol / diet / decaf / “lite” / “spiked”.** Confirm these are
   always different variants, including when the uploaded title is silent.
5. **Private label, produce, deli, bakery, food-service “Apple”.** We
   abstain. Is that also the product-UI policy (show “not enough to link”)
   or should we still surface a generic education page?
6. **Duplicates.** Same product on Amazon + Kroger + manufacturer: trust
   all matching pages, or one canonical URL? Enrichment cost scales with
   the list we return.
7. **User-visible set.** Do we show every trusted index, or top-N by
   confidence? That changes the precision/recall target.
8. **Analyst override.** Can tax users add/remove a link in the app, and
   may we treat those overrides as future training labels?
9. **Latency and cost SLO.** What is acceptable p95 per product and
   budget per 100k-SKU job? That decides model class and whether we
   page-fetch.
10. **Search ownership.** Who owns the query, engine, and result count?
    Can this team change retrieval, or only judge the results we are given?
11. **Data handling.** May product titles (sometimes customer-specific
    SKU strings) be sent to OpenAI, and for how long may we retain the
    audit log?
12. **Launch bar and abstain UX.** What clean-product / precision bar
    unlocks the rest of Smart Categorization? When we return `[]`, does
    the product stay title-only, queue for a human, or fall back to a
    category prior?

## How we will run the hidden test

```
# Windows, production path — OPENAI_API_KEY loads automatically from .env
python run.py test.json
# or the eval harness (same run, formatted report)
python eval.py test.json
# only needed to override the .env key for one run:
$env:OPENAI_API_KEY='sk-...'; python run.py test.json
```

- Output: `test.output.json` + `test.output.audit.json` while we talk.
- If the file has labels, the headline numbers print immediately.
- If it does not, you still get the predictions; metrics say `scored: false`.
- No key (and no `.env`): cached products replay; uncached products fail
  closed. Offline demo: `python run.py test.json --backend heuristic`.
