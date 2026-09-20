# Brand Trust Agent

A verification pipeline for brand-safe campaign generation, instrumented with **Arize AX**.
Built for Verdant, a sustainable activewear brand, as a demonstration of claim
grounding, guardrails and per-step cost observability.

## Architecture

![Architecture Diagram](architecture.svg)

## What It Does

A **pipeline**, not an agent system. The control flow is fixed in `pipeline/run.py` — no model decides what happens next. The only branches are on data: did verification pass, did routing publish.

| Step | What it does | Span kind |
|------|--------------|-----------|
| **Retrieve** | ChromaDB over the brand docs. Plain function, no model call. | `RETRIEVER` |
| **Generate** | One `gpt-5.6-luna` call, structured output: copy **plus an explicit `claims[]` list**, each claim quoting its source. | `LLM` |
| **Verify** | Deterministic checks (free), then the `gpt-5.6-terra` judge only on survivors. | `GUARDRAIL` → `TOOL` + `LLM` |
| **Repair** | One rewrite against the contradicting source line. Hard cap of one attempt. | `LLM` |
| **Re-verify** | The repair gets checked too. Nothing is trusted for having been rewritten. | `GUARDRAIL` |
| **Route** | Publish with receipts, flag to human, or sample for audit. Plain function. | `CHAIN` |
| **Assets** | Veo video, only once routing publishes. | `TOOL` |

```
campaign-run                  CHAIN   (root)
├── retrieve                  RETRIEVER
├── generate                  LLM
├── verify                    GUARDRAIL
│   ├── deterministic-checks  TOOL
│   └── grounding-judge       LLM        (only if deterministic passed)
├── repair                    LLM        (only on failure)
├── re-verify                 GUARDRAIL  (only on failure)
├── route                     CHAIN
└── assets                    TOOL       (only when routing publishes)
```

**The guardrail** (`evals/grounding_check.py`) compares the copy against *the chunks actually retrieved*, not the full corpus — so a retrieval failure is visible rather than silently papered over. It catches claims that quietly broaden a narrow source ("our Portugal factory is Fair Trade certified" → "our manufacturing is Fair Trade certified"), and the halt reason names the specific claim.

The retrieval-quality score is **reported, not gating**. It is built from retrieval signals and never sees the generated text, so a perfect retrieval plus a fabricating model scored high and passed.

**Routing and the audit sample.** A flagged campaign goes to a human and does not publish. A passing campaign publishes — *including when it is sampled*. Sampling audits what normally happens, so a sampled item takes the normal path and the review copy goes out in parallel. Holding 10% of approved content back would create a second behaviour and mean observing that instead.

**Cost.** Every step's token usage is measured and priced; `cost.step_tokens` / `cost.step_usd` sit on each span and the total on the root, so cost per campaign is visible broken down by step. A typical clean run is about 5,000 tokens and $0.0026. Where a price is unknown or a run crosses into the long-context tier, that is flagged rather than quietly under-reported.

## Evals

Three scripts, three different questions. They were one script; that was the problem.

```bash
python3 -m evals.eval_retrieval     # Did search find the right chunks?
python3 -m evals.eval_end_to_end    # Does the pipeline publish anything bad?
python3 -m evals.eval_judge         # Does the judge agree with a fixed label?
```

**Retrieval eval** — deterministic, no judge, costs a fraction of a cent. Checks whether the facts a brief needs are actually in the retrieved chunks, using the golden dataset's `ground_truth_answer_contains` as ground truth. Current result: **70% mean fact recall @4, 9 of 20 briefs missing at least one required fact — while all 20 scored "high" grounding.** That gap is the whole argument for not gating on the grounding score.

**End-to-end eval** — runs the real pipeline. The number that matters is **escapes**: campaigns routed to publish that still contain a claim the brief's row lists as prohibited. Its inputs move whenever the generator changes, so it is a snapshot, not a measurement.

**Judge eval** — frozen text, fixed hand-assigned labels, no pipeline at all. Sixteen cases in `data/judge_eval_set.csv` that never change, so a prompt or model change is measured against a fixed target.

> The labels in `data/judge_eval_set.csv` were hand-assigned against the brand documents and **need review by someone who owns the brand voice**. Read them before trusting the numbers.

Keeping these separate matters. When they were one script, the judge was scored against `expected_hallucination` — a column describing how risky a *query* looked when the dataset was written, not whether the *generated text* contained a false claim. That produced TPR 0.00 for a judge that was right on every disputed row.

## Series Mode

The app runs a campaign series — up to five cycles — the way an autonomous content calendar would.

Each cycle's brief is the original brief, plus the copy that was actually approved and published last cycle, plus what the audience said about it. Nothing else carries forward. Specifically:

- **Brand facts are re-retrieved from source every cycle.** No chunk, fact or summary from a prior cycle is reused. Each cycle asks the documents the same question and gets whatever they currently say.
- **Rejected drafts are never passed forward.** A cycle that fails the grounding check publishes nothing, so it contributes no copy and collects no audience comments. The next cycle builds on the last campaign that actually shipped.
- **No engagement metrics.** There is no CTR, no engagement rate, no view count and no "insight" telling the next cycle what to lean into. Audience comments are carried because they are what the next brief is responding to, not because they are a score to optimize.

The pressure the series puts on the system is real without being simulated: audience comments contain things Verdant has not earned ("carbon neutral", "B Corp certified soon?"), and each cycle has to answer them from the documents rather than from the flattery.

**The grounding check runs every cycle, before any video is generated.** On a flag, the copy is rewritten once against the source line the claim ran into, and re-checked once. Two attempts, hard cap. If the second check fails, the cycle halts and the reviewer sees both drafts, the flagged claim, the source line that contradicts it, and the reason.

Every cycle is recorded: what was flagged, which layer caught it, whether the rewrite fixed it. The record is shown in the UI as a per-cycle table and written to Arize as `campaign.*` and `trust.*` span attributes.

## Demo Scenarios

Use the sidebar toggle to switch between scenarios:

1. ✅ **Normal run** — retrieval, generation, verification, publish with receipts
2. ⚠️ **Weak retrieval** — one chunk instead of four, so claims may have no retrieved source to check against
3. 🔴 **Prompt injection** — an adversarial document tries to hijack the output; Verify reads the generated text, not the instruction

The old "trust gap" and "trust-aware" scenarios are gone. They existed so Agent 2 could inherit Agent 1's confidence score; there are no agents and nothing to propagate between.

## Setup

### 1. Clone and install

```bash
git clone https://github.com/rr0214/brand-trust-agent
cd brand-trust-agent
pip install -r requirements.txt
```

### 2. Configure credentials

Create `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "sk-..."
GOOGLE_API_KEY = "..."        # For Veo video generation (optional)
ARIZE_API_KEY = "..."
ARIZE_SPACE_ID = "..."
```

You need:
- **OpenAI API key** — [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **Arize AX Space ID + API Key** — [app.arize.com](https://app.arize.com) → Settings → API Keys
- **Google API key** — [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) *(optional — video generation only)*

> The app runs fully without a Google API key. Agent 3 will generate the video prompt but skip video generation. All trust gate logic, tracing, and caption generation still work.

### 3. Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

## Project Structure

```
brand-trust-agent/
├── app.py                              # Streamlit entry point. Tracing initialized here.
├── agents/                             # Legacy three-agent implementation, superseded
│   ├── brand_research_agent.py         # Agent 1: ChromaDB RAG + grounding score
│   ├── campaign_strategy_agent.py      # Agent 2: Strategy + brand policy tool calls
│   └── creative_execution_agent.py     # Agent 3: Trust gate + Veo prompt + caption
├── brand_policy.py                     # Approved/prohibited claim tables + policy check
├── pipeline/
│   ├── retrieval.py                    # RETRIEVE + query derivation
│   ├── steps.py                        # GENERATE, VERIFY, REPAIR, ROUTE, ASSETS
│   ├── run.py                          # Fixed control flow + tracing
│   ├── pricing.py                      # Token accounting, USD, tier flagging
│   └── usage_capture.py                # Token capture for calls we don't own
├── evals/
│   ├── grounding_check.py              # Deterministic checks + LLM judge. The gate Agent 3 calls.
│   └── run_evals.py                    # Golden-dataset runner + judge validation (TPR/TNR)
├── instrumentation/
│   └── arize_setup.py                  # Arize OTel setup. Pure instrumentation — no agent logic.
├── data/
│   └── brand_docs/
│       ├── verdant_brand_guide.txt
│       ├── verdant_products.txt
│       ├── verdant_sustainability.txt
│       └── verdant_poisoned.txt        # Prompt injection demo document
├── architecture.svg
└── requirements.txt
```

---
Built by Rebecca Riggs · 2026
