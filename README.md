# Brand Trust Agent

A three-agent pipeline for brand-safe campaign generation, instrumented with **Arize AX**.
Built for Verdant, a sustainable activewear brand, as a demonstration of multi-agent
observability and trust signal propagation.

## Architecture

![Architecture Diagram](architecture.svg)

## What It Does

| Agent | Role | Arize Spans |
|-------|------|-------------|
| **Agent 1: Brand Research** | RAG over Verdant brand docs via ChromaDB. Returns a grounding score (0–1) based on retrieval quality — reported, not used as the gate. | `RETRIEVER` + `LLM` |
| **Agent 2: Campaign Strategy** | Builds campaign strategy from research. Validates every factual claim via `check_brand_policy()` tool call. | `AGENT` → `LLM` → `TOOL` → `LLM` |
| **Agent 3: Creative Execution** | Generates caption, hashtags, and Veo video prompt. Only runs if trust gate passes. | `AGENT` → `LLM` → `TOOL` (Veo) → `LLM` |

**Trust gate:** Before Agent 3 generates anything, the campaign text is checked against the brand documents Agent 1 actually retrieved (`evals/grounding_check.py`). Cheap checks run first — prohibited phrases, statistics outside the approved set, competitor names — in plain Python, no model calls. Only text that survives them reaches an LLM judge, which looks for claims the sources do not support, including claims that quietly broaden a narrow source ("our Portugal factory is Fair Trade certified" → "our manufacturing is Fair Trade certified"). Content is blocked, not just flagged, and the halt reason names the specific claim.

The grounding score no longer gates. It is built from retrieval signals and never saw the generated text, so a perfect retrieval plus a fabricating model scored high and passed.

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

1. ✅ **Normal run** — all three agents complete, creative package delivered
2. ⚠️ **Trust gap (silent failure)** — weak retrieval, Agent 2 operates without grounding context
3. 🔴 **Prompt injection** — adversarial document hijacks Agent 2 output
4. 🛡️ **Trust-aware mode (the fix)** — grounding score and injection risk propagate end-to-end

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
├── agents/
│   ├── brand_research_agent.py         # Agent 1: ChromaDB RAG + grounding score
│   ├── campaign_strategy_agent.py      # Agent 2: Strategy + brand policy tool calls
│   └── creative_execution_agent.py     # Agent 3: Trust gate + Veo prompt + caption
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
