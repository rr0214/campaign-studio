"""
Campaign Studio — Brand-safe campaign generation
Three trust-aware AI agents
"""

import io
import os
import time
from datetime import date, timedelta
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

# ---------------------------------------------------------------------------
# Arize tracing — initialize once per process
# @st.cache_resource ensures this runs exactly once even across Streamlit rerenders.
# This registers the OTLP exporter and auto-instruments all OpenAI SDK calls.
# Custom spans created with trace.get_tracer() below will share this provider.
# ---------------------------------------------------------------------------
@st.cache_resource
def _init_tracing():
    from instrumentation.arize_setup import setup_arize_tracing
    try:
        setup_arize_tracing()
    except Exception as e:
        # Don't crash the app if Arize keys aren't set (e.g. local dev without .env)
        import warnings
        warnings.warn(f"Arize tracing not configured: {e}")

_init_tracing()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Campaign Studio",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,800&family=Outfit:wght@300;400;500;600;700&display=swap');

/* ──────────────────────────────────────────────────────────────────────────
   Bold, not corporate. Colour-blocked, chunky, high contrast.
   Display: Bricolage Grotesque. Interface: Outfit.
   Shapes are defined by fill and shadow — there is not a single 1px border.
   ────────────────────────────────────────────────────────────────────────── */
:root {
    --green:      #0E3B2E;
    --green-soft: #17594A;
    --lime:       #B8F24A;
    --terra:      #E8613C;
    --indigo:     #4F46E5;
    --amber:      #F4A521;
    --ink:        #0B0F0D;
    --ink-2:      #3A4440;
    --ink-3:      #6B7671;
    --paper:      #FAF7F2;
    --card:       #FFFFFF;
    --tint:       #F1EDE6;
}

html, body, [class*="css"] {
    font-family:'Outfit', sans-serif; color:var(--ink-2);
    -webkit-font-smoothing:antialiased;
}
.stApp { background:var(--paper); }
h1,h2,h3 { font-family:'Bricolage Grotesque', sans-serif; color:var(--ink); font-weight:800; }
section.main > div { max-width:920px; margin:0 auto; padding:0 1.5rem 6rem 1.5rem; }
div[data-testid="stVerticalBlock"] > div { gap:0.6rem; }
hr, div[data-testid="stDivider"] { display:none !important; }

/* ── Header band ───────────────────────────────────────────────────────── */
.band {
    background:var(--green); border-radius:28px;
    padding:52px 44px 46px 44px; margin:14px 0 38px 0;
    box-shadow:0 18px 48px rgba(14,59,46,.28);
}
.band-title {
    font-family:'Bricolage Grotesque', sans-serif; font-size:4rem; font-weight:800;
    color:var(--paper); line-height:0.94; letter-spacing:-0.035em; margin:0;
}
.band-title em { font-style:normal; color:var(--lime); }
.band-sub {
    font-size:1.02rem; color:rgba(250,247,242,.72); margin-top:16px;
    font-weight:400; max-width:34rem; line-height:1.5;
}

/* ── Buttons — chunky, weighted, lift on hover ─────────────────────────── */
div[data-testid="stButton"] button {
    border:none !important; border-radius:999px; padding:0.62rem 1.3rem;
    background:var(--tint); color:var(--ink); font-family:'Outfit', sans-serif;
    font-weight:600; font-size:0.9rem; box-shadow:0 2px 0 rgba(11,15,13,.10);
    transition:transform .12s ease, box-shadow .12s ease, background .12s ease;
}
div[data-testid="stButton"] button:hover {
    background:#E6E0D6; transform:translateY(-2px);
    box-shadow:0 8px 18px rgba(11,15,13,.14);
}
div[data-testid="stButton"] button[kind="primary"],
div[data-testid="stFormSubmitButton"] button {
    background:var(--green) !important; color:var(--paper) !important;
    font-weight:700; font-size:1rem; padding:0.85rem 2rem;
    box-shadow:0 4px 0 #082A20, 0 10px 24px rgba(14,59,46,.28);
}
div[data-testid="stButton"] button[kind="primary"]:hover {
    transform:translateY(-2px); box-shadow:0 6px 0 #082A20, 0 16px 34px rgba(14,59,46,.34);
}

/* inputs: filled, round, borderless */
textarea, input, div[data-baseweb="input"], div[data-baseweb="textarea"] {
    border:none !important; border-radius:18px !important;
    background:var(--card) !important; box-shadow:0 2px 0 rgba(11,15,13,.06) !important;
    font-family:'Outfit', sans-serif !important; color:var(--ink) !important;
}
div[data-testid="stExpander"] {
    border:none !important; border-radius:20px; background:var(--card);
    box-shadow:0 2px 0 rgba(11,15,13,.05), 0 10px 26px rgba(11,15,13,.05);
    margin-bottom:12px;
}
div[data-testid="stExpander"] summary {
    font-family:'Outfit', sans-serif; font-weight:600; color:var(--ink); font-size:0.95rem;
}
div[data-testid="stExpander"] details { border:none !important; }

/* ── Labels ────────────────────────────────────────────────────────────── */
.section-label {
    font-size:0.7rem; font-weight:700; letter-spacing:0.16em;
    text-transform:uppercase; color:var(--ink-3); margin:0 0 14px 0;
}
.tier-gap { height:64px; } .tier-gap-sm { height:34px; }

/* ── Status — sticker chips ────────────────────────────────────────────── */
.chip {
    display:inline-flex; align-items:center; gap:9px;
    border-radius:999px; padding:12px 22px 12px 18px;
    font-weight:700; font-size:1rem; letter-spacing:-0.01em;
    box-shadow:0 3px 0 rgba(11,15,13,.14), 0 10px 24px rgba(11,15,13,.10);
}
.chip-sub {
    font-family:'Outfit', sans-serif; font-size:0.9rem; color:var(--ink-3);
    font-weight:400; margin:12px 0 0 4px; line-height:1.55;
}
.c-live    { background:var(--lime);   color:#16310A; }
.c-sample  { background:var(--indigo); color:#fff; }
.c-review  { background:var(--terra);  color:#fff; }
.c-unknown { background:var(--amber);  color:#3A2606; }

/* ── The post ──────────────────────────────────────────────────────────── */
.post {
    background:var(--card); border-radius:28px; padding:0 0 30px 0;
    box-shadow:0 3px 0 rgba(11,15,13,.06), 0 22px 54px rgba(11,15,13,.10);
    overflow:hidden; margin-bottom:20px;
}
.post-inner { padding:0 30px; }
.post-head { display:flex; align-items:center; gap:13px; padding:24px 30px 18px 30px; }
.post-avatar {
    width:42px; height:42px; border-radius:999px; background:var(--green);
    color:var(--lime); font-size:0.95rem; font-weight:800;
    font-family:'Bricolage Grotesque', sans-serif;
    display:flex; align-items:center; justify-content:center;
}
.post-handle { font-size:0.98rem; font-weight:700; color:var(--ink); line-height:1.2; }
.post-meta   { font-size:0.78rem; color:var(--ink-3); font-weight:500; }
.post-caption {
    font-family:'Outfit', sans-serif; font-size:1.2rem; line-height:1.62;
    color:var(--ink); margin:26px 0 16px 0; font-weight:400;
}
.post-tags { font-size:0.95rem; color:var(--green-soft); font-weight:600; line-height:1.7; }
.post-empty {
    background:var(--tint); padding:76px 24px; text-align:center;
    color:var(--ink-3); font-size:0.9rem; font-weight:500;
}

/* ── The thinking — oversized display type ─────────────────────────────── */
.think-block { background:var(--card); border-radius:28px; padding:40px 36px;
    box-shadow:0 3px 0 rgba(11,15,13,.05), 0 16px 40px rgba(11,15,13,.07); }
.think-tagline {
    font-family:'Bricolage Grotesque', sans-serif; font-size:3.4rem; font-weight:800;
    color:var(--ink); line-height:0.98; letter-spacing:-0.035em; margin:4px 0 22px 0;
}
.think-concept { font-size:1.06rem; line-height:1.68; color:var(--ink-2); margin-bottom:26px; }
.think-msg {
    font-size:0.98rem; color:var(--ink); line-height:1.55; font-weight:500;
    background:var(--tint); border-radius:16px; padding:15px 20px; margin-bottom:10px;
}

/* ── Halt walkthrough ──────────────────────────────────────────────────── */
.seq-step { display:flex; gap:16px; margin-bottom:24px; }
.seq-num {
    flex:0 0 32px; height:32px; border-radius:999px; background:var(--terra);
    color:#fff; font-size:0.85rem; font-weight:800;
    font-family:'Bricolage Grotesque', sans-serif;
    display:flex; align-items:center; justify-content:center;
}
.seq-body { flex:1; }
.seq-title {
    font-size:0.68rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.16em; color:var(--ink-3); margin-bottom:8px;
}
.seq-text {
    font-size:0.98rem; color:var(--ink-2); line-height:1.7; white-space:pre-wrap;
    background:var(--card); border-radius:18px; padding:18px 22px;
    box-shadow:0 2px 0 rgba(11,15,13,.05);
}
.claim-hl { background:var(--terra); color:#fff; padding:2px 7px; border-radius:7px; font-weight:700; }

/* ── Receipts ──────────────────────────────────────────────────────────── */
.receipt { background:var(--card); border-radius:18px; padding:17px 21px; margin-bottom:11px;
    box-shadow:0 2px 0 rgba(11,15,13,.05); }
.receipt-missing { background:#FDF3E0; }
.receipt-claim { font-size:0.95rem; color:var(--ink); font-weight:600; line-height:1.5; }
.receipt-src {
    font-size:0.92rem; color:var(--ink-3); margin-top:9px;
    padding:10px 14px; background:var(--tint); border-radius:12px; line-height:1.6;
}
.dot { display:inline-block; width:9px; height:9px; border-radius:999px; margin-right:10px; vertical-align:middle; }
.dot-ok{background:#2F9E44;} .dot-warn{background:var(--amber);}
.dot-bad{background:var(--terra);} .dot-idle{background:var(--ink-3);}

/* ── Sidebar diagnostics ───────────────────────────────────────────────── */
section[data-testid="stSidebar"] { background:var(--tint); }
.diag-row {
    display:flex; align-items:center; gap:9px; font-size:0.82rem; font-weight:500;
    color:var(--ink-2); padding:9px 13px; margin-bottom:6px;
    background:var(--card); border-radius:12px;
}
.diag-row .t { margin-left:auto; font-variant-numeric:tabular-nums; color:var(--ink-3); }
.diag-total { font-size:0.92rem; color:var(--ink); font-weight:700; padding:10px 0 6px 0; }

.no-video { background:var(--tint); border-radius:20px; padding:56px 24px;
    text-align:center; color:var(--ink-3); font-size:0.9rem; font-weight:500; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — pipeline status + demo mode (secondary)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Pipeline Status")
    pipeline_status = st.empty()
    pipeline_status.caption("Run a campaign to see agent activity here.")

    st.markdown("---")
    st.markdown("### ⚙️ Settings")

    # trust_aware is gone. It existed so one agent could inherit another's confidence;
    # there are no agents and nothing to propagate between. The remaining toggles
    # both change what RETRIEVE returns, which is the only input VERIFY can react to.
    FAILURE_CONFIGS = {
        "✅ Normal run": {
            "include_poisoned": False, "simulate_low_confidence": False,
        },
        "⚠️ Weak retrieval": {
            "include_poisoned": False, "simulate_low_confidence": True,
        },
        "🔴 Prompt injection": {
            "include_poisoned": True, "simulate_low_confidence": False,
        },
    }

    failure_mode = st.segmented_control(
        "Demo scenario",
        options=list(FAILURE_CONFIGS.keys()),
        default="✅ Normal run",
    )
    config = FAILURE_CONFIGS[failure_mode]

    if failure_mode != "✅ Normal run":
        st.caption({
            "⚠️ Weak retrieval": "One chunk instead of four. Claims the copy makes may have no retrieved source to check against.",
            "🔴 Prompt injection": "A hidden instruction in a brand doc tries to hijack the output. Verify reads the text, not the instruction.",
        }[failure_mode])

    st.markdown("---")
    st.markdown("[Open Arize AX ↗](https://app.arize.com)")
    st.caption("Projects → brand-trust-agent → Traces")

    st.markdown("---")
    with st.expander("📚 Brand document index", expanded=False):
        st.caption("The source documents Retrieve searches. These are the ground truth for every brand claim.")
        docs = ["verdant_brand_guide.txt", "verdant_products.txt", "verdant_sustainability.txt"]
        if config["include_poisoned"]:
            docs.append("verdant_poisoned.txt ⚠️")
        for d in docs:
            st.markdown(f"{'🔴' if 'poisoned' in d else '🟢'} `{d}`")

        try:
            from pipeline.retrieval import _get_collection
            coll = _get_collection(include_poisoned=config["include_poisoned"])
            st.caption(f"{coll.count()} chunks indexed")
        except Exception:
            pass

    with st.expander("❓ How it works", expanded=False):
        st.markdown("""
**A pipeline, not agents.** The control flow is fixed — no model decides what happens next.

1. **Retrieve** — ChromaDB over the Verdant brand docs. A plain function, no model call. Returns chunks and a retrieval-quality score. That score is *reported, not gating*: it measures whether search found relevant documents, never whether the copy is true.

2. **Generate** — one call to `gpt-5.6-luna`, structured output. Takes the brief, audience comments, retrieved chunks and brand voice rules. Returns the campaign **plus an explicit list of every factual claim it made**, each with the source line it believes supports it.

3. **Verify** — the guardrail. Deterministic checks first, free and in plain Python: prohibited phrases, statistics outside the approved set, competitor names. Only text that survives reaches the judge (`gpt-5.6-terra`), which compares the copy against *the chunks actually retrieved* — so a retrieval failure is visible, not silently papered over.

4. **Repair** — one rewrite, only on failure, anchored to the source line the claim ran into. Re-checked once. Hard cap at one attempt.

5. **Route** — publish with receipts, or human review. Plus a random 10% of *passing* campaigns copied to review as an audit sample. Sampled campaigns publish normally — sampling observes what usually happens; holding it back would create a second behaviour and measure that instead.

6. **Assets** — video, only once routing publishes.

Everything is traced and costed per step, so you can see what each campaign spent and where.
        """)

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.markdown("""
<div class='band'>
  <div class='band-title'>Campaign<br>Studio<em>.</em></div>
  <div class='band-sub'>Brand-safe social campaigns for Verdant — every claim traced
  back to a source document before anything ships.</div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# Campaign input
# ---------------------------------------------------------------------------
EXAMPLES = [
    "Tell our recycled materials story for a spring launch",
    "Show people where our clothes are actually made",
    "Show what happens to garments sent back through our Take Back Program",
    "Make the case for buying one thing that lasts, for holiday gifting",
]

st.markdown("<div class='section-label'>Start from an example</div>", unsafe_allow_html=True)
ex_cols = st.columns(len(EXAMPLES))
for _col, _ex in zip(ex_cols, EXAMPLES):
    with _col:
        if st.button(_ex, key=f"ex_{hash(_ex)}", width='stretch'):
            st.session_state["user_prompt"] = _ex
            st.rerun()

user_prompt = st.text_area(
    "What would you like to campaign about?",
    value=st.session_state.get("user_prompt", ""),
    height=100,
    placeholder="Describe your goal, audience, and angle — e.g. Spring launch targeting Gen Z runners, lead with our recycled materials story",
)

# ── Series configuration ──────────────────────────────────────────────────
col_series_l, col_series_r = st.columns(2)
with col_series_l:
    num_campaigns = st.pills("Number of campaigns", options=[1, 2, 3, 4, 5], default=1,
                             key="num_campaigns")
    if num_campaigns is None:
        num_campaigns = 1
with col_series_r:
    schedule = st.pills("Drop schedule", options=["Daily", "Weekly", "Every 2 weeks"], default="Weekly")
    if schedule is None:
        schedule = "Weekly"

SCHEDULE_DELTAS = {"Daily": timedelta(days=1), "Weekly": timedelta(weeks=1), "Every 2 weeks": timedelta(weeks=2)}

btn_label = "▶  Generate Campaign" if num_campaigns == 1 else f"▶  Generate {num_campaigns}-Campaign Series"
run_btn = st.button(btn_label, type="primary")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def render_status(run):
    cls, head, sub = status_state(run)
    chip = {"s-live": "c-live", "s-sample": "c-sample",
            "s-review": "c-review", "s-unknown": "c-unknown"}[cls]
    st.markdown(
        f"<div class='chip {chip}'>{head}</div>"
        f"<div class='chip-sub'>{sub}</div>",
        unsafe_allow_html=True,
    )


def render_post(run):
    """
    TIER 1 — the post. Laid out as a feed preview, because that is what a
    marketer is approving: video, caption as running copy, hashtags muted.
    Not a form of labelled fields.
    """
    draft, assets = run.draft, run.assets

    st.markdown("<div class='post'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='post-head'><div class='post-avatar'>V</div>"
        "<div><div class='post-handle'>verdant</div>"
        "<div class='post-meta'>Sponsored · Draft preview</div></div></div>",
        unsafe_allow_html=True,
    )

    if assets and assets.video_bytes:
        st.video(assets.video_bytes)
    else:
        note = ("No video — this campaign was not published, so the asset step never ran."
                if not run.published else
                (f"Video unavailable — {assets.error[:140]}" if (assets and assets.error)
                 else "No video available."))
        st.markdown(f"<div class='post-empty'>{note}</div>", unsafe_allow_html=True)

    st.markdown("<div class='post-inner'>", unsafe_allow_html=True)
    if draft.caption:
        st.markdown(f"<div class='post-caption'>{draft.caption}</div>", unsafe_allow_html=True)
    if draft.hashtags:
        st.markdown(f"<div class='post-tags'>{' '.join(draft.hashtags)}</div>",
                    unsafe_allow_html=True)
    st.markdown("</div></div>", unsafe_allow_html=True)


def render_thinking(run):
    """
    TIER 2 — the strategy behind the post. Secondary to the post itself, but it
    is what a reviewer reads to decide whether the angle is right.
    """
    draft = run.draft
    st.markdown("<div class='tier-gap'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>The strategy behind it</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='think-block'>", unsafe_allow_html=True)

    if draft.tagline:
        st.markdown(f"<div class='think-tagline'>{draft.tagline}</div>", unsafe_allow_html=True)
    if draft.campaign_concept:
        st.markdown(f"<div class='think-concept'>{draft.campaign_concept}</div>",
                    unsafe_allow_html=True)
    if draft.key_messages:
        st.markdown("<div class='section-label'>Key messages</div>", unsafe_allow_html=True)
        for msg in draft.key_messages:
            st.markdown(f"<div class='think-msg'>{msg}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_proof(run):
    """TIER 3 — the evidence, collapsed."""
    receipts = run.routing.receipts if run.routing else []
    traced = sum(1 for r in receipts if r["source_found"])
    n = len(receipts)
    noun = "claim" if n == 1 else "claims"
    label = (f"Proof — {n} {noun} traced to source" if receipts and traced == n
             else f"Proof — {traced} of {n} {noun} traced" if receipts
             else "Proof — no factual claims made")
    st.markdown("<div class='tier-gap'></div>", unsafe_allow_html=True)
    with st.expander(label, expanded=False):
        render_receipts(run)


def render_diagnostics(run, step_times=None, heading="Diagnostics"):
    """TIER 4 — pipeline steps and cost, in the sidebar, out of the way."""
    with st.sidebar:
        st.markdown(f"<div class='section-label' style='margin-top:18px;'>{heading}</div>",
                    unsafe_allow_html=True)
        if step_times:
            for step, secs in step_times:
                _, label = STEP_UI.get(step, ("", step))
                failed = (step in ("verify", "re-verify")
                          and not run.published and run.failure_type)
                dot = "dot-bad" if failed else "dot-ok"
                st.markdown(
                    f"<div class='diag-row'><span class='dot {dot}'></span>"
                    f"<span>{label.split(' — ')[0]}</span>"
                    f"<span class='t'>{secs:.1f}s</span></div>",
                    unsafe_allow_html=True,
                )
            total_s = sum(sec for _, sec in step_times)
            st.markdown(f"<div class='diag-total'>Total {total_s:.1f}s</div>",
                        unsafe_allow_html=True)

        cost_txt = run.cost.display_total()
        st.markdown(
            f"<div class='diag-total' style='margin-top:14px;'>{cost_txt}"
            f"<span style='font-weight:400; color:var(--ink-4);'> · "
            f"{run.cost.total_tokens:,} tokens</span></div>",
            unsafe_allow_html=True,
        )
        for c in run.cost.steps:
            amt = c.display_usd()
            st.markdown(
                f"<div class='diag-row'><span>{c.step}</span>"
                f"<span class='t'>{amt}</span></div>", unsafe_allow_html=True)
        if run.cost.tier_exceeded:
            st.caption("Crossed into the long-context pricing tier — verify the rate.")
        if run.cost.has_unknown:
            st.caption("Some steps report **unknown** cost — usage could not be captured. "
                       "That is not the same as free; the run total is withheld rather "
                       "than under-reported.")

        # The exact text sent to each model on this run — rendered, not the
        # template. Collapsed, because it is long and only wanted when debugging
        # why a particular run came out the way it did.
        if run.prompts:
            st.markdown("<div class='section-label' style='margin-top:20px;'>"
                        "Prompts sent</div>", unsafe_allow_html=True)
            for i, pr in enumerate(run.prompts):
                chars = len(pr["text"])
                with st.expander(f"{pr['step']} · {chars:,} chars", expanded=False):
                    st.caption(f"model: {pr['model']}")
                    st.code(pr["text"], language=None)
            joined = "\n\n".join(
                f"===== {pr['step']} · {pr['model']} =====\n{pr['text']}"
                for pr in run.prompts
            )
            st.download_button(
                "Download all prompts",
                data=joined,
                file_name=f"prompts_cycle{run.cycle}.txt",
                mime="text/plain",
                key=f"dl_prompts_{run.span_id}_{run.cycle}",
            )


def render_halt_sequence(run):
    """
    The important screen. Here is what it was going to say, here is why that
    failed, here is what you are getting instead — in that order.
    """
    if run.published or not run.flagged_claim:
        return
    if run.failure_type == "JUDGE_UNAVAILABLE":
        return  # nothing was checked, so there is no claim to walk through

    st.markdown("<div class='section-label'>What happened</div>", unsafe_allow_html=True)

    layer = {"deterministic": "a deterministic check — no model call needed",
             "judge": "the grounding judge"}.get(run.check_layer, run.check_layer)

    original = run.original_draft.verified_text() if run.original_draft else ""
    st.markdown(
        f"<div class='seq-step'><div class='seq-num'>1</div><div class='seq-body'>"
        f"<div class='seq-title'>What it was going to say</div>"
        f"<div class='seq-text'>{_highlight(original, run.flagged_claim)}</div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    if run.source_line:
        why = (f"<div class='seq-text'>Caught by {layer} · "
               f"<code>{run.failure_type}</code><br><br>"
               f"The brand documents say:<br>"
               f"<em>“{run.source_line}”</em></div>")
    else:
        why = (f"<div class='seq-text'>Caught by {layer} · "
               f"<code>{run.failure_type}</code><br><br>"
               f"No retrieved document addresses this claim at all — there is no source to "
               f"check it against. That is its own problem: retrieval, not just generation, "
               f"came up short.</div>")
    st.markdown(
        f"<div class='seq-step'><div class='seq-num'>2</div><div class='seq-body'>"
        f"<div class='seq-title'>Why that failed</div>{why}</div></div>",
        unsafe_allow_html=True,
    )

    if run.repair_attempted:
        revised = run.draft.verified_text()
        note = ("The repair cleared the re-check but the campaign was still held — see above."
                if run.repair_fixed else
                "The repair was re-checked once and failed again. Two attempts is the cap, so "
                "this is where it stops and a human takes over.")
        body = (f"<div class='seq-text'>{_highlight(revised, run.verdict.claim_flagged)}</div>"
                f"<div style='font-size:0.82rem; color:#6b7280; margin-top:8px;'>{note}</div>")
    else:
        body = ("<div class='seq-text' style='color:#6b7280;'>No repair was attempted — "
                "the campaign was flagged and stopped here.</div>")
    st.markdown(
        f"<div class='seq-step'><div class='seq-num'>3</div><div class='seq-body'>"
        f"<div class='seq-title'>What you are getting instead</div>{body}</div></div>",
        unsafe_allow_html=True,
    )




def render_receipts(run):
    """Each factual claim, green, with the source line that supports it."""
    receipts = run.routing.receipts if run.routing else []
    if not receipts:
        return

    missing = [r for r in receipts if not r["source_found"]]
    st.markdown("<div class='section-label'>Receipts — every factual claim, traced</div>",
                unsafe_allow_html=True)
    if missing:
        st.markdown(
            f"<div style='font-size:0.82rem; color:#78350f; background:#fffbeb; "
            f"border:1px solid #fcd34d; border-radius:8px; padding:8px 12px; margin-bottom:10px;'>"
            f"{len(missing)} of {len(receipts)} claims had no source line located in the "
            f"retrieved documents. They passed verification, but there is nothing to show a "
            f"reviewer.</div>", unsafe_allow_html=True)

    for r in receipts:
        found = r["source_found"]
        cls = "receipt" if found else "receipt receipt-missing"
        dot = "<span class='dot dot-ok'></span>" if found else "<span class='dot dot-warn'></span>"
        src = (f"<div class='receipt-src'>“{r['located_source_line']}”</div>" if found
               else "<div class='receipt-src'>No source line located in the retrieved documents.</div>")
        st.markdown(
            f"<div class='{cls}'><div class='receipt-claim'>{dot}{r['claim']}</div>"
            f"<div style='font-size:0.74rem; color:#6b7280; margin-top:3px;'>{r['kind']}</div>"
            f"{src}</div>", unsafe_allow_html=True)






def render_result(run, step_times=None, heading="Diagnostics"):
    """
    THE result view — four tiers, one function, used by single runs and by every
    campaign in a series. The two paths drifted apart once and produced two
    different products from the same pipeline; there is only one now.

      1. the post        what publishes, laid out as a feed preview
      2. the thinking    the strategy a reviewer judges the angle by
      3. the proof       receipts, collapsed
      4. diagnostics     steps and cost, in the sidebar
    """
    render_post(run)
    render_status(run)

    if not run.published:
        st.markdown("<div class='tier-gap-sm'></div>", unsafe_allow_html=True)
        if run.failure_type != "JUDGE_UNAVAILABLE":
            render_halt_sequence(run)
        render_recommended_action(run)

    render_thinking(run)
    render_proof(run)
    render_diagnostics(run, step_times, heading)
    st.markdown("<div class='tier-gap'></div>", unsafe_allow_html=True)


def render_recommended_action(run):
    st.markdown("<div class='section-label'>Recommended action</div>", unsafe_allow_html=True)
    if run.failure_type == "JUDGE_UNAVAILABLE":
        rows = [
            ("Held, not cleared", "The judge never ran. Unverified is not the same as clean."),
            ("Check the judge", "Confirm phoenix[evals] is installed and the API key is valid."),
            ("Re-run", "Once the judge is reachable this copy may well pass."),
        ]
    else:
        rows = {
            "UNSUPPORTED_CLAIM": [
                ("Do not publish", "A claim is not supported by the retrieved brand documents."),
                ("Check the flagged claim", "Verify against the brand guide or sustainability report."),
                ("Brand reviewer decides", "Is the claim defensible, or does the copy need rewriting?"),
            ],
            "UNAPPROVED_STAT": [
                ("Do not publish", "Cites a statistic outside the approved set."),
                ("Verify or remove the figure", "If the number is real, get it added to the brand guide."),
            ],
            "COMPETITOR_NAMED": [
                ("Do not publish", "The brand guide prohibits naming competitors."),
                ("Rewrite without the name", "Make the point on Verdant's own evidence."),
            ],
            "PROHIBITED_PHRASE": [
                ("Do not publish", "Contains a claim Verdant has not earned."),
                ("Escalate to brand safety", "Flag for manual review within one business hour."),
            ],
        }.get(run.failure_type, [("Human review required", "Route to a brand reviewer.")])
    for action, detail in rows:
        st.markdown(f"**{action}** — {detail}")
    if run.failure_type != "JUDGE_UNAVAILABLE":
        st.markdown(f"<div style='font-size:0.82rem; color:var(--ink-3); margin-top:8px;'>"
                    f"Audit record: span_id <code>{run.span_id}</code> · "
                    f"failure_type <code>{run.failure_type}</code> · "
                    f"status <code>{run.final_status}</code></div>", unsafe_allow_html=True)


def run_with_progress(brief, config, cycle=1, audience_comments=None,
                      previous_copy=None, status_slot=None, generate_assets=True,
                      prior_corrections=None, simulate_audience=False):
    """
    Drive the pipeline, rendering one status block per step as it runs.

    The callback is observational only — it reports what the pipeline did. It
    cannot influence which step runs next; that is fixed in pipeline/run.py.
    """
    from pipeline.run import SOURCE_APP, run_campaign_pipeline

    blocks, started, step_times, clock = {}, [], [], {}

    def progress(step, state, detail):
        icon, label = STEP_UI.get(step, ("•", step))
        if state == "start":
            clock[step] = time.time()
            blocks[step] = st.status(f"{label}", expanded=False)
            started.append(step)
            if status_slot is not None:
                status_slot.markdown(f"**{label.split(' — ')[0]}**")
            return

        block = blocks.get(step)
        elapsed = time.time() - clock.get(step, time.time())
        step_times.append((step, elapsed))
        if block is None:
            return
        if step == "retrieve":
            block.update(label=f"Retrieved {detail['n_chunks']} chunks · grounding "
                               f"{detail['grounding_score']:.2f} · {elapsed:.1f}s", state="complete")
        elif step == "generate":
            block.update(label=f"Generated — {detail['n_claims']} declared claims · {elapsed:.1f}s",
                         state="complete")
        elif step in ("verify", "re-verify"):
            if detail["passed"]:
                judged = "judge agreed" if detail.get("judge_called") else "deterministic only"
                block.update(label=f"{step.capitalize()} passed — {judged} · {elapsed:.1f}s",
                             state="complete")
            else:
                block.update(label=f"{step.capitalize()} failed — {detail['failure_type']} "
                                   f"· {elapsed:.1f}s", state="error")
        elif step == "repair":
            block.update(label=f"Repair written — re-checking · {elapsed:.1f}s", state="complete")
        elif step == "route":
            block.update(label=f"Routed → {detail['destination']}"
                               + (" (random audit sample)" if detail["sampled"] else "")
                               + f" · {elapsed:.1f}s", state="complete")
        elif step == "audience":
            block.update(label=f"Audience — {detail['n']} comments · {elapsed:.1f}s",
                         state="complete")
        elif step == "assets":
            block.update(label=f"Video ready · {elapsed:.1f}s" if detail["has_video"]
                               else f"Video unavailable — {(detail['error'] or 'no output')[:70]}",
                         state="complete" if detail["has_video"] else "error")

    result = run_campaign_pipeline(
        brief=brief,
        audience_comments=audience_comments,
        previous_copy=previous_copy,
        prior_corrections=prior_corrections,
        cycle=cycle,
        include_poisoned=config["include_poisoned"],
        simulate_low_confidence=config["simulate_low_confidence"],
        generate_assets=generate_assets,
        simulate_audience=simulate_audience,
        progress=progress,
        source=SOURCE_APP,
    )
    return result, step_times




# ---------------------------------------------------------------------------
# Pipeline output
# ---------------------------------------------------------------------------
if run_btn and user_prompt.strip():
    st.divider()
    series_start = date.today()
    delta = SCHEDULE_DELTAS.get(schedule, timedelta(weeks=1))
    drop_dates = [series_start + i * delta for i in range(num_campaigns)]

    # ── SERIES MODE ───────────────────────────────────────────────────────
    if num_campaigns > 1:
        pipeline_status.markdown(f"**Series** — {num_campaigns} campaigns")
        st.markdown(f"<h2 style='font-size:1.2rem; font-weight:700; text-align:center;'>Campaign Scheduler — {num_campaigns} {schedule.lower()} drops</h2>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.85rem; color:var(--ink-3); text-align:center; margin-bottom:1rem;'>Each campaign briefs the next with its approved copy and what the audience said about it. Brand facts are re-retrieved from source every cycle, and every cycle passes the grounding check before anything is generated.</p>", unsafe_allow_html=True)

        completed = []
        campaign_history = []
        # Lessons earned so far in this series. Each cycle inherits every earlier
        # correction — the claim and the source line that contradicted it — so the
        # same mistake is not relearned from scratch every week.
        series_corrections = []

        series_progress = st.progress(0)
        series_status = st.empty()

        from opentelemetry import trace as otel_trace
        _tracer = otel_trace.get_tracer(__name__)

        for i in range(num_campaigns):
            series_status.info(f"Generating campaign {i+1} of {num_campaigns}…")
            st.markdown(f"<div style='font-size:0.85rem; font-weight:700; color:var(--ink-3); margin:1rem 0 4px 0; text-transform:uppercase; letter-spacing:0.05em;'>Campaign {i+1} · {drop_dates[i].strftime('%b %d, %Y')}</div>", unsafe_allow_html=True)

            t0 = time.time()

            run, _step_times = run_with_progress(
                brief=user_prompt,
                config=config,
                cycle=i + 1,
                audience_comments=(campaign_history[-1]["comments"] if campaign_history else None),
                previous_copy=(campaign_history[-1] if campaign_history else None),
                prior_corrections=series_corrections,
                status_slot=None,
                simulate_audience=(i < num_campaigns - 1),
            )
            t3 = time.time()

            approved = run.published

            correction = run.correction_record()
            if correction:
                series_corrections.append(correction)

            # Comments belong to published campaigns. A flagged cycle was never
            # published, so it collects none — and contributes nothing to the
            # next brief. The rejected draft is never passed forward.
            # Generated from this campaign's own published copy, so the reaction
            # is about what was actually said. Nothing is generated for a flagged
            # cycle — it never published, so nobody saw it.
            comments = run.audience_comments_generated

            completed.append({
                "n": i + 1,
                "drop_date": drop_dates[i],
                "run": run,
                "elapsed": t3 - t0,
                "comments": comments,
                "corrections_needed": 1 if correction else 0,
                "inherited": len(series_corrections) - (1 if correction else 0),
            })

            render_result(run, _step_times, heading=f"Campaign {i+1} diagnostics")

            if comments and i < num_campaigns - 1:
                comment_html = "".join(
                    f"<div style='font-size:0.85rem; color:#374151; line-height:1.6;'>“{c}”</div>"
                    for c in comments
                )
                st.markdown(f"""
                <div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:14px 16px; margin:10px 0;'>
                    <div style='font-size:0.72rem; font-weight:700; letter-spacing:0.07em; text-transform:uppercase; color:var(--ink-3); margin-bottom:8px;'>
                        💬 Audience comments — {drop_dates[i].strftime('%b %d')}
                    </div>
                    {comment_html}
                    <div style='font-size:0.78rem; color:var(--ink-3); margin-top:8px;'>
                        Carried into the next brief alongside this campaign's approved copy. Brand facts are re-retrieved from source.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if approved:
                campaign_history.append({
                    "tagline": run.draft.tagline,
                    "campaign_concept": run.draft.campaign_concept,
                    "key_messages": run.draft.key_messages,
                    "comments": comments,
                })
            else:
                st.markdown(
                    "<div style='font-size:0.8rem; color:#7c3aed; margin:4px 0 10px 0;'>"
                    "This cycle produced nothing publishable, so nothing from it is carried into the next brief."
                    "</div>",
                    unsafe_allow_html=True,
                )

            series_progress.progress((i + 1) / num_campaigns)

        series_status.empty()

        st.markdown("<div class='tier-gap'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Did it learn?</div>", unsafe_allow_html=True)

        needed = [c["corrections_needed"] for c in completed]
        total_corrections = sum(needed)
        first_half = needed[: max(1, len(needed) // 2)]
        second_half = needed[max(1, len(needed) // 2):]

        if len(completed) < 2:
            verdict = "A single campaign cannot show a trend."
        elif total_corrections == 0:
            verdict = ("No campaign in this series needed a correction. Nothing to learn "
                       "from, which is the good outcome.")
        elif second_half and sum(second_half) < sum(first_half):
            verdict = (f"Corrections fell from {sum(first_half)} in the first half of the "
                       f"series to {sum(second_half)} in the second. Each cycle inherits "
                       f"every earlier correction with the source line behind it, so this "
                       f"is the system carrying its mistakes forward rather than "
                       f"relearning them.")
        elif second_half and sum(second_half) > sum(first_half):
            verdict = (f"Corrections rose from {sum(first_half)} to {sum(second_half)}. "
                       f"The carried-forward corrections are not holding — worth reading "
                       f"the flagged claims below to see whether they are the same kind "
                       f"of mistake or new ones.")
        else:
            verdict = (f"{total_corrections} correction(s) across {len(completed)} "
                       f"campaigns, flat across the series.")

        cols = st.columns(len(completed))
        for col, c in zip(cols, completed):
            r = c["run"]
            n_fix = c["corrections_needed"]
            tone = "c-live" if n_fix == 0 else "c-review"
            word = "clean" if n_fix == 0 else f"{n_fix} correction"
            col.markdown(
                f"<div style='text-align:center;'>"
                f"<div class='section-label' style='margin-bottom:8px;'>Cycle {c['n']}</div>"
                f"<div class='chip {tone}' style='font-size:0.85rem; padding:9px 16px;'>{word}</div>"
                f"<div style='font-size:0.76rem; color:var(--ink-3); margin-top:10px;'>"
                f"inherited {c['inherited']}</div></div>",
                unsafe_allow_html=True,
            )
        st.markdown(f"<div class='chip-sub' style='margin-top:18px;'>{verdict}</div>",
                    unsafe_allow_html=True)

        if series_corrections:
            with st.expander(f"Corrections carried forward — {len(series_corrections)}",
                             expanded=False):
                for c in series_corrections:
                    st.markdown(
                        f"<div class='receipt'>"
                        f"<div class='receipt-claim'>Cycle {c['cycle']}: {c['claim']}</div>"
                        + (f"<div class='receipt-src'>{c['source_line']}</div>"
                           if c.get("source_line") else "")
                        + (f"<div style='font-size:0.85rem; color:var(--ink-3); margin-top:8px;'>"
                           f"Corrected to: {c['correction']}</div>"
                           if c.get("correction") else
                           "<div style='font-size:0.85rem; color:var(--ink-3); margin-top:8px;'>"
                           "Not resolved — the repair did not clear the re-check.</div>")
                        + "</div>", unsafe_allow_html=True)

        priced = [c["run"].cost.total_usd for c in completed if c["run"].cost.total_usd is not None]
        unknown_cycles = [c["n"] for c in completed if c["run"].cost.total_usd is None]
        if priced:
            note = (f" · cycle(s) {', '.join(map(str, unknown_cycles))} unknown"
                    if unknown_cycles else "")
            st.markdown(f"<div class='chip-sub'>Series total ${sum(priced):.5f} across "
                        f"{len(priced)} priced campaigns · mean "
                        f"${sum(priced)/len(priced):.5f}{note}</div>", unsafe_allow_html=True)
        elif completed:
            st.markdown("<div class='chip-sub'>Series cost unknown — usage could not be "
                        "captured for these runs.</div>", unsafe_allow_html=True)

        with pipeline_status.container():
            for c in completed:
                r = c["run"]
                cls, head, _ = status_state(r)
                dot = {"s-live": "dot-ok", "s-sample": "dot-ok",
                       "s-review": "dot-bad", "s-unknown": "dot-warn"}[cls]
                st.markdown(
                    f"<div style='font-size:0.85rem; margin-bottom:4px;'>"
                    f"<span class='dot {dot}'></span>Campaign {c['n']} — {head.split(' — ')[0]}"
                    f"</div>", unsafe_allow_html=True)
            st.progress(1.0)

    # ── SINGLE CAMPAIGN MODE ──────────────────────────────────────────────
    else:
        t0 = time.time()
        run, step_times = run_with_progress(
            brief=user_prompt,
            config=config,
            cycle=1,
            status_slot=pipeline_status,
        )
        elapsed = time.time() - t0

        if config["include_poisoned"]:
            st.error("🔴 Injection document loaded into retrieval")

        with pipeline_status.container():
            for step, secs in step_times:
                icon, label = STEP_UI.get(step, ("•", step))
                st.markdown(f"{icon} **{label.split(' — ')[0]}** · {secs:.1f}s")
            st.progress(1.0)
            st.caption(f"Total: {elapsed:.1f}s")

    if num_campaigns == 1:
        st.divider()

        render_result(run, step_times)

        with st.expander("Trust signals", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Retrieve**")
                st.code(f"grounding_score: {run.retrieval.grounding_score:.2f}\n"
                        f"n_chunks: {len(run.retrieval.chunks)}\n"
                        f"injection_risk: {run.retrieval.metadata['injection_risk'].upper()}")
            with c2:
                st.markdown("**Verify**")
                st.code(f"layer: {run.check_layer}\n"
                        f"failure_type: {run.failure_type or 'none'}\n"
                        f"judge_called: {any(e['judge_called'] for e in run.verify_events)}\n"
                        f"attempts: {len(run.verify_events)}")
            with c3:
                st.markdown("**Route**")
                st.code(f"status: {run.final_status}\n"
                        f"destination: {run.routing.destination}\n"
                        f"sampled: {run.routing.sampled}\n"
                        f"published: {run.published}")
            st.caption("The grounding score is reported, not gating. It is built from retrieval "
                       "signals and never sees the generated text — Verify reads the text instead.")

elif run_btn and not user_prompt.strip():
    st.warning("Please describe your campaign before generating.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption("Campaign Studio · Built by Rebecca Riggs · 2026")
