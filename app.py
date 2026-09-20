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
    initial_sidebar_state="expanded",
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
/* ── Inputs and cards: explicit foreground AND background everywhere ──────
   Literal hex, not custom properties, and -webkit-text-fill-color alongside
   colour — that is the one that actually wins on input elements. Without this
   a dark-mode machine rendered Streamlit's near-white input text onto the white
   background this stylesheet forces, i.e. invisible typing. The app must not
   depend on the viewer's OS theme.                                          */
div[data-baseweb="input"], div[data-baseweb="textarea"],
div[data-baseweb="base-input"], .stTextInput > div, .stTextArea > div {
    border:none !important; border-radius:18px !important;
    background-color:#FFFFFF !important;
    box-shadow:0 2px 0 rgba(11,15,13,.06) !important;
}
.stTextArea textarea, .stTextInput input, textarea, input[type="text"] {
    background-color:#FFFFFF !important;
    color:#0B0F0D !important;
    -webkit-text-fill-color:#0B0F0D !important;
    caret-color:#0E3B2E !important;
    font-family:'Outfit', sans-serif !important;
    border:none !important;
}
.stTextArea textarea::placeholder, .stTextInput input::placeholder {
    color:#9AA39E !important; -webkit-text-fill-color:#9AA39E !important;
}

/* expanders — header and body both pinned */
div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] summary * { color:#0B0F0D !important; }
div[data-testid="stExpander"] > div,
div[data-testid="stExpanderDetails"] {
    background-color:#FFFFFF !important; color:#3A4440 !important;
}

/* code blocks and captions */
/* Streamlit highlights code with Prism, whose token spans carry their own
   colours chosen for a dark background. Setting colour on the container alone
   left those spans light-on-light — the prompts reported thousands of chars and
   rendered as empty blocks. Descendants have to be forced too. */
.stCode, pre, code, div[data-testid="stCode"] {
    background-color:#F1EDE6 !important; color:#0B0F0D !important;
}
div[data-testid="stCode"] *, .stCode *, pre *, code *, pre span, code span {
    color:#0B0F0D !important; -webkit-text-fill-color:#0B0F0D !important;
    background:transparent !important;
}
div[data-testid="stCode"] pre, .stCode pre {
    white-space:pre-wrap !important; word-break:break-word !important;
}
div[data-testid="stCaptionContainer"], div[data-testid="stCaptionContainer"] * {
    color:#6B7671 !important;
}
label, .stTextArea label, .stTextInput label, label * { color:#3A4440 !important; }

/* dataframes keep their own chrome — pin the surface they sit on */
div[data-testid="stDataFrame"] { background-color:#FFFFFF !important; }
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
    color:#0B0F0D;
}
/* The card split around the video widget: same fill, rounding only on the outer
   corners, so the seam behind the player is invisible. A single wrapper div is
   not possible — st.video is a widget, not HTML. */
.post-top {
    background:var(--card); color:#0B0F0D; border-radius:28px 28px 0 0;
    padding:0 0 2px 0; margin-bottom:0;
}
.post-bottom {
    background:var(--card); color:#0B0F0D; border-radius:0 0 28px 28px;
    padding:2px 30px 26px 30px; margin:0 0 18px 0;
    box-shadow:0 3px 0 rgba(11,15,13,.06), 0 22px 54px rgba(11,15,13,.10);
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
.post-tagline {
    font-family:'Bricolage Grotesque', sans-serif; font-size:1.5rem; font-weight:800;
    color:var(--ink); line-height:1.15; letter-spacing:-0.02em; margin:22px 0 2px 0;
}
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
    box-shadow:0 2px 0 rgba(11,15,13,.05);     color:#0B0F0D;
}
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
# Step labels for progress and diagnostics. No emoji — Streamlit's own status
# icons carry the state.
STEP_UI = {
    "retrieve":  ("", "Retrieve — querying brand documents (no model call)"),
    "generate":  ("", "Generate — one model call, structured output"),
    "verify":    ("", "Verify — deterministic checks, then judge"),
    "repair":    ("", "Repair — rewriting against the source line"),
    "re-verify": ("", "Re-verify — checking the repair"),
    "route":     ("", "Route — publish, flag, or sample"),
    "assets":    ("", "Assets — generating video"),
    "audience":  ("", "Audience — simulating reaction to what published"),
}


def _failure_noun(run):
    return {
        "UNSUPPORTED_CLAIM": "unsupported claim",
        "UNAPPROVED_STAT":   "unapproved statistic",
        "COMPETITOR_NAMED":  "competitor reference",
        "PROHIBITED_PHRASE": "prohibited claim",
    }.get(run.failure_type, "flagged claim")


def status_state(run):
    """
    The review decision, in four unambiguous states.

    Published vs needs-review is the call a reviewer scanning a queue has to make
    in under a second, so the two published states share cool colours and the two
    unpublished ones share warm — the grouping reads before the words do.

    Returns (css_class, headline, detail).
    """
    receipts = run.routing.receipts if run.routing else []
    traced = sum(1 for r in receipts if r["source_found"])

    # Checked first: failure_type can now carry a video failure, and the copy-judge
    # branch below would otherwise claim the copy was never verified.
    blocked = getattr(run, "video_blocked", None)
    if blocked is not None:
        return ("s-review", "Held for review — video blocked",
                blocked.reason or "The shot was refused by the video guardrail.")

    if run.failure_type == "JUDGE_UNAVAILABLE":
        return ("s-unknown", "Not published — check unavailable",
                "The judge errored, so this copy was never verified. Held fail-closed: "
                "unverified is not the same as clean.")
    if not run.published:
        repaired = " couldn't be repaired" if run.repair_attempted else " was flagged"
        return ("s-review", "Not published — needs review",
                f"1 {_failure_noun(run)}{repaired}.")
    if run.routing.sampled:
        # "Flagged" is reserved for campaigns where a check found something.
        # Nothing was wrong with this one — it was drawn at random.
        return ("s-sample", "Published · in 10% audit sample",
                "Randomly selected for quality review. Already live.")

    n = len(receipts)
    noun = "claim" if n == 1 else "claims"
    if receipts and traced == n:
        detail = f"{n} {noun}, {'traced' if n == 1 else 'all traced'}."
    elif receipts:
        detail = f"{n} {noun}, {n - traced} without a source line."
    else:
        detail = "No factual claims made."
    return ("s-live", "Published — no review needed", detail)


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
    TIER 1 — the post, as a feed preview.

    Streamlit wraps every st.markdown() in its own container, so an opening
    <div> emitted alone is sealed empty and the content lands outside it. Each
    emission below is therefore a COMPLETE, self-closing fragment.

    st.video is a real widget and cannot live inside an HTML string, so the card
    is drawn as two halves — .post-top above the video, .post-bottom below —
    sharing a fill, with rounding only on their outer corners.
    """
    draft, assets = run.draft, run.assets
    has_video = bool(assets and assets.video_bytes)

    head = (
        "<div class='post-head'><div class='post-avatar'>V</div>"
        "<div><div class='post-handle'>verdant</div>"
        "<div class='post-meta'>Sponsored · Draft preview</div></div></div>"
    )
    body = ""
    if draft.tagline:
        body += f"<div class='post-tagline'>{draft.tagline}</div>"
    if draft.caption:
        body += f"<div class='post-caption'>{draft.caption}</div>"
    if draft.hashtags:
        body += f"<div class='post-tags'>{' '.join(draft.hashtags)}</div>"

    if has_video:
        st.markdown(f"<div class='post-top'>{head}</div>", unsafe_allow_html=True)
        st.video(assets.video_bytes)
        st.markdown(f"<div class='post-bottom'>{body}</div>", unsafe_allow_html=True)
        return

    if run.video_blocked is not None:
        note = ("Video blocked by the guardrail — the shot was refused before "
                "filming. See “The shot we asked for” below.")
    elif not run.published:
        note = "No video — the campaign was held, so the asset step never ran."
    elif assets and assets.error:
        note = f"Video unavailable — {assets.error[:140]}"
    else:
        note = "No video available."
    st.markdown(
        f"<div class='post'>{head}<div class='post-empty'>{note}</div>"
        f"<div class='post-inner'>{body}</div></div>",
        unsafe_allow_html=True,
    )


def render_shot(run):
    """
    "The shot we asked for" — the description sent to Veo and what the video
    guardrail made of it.

    Always collapsed, always rendered, whether the video generated or was
    refused. When it was refused this is the only place a reviewer can see what
    was actually asked for and why it was stopped.
    """
    assets = run.assets
    prompt = (assets.video_prompt if assets else "") or ""
    blocked = run.video_blocked
    leaked = getattr(run, "video_concept_leaked", []) or []
    if not prompt and not blocked:
        return

    with st.expander("The shot we asked for", expanded=False):
        if blocked is not None:
            layer = {"deterministic": "deterministic check — no model call",
                     "judge": "visual-claim judge"}.get(blocked.layer, blocked.layer or "—")
            st.markdown(
                f"<div style='background:#FDEEE9; color:#7F1D1D; border-radius:14px; "
                f"padding:14px 18px; margin-bottom:14px; font-size:0.9rem;'>"
                f"<strong>Blocked — not sent to Veo.</strong><br>"
                f"Caught by the {layer} · <code>{blocked.failure_type}</code><br><br>"
                f"{blocked.reason}</div>", unsafe_allow_html=True)
            if blocked.claim_flagged:
                # Only the judge has a view about viewers. The deterministic layer
                # matches strings and knows nothing about what anyone would think.
                heading = ("What a viewer would have concluded"
                           if blocked.layer == "judge" else "What the check found")
                st.markdown(f"<div class='section-label'>{heading}</div>",
                            unsafe_allow_html=True)
                st.markdown(f"<div class='seq-text'>{blocked.claim_flagged}</div>",
                            unsafe_allow_html=True)
        else:
            verdict = getattr(run, "video_judge_label", "") or "not called"
            st.markdown(
                f"<div style='background:#F1F7F2; color:#14532D; border-radius:14px; "
                f"padding:12px 18px; margin-bottom:14px; font-size:0.9rem;'>"
                f"<strong>Passed both guardrail layers.</strong><br>"
                f"Deterministic checks clear · visual-claim judge: "
                f"<strong>{verdict}</strong></div>", unsafe_allow_html=True)

        if leaked:
            st.markdown("<div class='section-label' style='margin-top:14px;'>"
                        "Concept terms that leaked</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='seq-text'>The shot still used "
                f"<strong>{', '.join(leaked)}</strong> after a retry, so the generic "
                f"fallback shot was filmed instead. A video model renders those "
                f"words as literal objects.</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-label' style='margin-top:14px;'>"
                    "Shot description</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='seq-text'>{prompt}</div>", unsafe_allow_html=True)


def render_proof(run):
    """TIER 3 — the evidence, collapsed."""
    receipts = run.routing.receipts if run.routing else []
    traced = sum(1 for r in receipts if r["source_found"])
    n = len(receipts)
    noun = "claim" if n == 1 else "claims"
    if receipts:
        label = (f"Proof — {n} {noun} traced to source" if traced == n
                 else f"Proof — {traced} of {n} {noun} traced")
    elif not run.published:
        # Receipts are only built for copy that publishes. Saying "no factual
        # claims made" here would be false — the claims exist, they just were
        # never cleared.
        label = "Proof — not generated; this campaign was not published"
    else:
        label = "Proof — no factual claims made"

    # The rail reports the claim count at GENERATE; receipts are built after
    # REPAIR may have dropped or rewritten some. Two honest numbers that look
    # like a contradiction unless the relationship is spelled out.
    declared = len(run.original_draft.claims) if run.original_draft else None
    final = len(run.draft.claims)
    if run.repair_attempted and declared is not None and declared != final:
        label += f" · {declared} declared → {final} after repair"
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
        with st.expander("Cost by step", expanded=False):
            for c in run.cost.steps:
                st.markdown(
                    f"<div class='diag-row'><span>{c.step}</span>"
                    f"<span class='t'>{c.display_usd()}</span></div>",
                    unsafe_allow_html=True)
        if run.cost.tier_exceeded:
            st.caption("Crossed into the long-context pricing tier — verify the rate.")

        with st.expander("Trust signals", expanded=False):
            st.code(f"grounding_score: {run.retrieval.grounding_score:.2f}\n"
                    f"n_chunks: {len(run.retrieval.chunks)}\n"
                    f"injection_risk: {run.retrieval.metadata['injection_risk'].upper()}\n"
                    f"layer: {run.check_layer}\n"
                    f"failure_type: {run.failure_type or 'none'}\n"
                    f"judge_called: {any(e['judge_called'] for e in run.verify_events)}\n"
                    f"attempts: {len(run.verify_events)}\n"
                    f"status: {run.final_status}\n"
                    f"destination: {run.routing.destination}\n"
                    f"sampled: {run.routing.sampled}\n"
                    f"published: {run.published}")
            st.caption("The grounding score is reported, not gating. It is built from "
                       "retrieval signals and never sees the generated text.")
        if run.cost.has_unknown:
            st.caption("Some steps report **unknown** cost — usage could not be captured. "
                       "That is not the same as free; the run total is withheld rather "
                       "than under-reported.")

        # The exact text sent to each model on this run — rendered, not the
        # template. Collapsed, because it is long and only wanted when debugging
        # why a particular run came out the way it did.
        if run.prompts:
            st.markdown("---")
            st.markdown(f"#### Prompts sent to the model, every step")
            st.caption(f"Cycle {run.cycle} · {len(run.prompts)} calls · the exact "
                       f"rendered text, not the template")
            for i, pr in enumerate(run.prompts):
                chars = len(pr["text"])
                with st.expander(f"{i+1}. {pr['step']} · {chars:,} chars", expanded=False):
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


def _highlight(text, claim):
    """
    Mark the offending fragment inside the draft, so a reviewer sees the claim in
    place rather than quoted separately.
    """
    import html as _html
    safe = _html.escape(text or "")
    frag = (claim or "").strip()
    # UNAPPROVED_STAT flags arrive as '100% — "…context…"'; the number is the part
    # that actually appears in the copy.
    if "—" in frag:
        frag = frag.split("—")[0].strip()
    frag = frag.strip('"\u201c\u201d')
    if frag and len(frag) < 80 and _html.escape(frag) in safe:
        safe = safe.replace(_html.escape(frag),
                            f"<span class='claim-hl'>{_html.escape(frag)}</span>", 1)
    return safe


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
      2. the proof       receipts, collapsed
      3. diagnostics     steps and cost, in the sidebar

    campaign_concept and key_messages are NOT displayed. Both still feed
    verified_text() into the guardrail and carry into the next cycle's brief —
    checked, not shown. A reviewer judges the angle from the caption, which is
    what actually publishes.
    """
    render_post(run)
    render_shot(run)
    render_status(run)

    if not run.published:
        st.markdown("<div class='tier-gap-sm'></div>", unsafe_allow_html=True)
        if run.failure_type != "JUDGE_UNAVAILABLE":
            render_halt_sequence(run)
        render_recommended_action(run)

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

    # Live progress belongs in the main body. It used to go only to the sidebar
    # panel, which ships collapsed — so a 60-second run looked like nothing was
    # happening at all.
    expected = 6 if generate_assets else 4          # retrieve, generate, verify, route (+assets, +audience)
    # The rail lives in a placeholder so it can be replaced by a one-line summary
    # once the run finishes. Live it is the only sign anything is happening; after
    # the fact it is seven lines of history above the thing you came to look at.
    rail = st.empty()
    rail_box = rail.container()
    with rail_box:
        bar = st.progress(0.0, text="Starting…")
    done_count = {"n": 0}

    def progress(step, state, detail):
        icon, label = STEP_UI.get(step, ("•", step))
        name = label.split(" — ")[0]
        if state == "start":
            clock[step] = time.time()
            bar.progress(min(done_count["n"] / expected, 0.95),
                         text=f"Step {len(started) + 1} · {name}…")
            with rail_box:
                blocks[step] = st.status(f"{label}", expanded=False)
            started.append(step)
            if status_slot is not None:
                status_slot.markdown(f"**{name}**")
            return

        block = blocks.get(step)
        elapsed = time.time() - clock.get(step, time.time())
        step_times.append((step, elapsed))
        done_count["n"] += 1
        bar.progress(min(done_count["n"] / expected, 0.95),
                     text=f"{name} done · {elapsed:.0f}s")
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
                               + (" · in 10% audit sample" if detail["sampled"] else "")
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
    total = sum(sec for _, sec in step_times)
    bar.progress(1.0, text=f"Done · {len(step_times)} steps · {total:.0f}s")

    # Replace the live rail with a single collapsed row.
    rail.empty()
    with rail.container():
        with st.expander(f"Done · {len(step_times)} steps · {total:.0f}s", expanded=False):
            for step, secs in step_times:
                _, lbl = STEP_UI.get(step, ("", step))
                st.markdown(f"<div class='diag-row'><span class='dot dot-ok'></span>"
                            f"<span>{lbl.split(' — ')[0]}</span>"
                            f"<span class='t'>{secs:.1f}s</span></div>",
                            unsafe_allow_html=True)
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
                with st.expander(
                    f"Audience comments — {drop_dates[i].strftime('%b %d')} "
                    f"({len(comments)})", expanded=False):
                    for c in comments:
                        st.markdown(
                            f"<div style='font-size:0.9rem; color:var(--ink-2); "
                            f"line-height:1.65; padding:4px 0;'>“{c}”</div>",
                            unsafe_allow_html=True)
                    st.caption("Simulated. Carried into the next brief alongside this "
                               "campaign's approved copy. Brand facts are re-retrieved "
                               "from source every cycle.")

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

elif run_btn and not user_prompt.strip():
    st.warning("Please describe your campaign before generating.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption("Campaign Studio · Built by Rebecca Riggs · 2026")
