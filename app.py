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
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

h1, h2, h3 { font-family: 'DM Sans', sans-serif; font-weight: 700; }

.studio-title {
    font-family: 'DM Serif Display', serif;
    font-size: 5.5rem;
    font-weight: 400;
    color: #f97316;
    text-align: center;
    margin: 1.5rem 0 0 0;
    line-height: 1;
}

section.main > div { max-width: 740px; margin: 0 auto; padding: 0 1.5rem; }
/* tighten Streamlit's default widget spacing */
div[data-testid="stVerticalBlock"] > div { gap: 0.5rem; }
div[data-testid="stPills"] { margin-bottom: 0.25rem; }

.main-header {
    padding: 2rem 0 1rem 0;
}
.tagline-output {
    font-size: 2rem;
    font-weight: 700;
    color: #1a1a1a;
    line-height: 1.2;
    margin: 1rem 0 0.5rem 0;
}
.concept-output {
    font-size: 1rem;
    color: #4b5563;
    line-height: 1.6;
    margin-bottom: 1.5rem;
}
.example-chip {
    display: inline-block;
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 0.82rem;
    color: #475569;
    cursor: pointer;
    margin: 4px 4px 4px 0;
    transition: all 0.15s;
}
.example-chip:hover { background: #e2e8f0; color: #1e293b; }
.result-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
}
.trust-high   { color: #16a34a; font-weight: 700; }
.trust-medium { color: #d97706; font-weight: 700; }
.trust-low    { color: #dc2626; font-weight: 700; }
.halted-card {
    background: #fdf4ff;
    border: 1.5px solid #c084fc;
    border-radius: 12px;
    padding: 24px;
}
.step-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 4px;
}
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 700;
}
.badge-green  { background: #dcfce7; color: #166534; }
.badge-yellow { background: #fef9c3; color: #713f12; }
.badge-red    { background: #fee2e2; color: #991b1b; }
.badge-purple { background: #f3e8ff; color: #6b21a8; }

div[data-testid="stSidebar"] { background: #f8fafc; }
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

    FAILURE_CONFIGS = {
        "✅ Normal run": {
            "include_poisoned": False, "simulate_low_confidence": False, "trust_aware": True,
        },
        "⚠️ Trust gap (silent failure)": {
            "include_poisoned": False, "simulate_low_confidence": True, "trust_aware": False,
        },
        "🔴 Prompt injection": {
            "include_poisoned": True, "simulate_low_confidence": False, "trust_aware": False,
        },
        "🛡️ Trust-aware mode (the fix)": {
            "include_poisoned": True, "simulate_low_confidence": False, "trust_aware": True,
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
            "⚠️ Trust gap (silent failure)": "Agent 2 never sees Agent 1's low confidence. Claims drift without anyone knowing.",
            "🔴 Prompt injection": "A hidden instruction in a brand doc hijacks the output.",
            "🛡️ Trust-aware mode (the fix)": "Trust signals propagate end-to-end. Injection triggers a halt.",
        }[failure_mode])

    st.markdown("---")
    st.markdown("[Open Arize AX ↗](https://app.arize.com)")
    st.caption("Projects → brand-trust-agent → Traces")

    st.markdown("---")
    with st.expander("📚 Brand document index", expanded=False):
        st.caption("The source documents Agent 1 searches. These are the ground truth for every brand claim.")
        docs = ["verdant_brand_guide.txt", "verdant_products.txt", "verdant_sustainability.txt"]
        if config["include_poisoned"]:
            docs.append("verdant_poisoned.txt ⚠️")
        for d in docs:
            st.markdown(f"{'🔴' if 'poisoned' in d else '🟢'} `{d}`")

        try:
            from agents.brand_research_agent import _get_collection
            coll = _get_collection(include_poisoned=config["include_poisoned"])
            st.caption(f"{coll.count()} chunks indexed")
        except:
            pass

    with st.expander("❓ How it works", expanded=False):
        st.markdown("""
**Three agents, one trust gate:**

1. **Research (RAG · GPT-4o-mini + text-embedding-3-small)** — Retrieves Verdant brand document chunks via ChromaDB, generates a grounded research summary, and scores retrieval quality (0–1). That score measures whether retrieval *found* relevant documents — it is reported, not used as the gate.

2. **Strategy (GPT-4o-mini + Policy Check)** — Builds the campaign concept from Agent 1's research. Checks every factual claim against brand policy — prohibited claims like "carbon neutral" are blocked outright.

3. **Creative (GPT-4o-mini + Veo 2)** — Before generating anything, the campaign text is checked against the documents Agent 1 actually retrieved: deterministic checks first (prohibited phrases, unapproved statistics, competitor names), then an LLM judge on whatever survives. Only text that passes both gets a caption and video.

The trust gate is a hard stop, not a review queue. Content is blocked before it's created.
        """)

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.markdown("<p style='font-family:DM Serif Display, serif; font-size:1.8rem; font-weight:400; color:#ff4b4b; text-align:center; margin:1.5rem 0 0 0; line-height:1.1;'>Campaign Studio</p>", unsafe_allow_html=True)
st.markdown("<p style='font-size:0.95rem; color:#94a3b8; text-align:center; margin-top:4px; margin-bottom:2rem;'>Brand-safe social media campaign generation</p>", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# Campaign input
# ---------------------------------------------------------------------------
EXAMPLES = [
    "Spring launch · Gen Z runners · recycled materials + trail performance",
    "Summer re-engagement · lapsed customers · sustainability values",
    "New trail line · outdoor enthusiasts · environmental impact",
    "Holiday gifting · performance-conscious parents · sustainable options",
]

selected_example = st.pills("Try an example", EXAMPLES)
if selected_example:
    st.session_state["user_prompt"] = selected_example

user_prompt = st.text_area(
    "What would you like to campaign about?",
    value=st.session_state.get("user_prompt", ""),
    height=100,
    placeholder="Describe your goal, audience, and angle — e.g. Spring launch targeting Gen Z runners, lead with our recycled materials story",
)

# ── Series configuration ──────────────────────────────────────────────────
col_series_l, col_series_r = st.columns(2)
with col_series_l:
    num_campaigns = st.pills("Number of campaigns", options=[1, 2, 3, 4, 5], default=1)
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
# Audience comments on each published campaign. Comments only — no CTR, no
# engagement rate, no view counts, no "insight" telling the next cycle what to
# lean into. There is nothing here to optimize toward; the next brief carries
# what people said, and the grounding check still decides what can be claimed.
#
# Indexed by how many campaigns have actually been approved and published, not
# by cycle number: a halted cycle was never published, so nobody commented on it.
AUDIENCE_COMMENTS = [
    [
        "Love the recycled materials angle — feels so authentic",
        "87% certified sustainable is impressive, keep leading on this",
        "This is the brand story I've been waiting for",
    ],
    [
        "Are you guys fully carbon neutral yet? Feels like you're so close",
        "Most eco-committed brand I follow tbh",
        "Love this green mission, keep pushing it further",
    ],
    [
        "I thought you were already carbon neutral honestly",
        "Zero waste vibes, love it",
        "The most sustainable activewear brand, period",
    ],
    [
        "Carbon neutral queen era 👑",
        "Fully sustainable from production to delivery right?",
        "B Corp certified soon?? 👀 you deserve it",
    ],
]


def render_grounding_record(creative):
    """
    Show what the grounding check flagged, the source line it ran into, and how
    the rewrite corrected it — side by side, so a reviewer reads the claim
    against the evidence rather than against a score.
    """
    if not creative.grounding_events or not creative.flagged_claim:
        return

    if creative.rewrite_fixed:
        banner_bg, banner_border, banner_fg = "#f0fdf4", "#bbf7d0", "#16a34a"
        banner = "✅ Flagged on the first draft · corrected on the rewrite · re-checked and cleared"
    elif creative.rewrite_attempted:
        banner_bg, banner_border, banner_fg = "#fdf4ff", "#c084fc", "#7c3aed"
        banner = "🛑 Flagged · rewritten once · still failed the re-check · halted for human review"
    else:
        banner_bg, banner_border, banner_fg = "#fdf4ff", "#c084fc", "#7c3aed"
        banner = "🛑 Flagged · halted before any rewrite"

    layer_label = {
        "deterministic": "deterministic check (no model call)",
        "judge": "LLM judge",
    }.get(creative.check_layer, creative.check_layer or "—")

    st.markdown(
        f"<div style='background:{banner_bg}; border:1px solid {banner_border}; border-radius:8px; "
        f"padding:10px 14px; margin:8px 0; font-size:0.85rem; color:{banner_fg}; font-weight:600;'>"
        f"{banner}</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"Caught by: {layer_label} · {creative.failure_type or ''}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div style='font-size:0.72rem; font-weight:700; letter-spacing:0.06em; "
                    "text-transform:uppercase; color:#dc2626; margin-bottom:6px;'>Flagged claim</div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.85rem; color:#374151; line-height:1.55;'>"
                    f"{creative.flagged_claim}</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div style='font-size:0.72rem; font-weight:700; letter-spacing:0.06em; "
                    "text-transform:uppercase; color:#64748b; margin-bottom:6px;'>What the source says</div>",
                    unsafe_allow_html=True)
        if creative.source_line:
            st.markdown(f"<div style='font-size:0.85rem; color:#374151; line-height:1.55; "
                        f"font-style:italic;'>“{creative.source_line}”</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='font-size:0.85rem; color:#94a3b8; line-height:1.55;'>"
                        "No retrieved document addresses this claim — there is no source to check it "
                        "against, which is its own problem.</div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div style='font-size:0.72rem; font-weight:700; letter-spacing:0.06em; "
                    "text-transform:uppercase; color:#16a34a; margin-bottom:6px;'>Corrected version</div>",
                    unsafe_allow_html=True)
        if creative.corrected_claim:
            st.markdown(f"<div style='font-size:0.85rem; color:#374151; line-height:1.55;'>"
                        f"{creative.corrected_claim}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='font-size:0.85rem; color:#94a3b8; line-height:1.55;'>"
                        "No accepted correction.</div>", unsafe_allow_html=True)

    if creative.rewrite_attempted and creative.revised_draft:
        with st.expander("Both drafts", expanded=not creative.rewrite_fixed):
            d1, d2 = st.columns(2)
            with d1:
                st.markdown("**Draft 1 — as written**")
                st.markdown(f"<div style='font-size:0.85rem; color:#6b7280; line-height:1.6; "
                            f"white-space:pre-wrap;'>{creative.original_draft}</div>",
                            unsafe_allow_html=True)
            with d2:
                st.markdown("**Draft 2 — after rewrite**")
                st.markdown(f"<div style='font-size:0.85rem; color:#374151; line-height:1.6; "
                            f"white-space:pre-wrap;'>{creative.revised_draft}</div>",
                            unsafe_allow_html=True)


def run_single_campaign(user_prompt, config, campaign_history=None, series_position=1):
    from agents.brand_research_agent import run_brand_research
    from agents.campaign_strategy_agent import run_campaign_strategy
    from agents.creative_execution_agent import run_creative_execution

    research_query = f"What brand facts, certifications, and approved claims support this campaign: {user_prompt}"
    t0 = time.time()
    research = run_brand_research(
        query=research_query,
        include_poisoned=config["include_poisoned"],
        simulate_low_confidence=config["simulate_low_confidence"],
    )
    t1 = time.time()
    strategy = run_campaign_strategy(
        research=research,
        campaign_brief=user_prompt,
        trust_aware=config["trust_aware"],
        campaign_history=campaign_history,
        series_position=series_position,
    )
    t2 = time.time()
    creative = run_creative_execution(strategy=strategy, brand_name="Verdant")
    t3 = time.time()
    return research, strategy, creative, (t0, t1, t2, t3)


def trust_badge_html(score, halted=False):
    if halted:
        return "<span class='badge badge-purple'>🛑 HALTED</span>"
    if score >= 0.70:
        return f"<span class='badge badge-green'>✅ {score:.2f}</span>"
    elif score >= 0.50:
        return f"<span class='badge badge-yellow'>⚠️ {score:.2f}</span>"
    else:
        return f"<span class='badge badge-red'>🔴 {score:.2f}</span>"


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
        st.markdown("<p style='font-size:0.85rem; color:#94a3b8; text-align:center; margin-bottom:1rem;'>Each campaign briefs the next with its approved copy and what the audience said about it. Brand facts are re-retrieved from source every cycle, and every cycle passes the grounding check before anything is generated.</p>", unsafe_allow_html=True)

        completed = []
        campaign_history = []

        series_progress = st.progress(0)
        series_status = st.empty()

        from opentelemetry import trace as otel_trace
        _tracer = otel_trace.get_tracer(__name__)

        for i in range(num_campaigns):
            from agents.brand_research_agent import run_brand_research
            from agents.campaign_strategy_agent import run_campaign_strategy
            from agents.creative_execution_agent import run_creative_execution

            series_status.info(f"Generating campaign {i+1} of {num_campaigns}…")
            st.markdown(f"<div style='font-size:0.85rem; font-weight:700; color:#64748b; margin:1rem 0 4px 0; text-transform:uppercase; letter-spacing:0.05em;'>Campaign {i+1} · {drop_dates[i].strftime('%b %d, %Y')}</div>", unsafe_allow_html=True)

            t0 = time.time()

            with _tracer.start_as_current_span(f"campaign-{i+1}") as campaign_span:
                campaign_span.set_attribute("openinference.span.kind", "CHAIN")
                campaign_span.set_attribute("input.value", user_prompt)
                campaign_span.set_attribute("campaign.number", i + 1)
                campaign_span.set_attribute("campaign.drop_date", drop_dates[i].isoformat())
                campaign_span.set_attribute("campaign.series_total", num_campaigns)
                campaign_span.set_attribute("campaign.brief", user_prompt)

                with st.status("🔍 Researching brand facts...", expanded=True) as s1:
                    research_query = f"What brand facts, certifications, and approved claims support this campaign: {user_prompt}"
                    research = run_brand_research(
                        query=research_query,
                        include_poisoned=config["include_poisoned"],
                        simulate_low_confidence=config["simulate_low_confidence"],
                        series_position=i + 1,
                    )
                    t1 = time.time()
                    gs = research.grounding_score
                    score_class = "trust-high" if gs >= 0.70 else "trust-medium" if gs >= 0.50 else "trust-low"
                    st.markdown(f"**Brand grounding:** <span class='{score_class}'>{gs:.2f}</span>", unsafe_allow_html=True)
                    s1.update(label=f"✅ Research complete — grounding {gs:.2f}", state="complete", expanded=False)

                with st.status("📣 Building campaign strategy...", expanded=True) as s2:
                    strategy = run_campaign_strategy(
                        research=research,
                        campaign_brief=user_prompt,
                        trust_aware=config["trust_aware"],
                        campaign_history=campaign_history or None,
                        series_position=i + 1,
                    )
                    t2 = time.time()
                    if strategy.hallucination_detected:
                        st.error("🚨 Prohibited claims detected — pipeline will halt at creative step")
                    else:
                        st.success("✅ Claims validated against brand policy")
                    st.caption(f"{len(strategy.key_messages)} key messages · {len(strategy.risk_flags)} risk flags · {t2-t1:.1f}s")
                    s2.update(
                        label="🚨 Strategy flagged — prohibited claims" if strategy.hallucination_detected else "✅ Campaign strategy ready",
                        state="error" if strategy.hallucination_detected else "complete",
                        expanded=False,
                    )

                with st.status("🎬 Generating creative...", expanded=True) as s3:
                    creative = run_creative_execution(strategy=strategy, brand_name="Verdant")
                    t3 = time.time()
                    if creative.rewrite_attempted:
                        st.caption(
                            "Grounding check flagged the first draft · one rewrite attempted · "
                            + ("re-check passed" if creative.rewrite_fixed else "re-check failed")
                        )
                    if creative.status == "HALTED":
                        s3.update(label="🛑 Trust gate fired — no creative generated", state="error", expanded=False)
                    else:
                        s3.update(label=f"✅ Creative ready · {t3-t2:.0f}s", state="complete", expanded=False)

                campaign_span.set_attribute("campaign.grounding_score", research.grounding_score)
                campaign_span.set_attribute("campaign.halted", creative.status == "HALTED")
                campaign_span.set_attribute("campaign.caption", creative.caption or "")
                campaign_span.set_attribute("campaign.video_prompt", creative.video_prompt or "")
                campaign_span.set_attribute(
                    "output.value",
                    f"HALTED" if creative.status == "HALTED"
                    else f"{creative.tagline} | grounding={research.grounding_score:.2f}"
                )

            approved = creative.status != "HALTED"

            # Comments belong to published campaigns. A halted cycle was never
            # published, so it collects none — and contributes nothing to the
            # next brief. The rejected draft is never passed forward.
            comments = (
                AUDIENCE_COMMENTS[len(campaign_history)]
                if approved and len(campaign_history) < len(AUDIENCE_COMMENTS)
                else []
            )

            campaign_span.set_attribute("campaign.approved", approved)
            campaign_span.set_attribute("campaign.check_layer", creative.check_layer or "none")
            campaign_span.set_attribute("campaign.failure_type", creative.failure_type or "")
            campaign_span.set_attribute("campaign.flagged_claim", creative.flagged_claim or "")
            campaign_span.set_attribute("campaign.rewrite_attempted", creative.rewrite_attempted)
            campaign_span.set_attribute("campaign.rewrite_fixed", creative.rewrite_fixed)

            completed.append({
                "n": i + 1,
                "drop_date": drop_dates[i],
                "research": research,
                "strategy": strategy,
                "creative": creative,
                "elapsed": t3 - t0,
                "comments": comments,
            })

            render_grounding_record(creative)

            if comments and i < num_campaigns - 1:
                comment_html = "".join(
                    f"<div style='font-size:0.85rem; color:#374151; line-height:1.6;'>“{c}”</div>"
                    for c in comments
                )
                st.markdown(f"""
                <div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:14px 16px; margin:10px 0;'>
                    <div style='font-size:0.72rem; font-weight:700; letter-spacing:0.07em; text-transform:uppercase; color:#64748b; margin-bottom:8px;'>
                        💬 Audience comments — {drop_dates[i].strftime('%b %d')}
                    </div>
                    {comment_html}
                    <div style='font-size:0.78rem; color:#94a3b8; margin-top:8px;'>
                        Carried into the next brief alongside this campaign's approved copy. Brand facts are re-retrieved from source.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if approved:
                campaign_history.append({
                    "tagline": strategy.tagline,
                    "campaign_concept": strategy.campaign_concept,
                    "key_messages": strategy.key_messages,
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

        import pandas as pd
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:0.85rem; font-weight:600; color:#94a3b8; letter-spacing:0.05em; text-transform:uppercase; margin-bottom:4px;'>
            Grounding record — every cycle
        </div>
        <div style='font-size:0.8rem; color:#64748b; margin-bottom:8px;'>
            What was flagged, which layer caught it, and whether the one allowed rewrite fixed it.
        </div>
        """, unsafe_allow_html=True)

        record_rows = []
        for c in completed:
            cr = c["creative"]
            if not cr.flagged_claim:
                outcome = "clean — passed first check"
            elif cr.rewrite_fixed:
                outcome = "rewrite fixed it"
            elif cr.rewrite_attempted:
                outcome = "rewrite failed — halted"
            else:
                outcome = "halted, no rewrite"
            record_rows.append({
                "Cycle": c["n"],
                "Published": "no" if cr.status == "HALTED" else "yes",
                "Flagged": (cr.flagged_claim or "—").replace("\n", " ")[:70],
                "Caught by": cr.check_layer or "—",
                "Type": cr.failure_type or "—",
                "Attempts": len(cr.grounding_events) or 1,
                "Outcome": outcome,
                "Grounding": f"{c['research'].grounding_score:.2f}",
            })
        st.dataframe(pd.DataFrame(record_rows), use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        for c in completed:
            halted = c["creative"].status == "HALTED"
            score = c["research"].grounding_score
            badge = trust_badge_html(score, halted)
            header_html = f"""
            <div style='display:flex; align-items:center; gap:12px; padding:10px 0 4px 0;'>
                <div style='font-weight:700; font-size:1rem; color:{"#7c3aed" if halted else "#1a1a1a"};'>
                    {"🛑" if halted else f"#{c['n']}"} Campaign {c['n']}
                </div>
                <div style='font-size:0.8rem; color:#94a3b8;'>{c["drop_date"].strftime("%b %d, %Y")}</div>
                {badge}
            </div>
            """
            st.markdown(header_html, unsafe_allow_html=True)

            if halted:
                st.markdown(f"<div style='background:#fdf4ff; border:1.5px solid #c084fc; border-radius:8px; padding:12px; font-size:0.9rem; color:#7c3aed;'>🛑 {c['creative'].halt_reason}</div>", unsafe_allow_html=True)
            else:
                with st.expander(f'"{c["strategy"].tagline}"', expanded=(c["n"] == 1)):
                    vcol, ccol = st.columns([3, 2])
                    with vcol:
                        if c["creative"].video_bytes:
                            st.video(io.BytesIO(c["creative"].video_bytes), format="video/mp4")
                        elif c["creative"].video_url:
                            st.video(c["creative"].video_url)
                        if c["creative"].video_prompt:
                            with st.expander("Video prompt", expanded=False):
                                st.markdown(f"*{c['creative'].video_prompt}*")
                    with ccol:
                        if c["creative"].caption:
                            st.markdown("<div style='font-size:0.75rem; font-weight:600; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:4px;'>Caption</div>", unsafe_allow_html=True)
                            st.markdown(f"<div style='font-size:0.9rem; color:#374151; line-height:1.6;'>{c['creative'].caption}</div>", unsafe_allow_html=True)
                        if c["creative"].hashtags:
                            st.markdown(f"<div style='color:#475569; font-size:0.85rem; margin-top:8px;'>{' '.join(c['creative'].hashtags)}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='margin-top:12px;'><span style='font-size:0.8rem; color:#94a3b8;'>Grounding: {score:.2f} · {c['elapsed']:.0f}s</span></div>", unsafe_allow_html=True)

            st.markdown("<div style='border-bottom:1px solid #f1f5f9; margin:4px 0 8px 0;'></div>", unsafe_allow_html=True)

        with pipeline_status.container():
            for c in completed:
                icon = "🛑" if c["creative"].status == "HALTED" else "✅"
                st.markdown(f"**Campaign {c['n']}** {icon} {c['research'].grounding_score:.2f}")
            st.progress(1.0)

    # ── SINGLE CAMPAIGN MODE ──────────────────────────────────────────────
    else:
        from agents.brand_research_agent import run_brand_research
        from agents.campaign_strategy_agent import run_campaign_strategy
        from agents.creative_execution_agent import run_creative_execution
        from opentelemetry import trace as otel_trace
        _tracer = otel_trace.get_tracer(__name__)

        research_query = f"What brand facts, certifications, and approved claims support this campaign: {user_prompt}"

        with _tracer.start_as_current_span("campaign-1") as campaign_span:
            campaign_span.set_attribute("openinference.span.kind", "CHAIN")
            campaign_span.set_attribute("input.value", user_prompt)
            campaign_span.set_attribute("campaign.number", 1)
            campaign_span.set_attribute("campaign.series_total", 1)
            campaign_span.set_attribute("campaign.brief", user_prompt)

            with st.status("🔍 Researching brand facts...", expanded=True) as s1:
                pipeline_status.markdown("**Step 1 of 3** — Brand Research")
                t0 = time.time()
                research = run_brand_research(
                    query=research_query,
                    include_poisoned=config["include_poisoned"],
                    simulate_low_confidence=config["simulate_low_confidence"],
                )
                t1 = time.time()

                gs = research.grounding_score
                score_class = "trust-high" if gs >= 0.70 else "trust-medium" if gs >= 0.50 else "trust-low"
                score_label = "Strong" if gs >= 0.70 else "Moderate" if gs >= 0.50 else "Weak"

                st.markdown(f"**Brand grounding:** <span class='{score_class}'>{score_label} ({gs:.2f})</span>", unsafe_allow_html=True)
                st.markdown(f"{research.answer}")
                st.caption(f"Sources: {', '.join(research.sources)} · {(t1-t0):.1f}s")

                if config["include_poisoned"]:
                    st.error("🔴 Injection document detected in retrieval")
                if config["simulate_low_confidence"] and gs < 0.6:
                    st.warning(f"⚠️ Low grounding ({gs:.2f}) — Agent 2 will not be informed in this scenario")

                s1.update(label=f"✅ Brand research complete — grounding {gs:.2f}", state="complete", expanded=False)

            with pipeline_status.container():
                st.markdown("**Agent 1** ✅ Brand Research")
                st.progress(0.33)

            with st.status("📣 Building campaign strategy...", expanded=True) as s2:
                pipeline_status.markdown("**Step 2 of 3** — Campaign Strategy")
                t2 = time.time()
                strategy = run_campaign_strategy(
                    research=research,
                    campaign_brief=user_prompt,
                    trust_aware=config["trust_aware"],
                )
                t3 = time.time()

                if strategy.hallucination_detected:
                    st.error("🚨 Prohibited claims detected in output — pipeline will halt at creative step")
                elif not config["trust_aware"] and research.grounding_score < 0.70:
                    st.warning("⚠️ Strategy generated without grounding context — claims may be unsupported")
                else:
                    st.success("✅ All claims validated against brand policy")

                st.caption(f"{(t3-t2):.1f}s · {len(strategy.key_messages)} key messages · {len(strategy.risk_flags)} risk flags")

                s2.update(
                    label=f"{'🚨 Strategy flagged — prohibited claims' if strategy.hallucination_detected else '✅ Campaign strategy ready'}",
                    state="error" if strategy.hallucination_detected else "complete",
                    expanded=False
                )

            with pipeline_status.container():
                st.markdown("**Agent 1** ✅ Brand Research")
                st.markdown("**Agent 2** ✅ Campaign Strategy")
                st.progress(0.66)

            with st.status("🎬 Generating creative...", expanded=True) as s3:
                pipeline_status.markdown("**Step 3 of 3** — Creative Execution")
                t4 = time.time()
                creative = run_creative_execution(strategy=strategy, brand_name="Verdant")
                t5 = time.time()

                if creative.status == "HALTED":
                    s3.update(label="🛑 Creative halted — trust gate fired", state="error", expanded=False)
                else:
                    s3.update(label="✅ Creative package ready", state="complete", expanded=False)

            campaign_span.set_attribute("campaign.grounding_score", research.grounding_score)
            campaign_span.set_attribute("campaign.halted", creative.status == "HALTED")
            campaign_span.set_attribute("campaign.caption", creative.caption or "")
            campaign_span.set_attribute("campaign.video_prompt", creative.video_prompt or "")
            campaign_span.set_attribute(
                "output.value",
                "HALTED" if creative.status == "HALTED"
                else f"{creative.tagline} | grounding={research.grounding_score:.2f}"
            )

        with pipeline_status.container():
            st.markdown("**Agent 1** ✅ Brand Research")
            st.markdown("**Agent 2** ✅ Campaign Strategy")
            if creative.status == "HALTED":
                st.markdown("**Agent 3** 🛑 Halted")
                st.progress(1.0)
            else:
                st.markdown("**Agent 3** ✅ Creative Execution")
                st.progress(1.0)
            st.caption(f"Total: {(t5-t0):.1f}s")

    if num_campaigns == 1:
        st.divider()

    if num_campaigns == 1 and creative.status == "HALTED":
        st.markdown(f"""
        <div class='halted-card'>
            <div class='step-label'>Pipeline halted</div>
            <h3 style='color:#7c3aed; margin:8px 0;'>🛑 Content generation stopped</h3>
            <p style='color:#4b5563;'>{creative.halt_reason}</p>
            <p style='color:#94a3b8; font-size:0.85rem;'>No video or caption was generated. This is intentional — the trust gate prevents brand-unsafe content from being created, not just flagged after the fact.</p>
        </div>
        """, unsafe_allow_html=True)

        render_grounding_record(creative)

        st.markdown("---")
        st.markdown("**Recommended next steps**")
        halt_text = creative.halt_reason or ""
        if "not supported by the brand documents" in halt_text:
            actions = [
                ("🛑 Do not publish", "A factual claim in this copy is not supported by the retrieved brand documents."),
                ("🔍 Check the flagged claim", "The halt reason names it. Verify against the brand guide or sustainability report."),
                ("👤 Route to brand reviewer", "A human decides whether the claim is defensible or the copy needs rewriting."),
                ("📋 Create audit record", "Log span_id, failure_type=UNSUPPORTED_CLAIM, action=HALTED — required for EU AI Act Article 13."),
            ]
        elif "not in Verdant's approved statistics" in halt_text:
            actions = [
                ("🛑 Do not publish", "Cites a statistic outside the approved set."),
                ("🔢 Verify or remove the figure", "Approved statistics live in the brand guide. If the number is real, get it added there."),
                ("👤 Route to brand reviewer", "Thirty seconds of review, versus a published figure Verdant cannot substantiate."),
                ("📋 Create audit record", "Log span_id, failure_type=UNAPPROVED_STAT, action=HALTED."),
            ]
        elif "names the competitor" in halt_text:
            actions = [
                ("🛑 Do not publish", "Brand guide prohibits direct comparison to competitors by name."),
                ("✏️ Rewrite without the name", "Make the point on Verdant's own evidence."),
                ("📋 Create audit record", "Log span_id, failure_type=COMPETITOR_NAMED, action=HALTED."),
            ]
        elif "could not run" in halt_text:
            actions = [
                ("⏸️ Held, not cleared", "The grounding judge was unavailable, so this copy was never verified."),
                ("🔧 Check judge availability", "Confirm phoenix[evals] is installed and OPENAI_API_KEY is valid, then re-run."),
                ("👤 Human review required", "Do not publish on the assumption that unverified means clean."),
            ]
        elif strategy.hallucination_detected or "prohibited claim" in halt_text:
            actions = [
                ("🛑 Do not publish", "Contains prohibited claims — do not deliver to any downstream system."),
                ("👤 Escalate to brand safety team", "Flag for manual review within 1 business hour."),
                ("🔒 Quarantine source document", "Remove flagged chunk from active index. Source: `verdant_poisoned.txt`."),
                ("📋 Create audit record", f"Log span_id, injection_risk=HIGH, action=HALTED — required for EU AI Act Article 13."),
            ]
        else:
            actions = [
                ("⚠️ Do not auto-publish", "Confidence too low — flag as LOW_CONFIDENCE before any use."),
                ("🔄 Retry with expanded sources", "Increase retrieval top-k and run again."),
                ("👤 Human review required", "Route to brand reviewer before any customer-facing use."),
                ("📋 Create audit record", f"Log span_id, grounding_score=LOW, action=FLAGGED — required for SOC 2 CC7.2."),
            ]
        for action, detail in actions:
            st.markdown(f"**{action}** — {detail}")

    elif num_campaigns == 1:
        render_grounding_record(creative)
        st.markdown(f"<div class='tagline-output'>\"{strategy.tagline}\"</div>", unsafe_allow_html=True)

        col_meta1, col_meta2 = st.columns(2)
        with col_meta1:
            st.markdown("<div class='step-label'>Campaign Purpose</div>", unsafe_allow_html=True)
            purpose = strategy.key_messages[0] if strategy.key_messages else strategy.campaign_concept
            st.markdown(f"<div style='font-size:0.95rem; color:#374151; line-height:1.6;'>{purpose}</div>", unsafe_allow_html=True)
        with col_meta2:
            st.markdown("<div class='step-label'>Campaign Description</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:0.95rem; color:#374151; line-height:1.6;'>{strategy.campaign_concept}</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown("<div class='step-label'>Social Video</div>", unsafe_allow_html=True)
            if creative.video_bytes:
                st.video(io.BytesIO(creative.video_bytes), format="video/mp4")
            elif creative.video_url:
                st.video(creative.video_url)
            else:
                if creative.video_error:
                    st.warning(f"⚠️ Video generation failed: `{creative.video_error}`")
                else:
                    st.markdown("""
                <div style='background:#f1f5f9; border-radius:10px; padding:40px; text-align:center; color:#64748b;'>
                    <div style='font-size:2rem;'>📽️</div>
                    <div style='font-weight:600; margin-top:8px;'>Veo 3.1 Lite ready to generate</div>
                    <div style='font-size:0.85rem; margin-top:4px;'>Add a Google API key to generate the video</div>
                </div>
                    """, unsafe_allow_html=True)

            if creative.video_prompt:
                with st.expander("Video prompt", expanded=False):
                    st.markdown(f"*{creative.video_prompt}*")

        with col_right:
            st.markdown("<div class='step-label'>Caption</div>", unsafe_allow_html=True)
            if creative.caption:
                st.markdown(f"""
                <div style='background:#f8fafc; border-radius:10px; padding:16px; font-size:0.95rem; line-height:1.6; color:#1e293b;'>
                {creative.caption}
                </div>
                """, unsafe_allow_html=True)

            if creative.hashtags:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<div class='step-label'>Hashtags</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='color:#475569; font-size:0.9rem;'>{' '.join(creative.hashtags)}</div>", unsafe_allow_html=True)

            if len(strategy.key_messages) > 1:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<div class='step-label'>Key Messages</div>", unsafe_allow_html=True)
                for msg in strategy.key_messages[1:]:
                    st.markdown(f"<div style='padding:6px 0; border-bottom:1px solid #f1f5f9; font-size:0.9rem; color:#374151;'>→ {msg}</div>", unsafe_allow_html=True)

        st.markdown("---")
        with st.expander("🔬 Trust signals", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Agent 1**")
                st.code(f"grounding_score: {research.grounding_score:.2f}\ninjection_risk: {research.confidence_metadata.get('injection_risk','none').upper()}")
            with c2:
                st.markdown("**Agent 2**")
                st.code(f"score_inherited: {strategy.trust_score_inherited:.2f}\ntrust_aware: {strategy.trust_aware_mode}\nhallucination: {strategy.hallucination_detected}")
            with c3:
                st.markdown("**Agent 3**")
                st.code(f"pipeline_halted: {creative.status == 'HALTED'}\nvideo_generated: {bool(creative.video_url or creative.video_bytes)}")

            if not config["trust_aware"]:
                st.warning("⚠️ Trust gap visible: Agent 1's grounding score exists, but Agent 2 received no trust context.")

elif run_btn and not user_prompt.strip():
    st.warning("Please describe your campaign before generating.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption("Campaign Studio · Built by Rebecca Riggs · 2026")
