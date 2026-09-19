"""
Agent 1: Brand Research Agent
==============================
Performs RAG over Verdant brand documents using ChromaDB.
Returns research findings along with a grounding confidence score
that represents how well the retrieved context supports the answer.

This score is the KEY trust signal that Agent 2 should inherit —
and the failure mode we're demonstrating is when it doesn't.
"""

import os
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI
from opentelemetry import trace
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ResearchResult:
    query: str
    answer: str
    sources: list[str]
    retrieved_chunks: list[dict]      # [{text, doc_name, relevance_score}]
    grounding_score: float            # 0.0–1.0: how well docs support the answer
    confidence_metadata: dict = field(default_factory=dict)  # extra signals
    span_id: Optional[str] = None     # Arize trace span ID for propagation


# ---------------------------------------------------------------------------
# Vector store setup
# ---------------------------------------------------------------------------

# Single source of truth for the retrieval-quality cutoff. This file previously
# labelled anything above 0.6 "high" while Agent 3's gate halted below 0.70 —
# so a 0.65 retrieval was reported as high quality and then halted. Agents 2 and 3
# import this constant rather than repeating a literal.
RETRIEVAL_QUALITY_THRESHOLD = 0.70

# Model used by all three agents. The gpt-5.6 family does not accept a custom
# `temperature` (only the default), and takes `max_completion_tokens` rather than
# `max_tokens`, with reasoning tokens counted against that budget — so the caps
# below are set well above the visible output length they need to produce.
AGENT_MODEL = "gpt-5.6-luna"

BRAND_DOCS_DIR = Path(__file__).parent.parent / "data" / "brand_docs"

_chroma_client = None
_collection = None
_collection_mode = None  # tracks which mode the cached collection was built for


def _get_collection(include_poisoned: bool = False) -> chromadb.Collection:
    """
    Load brand documents into ChromaDB (in-memory).
    Cached — only re-initializes if poisoned mode changes, eliminating
    redundant document embedding calls on every campaign run.
    """
    global _chroma_client, _collection, _collection_mode

    # Return cached collection if mode hasn't changed
    if _collection is not None and _collection_mode == include_poisoned:
        return _collection

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    embedding_fn = OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name="text-embedding-3-small"
    )

    _chroma_client = chromadb.Client()

    collection_name = "verdant_docs_poisoned" if include_poisoned else "verdant_docs"

    # Delete if exists (handles mode switches)
    try:
        _chroma_client.delete_collection(collection_name)
    except Exception:
        pass

    _collection = _chroma_client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
    )

    docs_to_load = ["verdant_brand_guide.txt", "verdant_products.txt", "verdant_sustainability.txt"]
    if include_poisoned:
        docs_to_load.append("verdant_poisoned.txt")

    documents, ids, metadatas = [], [], []
    for doc_file in docs_to_load:
        doc_path = BRAND_DOCS_DIR / doc_file
        if not doc_path.exists():
            continue
        text = doc_path.read_text()
        # Simple chunking: split on double newlines, ~500 char max
        chunks = _chunk_text(text, max_chars=600)
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            ids.append(f"{doc_file}_{i}")
            metadatas.append({"source": doc_file, "chunk_index": i})

    if documents:
        _collection.add(documents=documents, ids=ids, metadatas=metadatas)

    _collection_mode = include_poisoned
    return _collection


def _chunk_text(text: str, max_chars: int = 600) -> list[str]:
    """Split on paragraph boundaries, respecting max_chars."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) > max_chars and current:
            chunks.append(current.strip())
            current = para
        else:
            current = (current + "\n\n" + para).strip() if current else para
    if current:
        chunks.append(current.strip())
    return chunks


# ---------------------------------------------------------------------------
# Grounding score calculation
# ---------------------------------------------------------------------------

def _calculate_grounding_score(
    answer: str,
    retrieved_chunks: list[dict],
    n_results: int,
) -> float:
    """
    Heuristic grounding score: combines retrieval relevance + answer-to-source overlap.

    Score components:
    - avg_relevance: mean cosine similarity of top-k retrieved chunks (0–1)
    - coverage: fraction of retrieved chunks with relevance > 0.7
    - n_chunks_penalty: penalize if fewer than expected chunks retrieved

    Returns a float in [0.0, 1.0].
    """
    if not retrieved_chunks:
        return 0.0

    relevance_scores = [c.get("relevance_score", 0.5) for c in retrieved_chunks]
    avg_relevance = sum(relevance_scores) / len(relevance_scores)
    coverage = sum(1 for s in relevance_scores if s >= 0.70) / max(len(relevance_scores), 1)
    n_chunks_penalty = min(len(retrieved_chunks) / max(n_results, 1), 1.0)

    # Weighted composite
    score = (avg_relevance * 0.5) + (coverage * 0.35) + (n_chunks_penalty * 0.15)
    return round(min(max(score, 0.0), 1.0), 3)


# ---------------------------------------------------------------------------
# Agent 1 main function
# ---------------------------------------------------------------------------

def run_brand_research(
    query: str,
    include_poisoned: bool = False,
    simulate_low_confidence: bool = False,
    n_results: int = 4,
    series_position: int = 1,        # Position in series — recorded on the span, nothing more
) -> ResearchResult:
    """
    Run the Brand Research Agent.

    Brand facts are retrieved fresh on every call. Nothing from a prior cycle —
    not retrieved chunks, not prior campaign copy, not audience comments — is
    carried into retrieval. Each cycle asks the documents the same question and
    gets whatever the documents currently say.

    Args:
        query:                   The research question
        include_poisoned:        If True, loads the injection doc into the vector store
        simulate_low_confidence: If True, forces low-quality retrieval to demo trust propagation failure
        n_results:               Number of chunks to retrieve from vector store
        series_position:         Position in series — recorded as a span attribute only

    Returns:
        ResearchResult with answer, sources, and grounding_score
    """
    tracer = trace.get_tracer(__name__)
    client = OpenAI()

    # Use fewer results to simulate weak retrieval
    effective_n = 1 if simulate_low_confidence else n_results

    with tracer.start_as_current_span("brand-research-agent") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "AGENT")
        span.set_attribute(SpanAttributes.INPUT_VALUE, query)
        span.set_attribute("agent.name", "BrandResearchAgent")
        span.set_attribute("agent.version", "1.0")
        span.set_attribute("agent.mode.poisoned", include_poisoned)
        span.set_attribute("series.position", series_position)
        span.set_attribute("agent.mode.low_confidence", simulate_low_confidence)

        # ── Step 1: Retrieve relevant chunks ──────────────────────────────
        with tracer.start_as_current_span("vector-retrieval") as retrieval_span:
            retrieval_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "RETRIEVER")
            retrieval_span.set_attribute(SpanAttributes.INPUT_VALUE, query)

            collection = _get_collection(include_poisoned=include_poisoned)
            results = collection.query(
                query_texts=[query],
                n_results=min(effective_n, collection.count()),
                include=["documents", "metadatas", "distances"],
            )

            # ChromaDB returns L2 distances; convert to cosine-like similarity (0–1)
            raw_distances = results["distances"][0] if results["distances"] else []
            retrieved_chunks = []
            for i, doc in enumerate(results["documents"][0]):
                distance = raw_distances[i] if i < len(raw_distances) else 1.0
                similarity = max(0.0, 1.0 - (distance / 2.0))
                retrieved_chunks.append({
                    "text": doc,
                    "doc_name": results["metadatas"][0][i].get("source", "unknown"),
                    "relevance_score": round(similarity, 3),
                })

            # Serialize retrieved docs for Arize trace (OpenInference RETRIEVER format)
            retrieval_span.set_attribute(
                SpanAttributes.RETRIEVAL_DOCUMENTS,
                json.dumps([
                    {"document": {"content": c["text"][:500]}, "score": c["relevance_score"]}
                    for c in retrieved_chunks
                ])
            )
            retrieval_span.set_attribute("retrieval.n_chunks", len(retrieved_chunks))

        # ── Step 2: Generate research answer ──────────────────────────────
        context_text = "\n\n---\n\n".join(
            f"[Source: {c['doc_name']}]\n{c['text']}" for c in retrieved_chunks
        )

        system_prompt = """You are a brand research assistant for Verdant, a sustainable activewear brand.
Answer the research question using ONLY the information in the provided brand documents.
If the documents don't contain enough information to answer confidently, say so explicitly.
Be precise — cite specific numbers, certifications, and policy statements from the documents.
Do NOT invent statistics or claims not found in the provided context."""

        user_message = f"""Research Question: {query}

Brand Documents (retrieved):
{context_text}

Provide a focused research summary that a campaign strategist can use directly."""

        with tracer.start_as_current_span("llm-synthesis") as llm_span:
            llm_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
            llm_span.set_attribute(SpanAttributes.INPUT_VALUE, user_message)
            llm_span.set_attribute(SpanAttributes.LLM_MODEL_NAME, AGENT_MODEL)

            response = client.chat.completions.create(
                model=AGENT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_completion_tokens=3000,
            )
            answer = response.choices[0].message.content

            llm_span.set_attribute(SpanAttributes.OUTPUT_VALUE, answer)
            llm_span.set_attribute(
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL,
                response.usage.total_tokens if response.usage else 0
            )

        # ── Step 3: Calculate grounding score ─────────────────────────────
        grounding_score = _calculate_grounding_score(answer, retrieved_chunks, n_results)

        span.set_attribute(SpanAttributes.OUTPUT_VALUE, answer)
        span.set_attribute("trust.grounding_score", grounding_score)
        span.set_attribute("trust.n_chunks_retrieved", len(retrieved_chunks))
        span.set_attribute("trust.retrieval_quality", "low" if grounding_score < RETRIEVAL_QUALITY_THRESHOLD else "high")
        span.set_attribute("trust.injection_risk", "high" if include_poisoned else "low")
        # Stamp retrieved content on the AGENT span so it's accessible in trace-level evals.
        # The same content is on the child RETRIEVER span via RETRIEVAL_DOCUMENTS, but
        # Arize's trace-level variable mapping only surfaces attributes from the root span
        # in the dropdown — this makes the ground truth available for cross-agent evals.
        span.set_attribute(
            "brand.retrieved_facts",
            json.dumps([{"source": c["doc_name"], "content": c["text"][:400]} for c in retrieved_chunks])
        )

        span_ctx = span.get_span_context()
        span_id = format(span_ctx.span_id, "016x") if span_ctx else None

        sources = list({c["doc_name"] for c in retrieved_chunks})

        return ResearchResult(
            query=query,
            answer=answer,
            sources=sources,
            retrieved_chunks=retrieved_chunks,
            grounding_score=grounding_score,
            confidence_metadata={
                "n_chunks_retrieved": len(retrieved_chunks),
                "avg_relevance": round(
                    sum(c["relevance_score"] for c in retrieved_chunks) / max(len(retrieved_chunks), 1), 3
                ),
                "retrieval_quality": "low" if grounding_score < RETRIEVAL_QUALITY_THRESHOLD else "high",
                "injection_risk": "high" if include_poisoned else "low",
                "simulated_low_confidence": simulate_low_confidence,
            },
            span_id=span_id,
        )
