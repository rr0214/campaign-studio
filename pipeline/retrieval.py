"""
RETRIEVE — step 1. Plain function, no model call.
=================================================

Queries the Verdant brand documents and returns chunks. The model never decides
what to retrieve or whether to retrieve; this always runs, always the same way.

Brand facts are retrieved fresh on every call. Nothing from a prior cycle — not
chunks, not prior copy, not audience comments — reaches this step. That is
enforced by the signature: there is nowhere to pass it.

Vector-store code is duplicated from agents/brand_research_agent.py rather than
imported, so retiring agents/ later is a clean delete instead of an unwind.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


BRAND_DOCS_DIR = Path(__file__).parent.parent / "data" / "brand_docs"

EMBEDDING_MODEL = "text-embedding-3-small"

# Retrieval-quality cutoff. Reported, never gating — this score is built from
# retrieval signals and never sees generated text, which is precisely why it was
# removed from the gate. VERIFY reads the text instead.
RETRIEVAL_QUALITY_THRESHOLD = 0.70


@dataclass
class RetrievalResult:
    query: str
    chunks: list            # [{text, doc_name, relevance_score}]
    sources: list
    grounding_score: float
    metadata: dict = field(default_factory=dict)

    @property
    def sources_text(self) -> str:
        return "\n\n---\n\n".join(
            f"[Source: {c['doc_name']}]\n{c['text']}" for c in self.chunks
        )


_chroma_client = None
_collection = None
_collection_mode = None


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


def _get_collection(include_poisoned: bool = False) -> chromadb.Collection:
    """Load brand documents into an in-memory ChromaDB, cached per mode."""
    global _chroma_client, _collection, _collection_mode

    if _collection is not None and _collection_mode == include_poisoned:
        return _collection

    embedding_fn = OpenAIEmbeddingFunction(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model_name=EMBEDDING_MODEL,
    )
    _chroma_client = chromadb.Client()
    collection_name = "verdant_docs_poisoned" if include_poisoned else "verdant_docs"

    try:
        _chroma_client.delete_collection(collection_name)
    except Exception:
        pass

    _collection = _chroma_client.create_collection(
        name=collection_name, embedding_function=embedding_fn
    )

    docs_to_load = [
        "verdant_brand_guide.txt",
        "verdant_products.txt",
        "verdant_sustainability.txt",
    ]
    if include_poisoned:
        docs_to_load.append("verdant_poisoned.txt")

    documents, ids, metadatas = [], [], []
    for doc_file in docs_to_load:
        doc_path = BRAND_DOCS_DIR / doc_file
        if not doc_path.exists():
            continue
        for i, chunk in enumerate(_chunk_text(doc_path.read_text(), max_chars=600)):
            documents.append(chunk)
            ids.append(f"{doc_file}_{i}")
            metadatas.append({"source": doc_file, "chunk_index": i})

    if documents:
        _collection.add(documents=documents, ids=ids, metadatas=metadatas)

    _collection_mode = include_poisoned
    return _collection


def _grounding_score(chunks: list, n_results: int) -> float:
    """
    Retrieval-quality heuristic: mean relevance, coverage, and chunk count.
    Reported for continuity with the old pipeline. It says nothing about whether
    generated text is supported — that is VERIFY's job.
    """
    if not chunks:
        return 0.0
    scores = [c.get("relevance_score", 0.5) for c in chunks]
    avg_relevance = sum(scores) / len(scores)
    coverage = sum(1 for s in scores if s >= 0.70) / max(len(scores), 1)
    n_penalty = min(len(chunks) / max(n_results, 1), 1.0)
    score = (avg_relevance * 0.5) + (coverage * 0.35) + (n_penalty * 0.15)
    return round(min(max(score, 0.0), 1.0), 3)


def derive_research_query(brief: str) -> str:
    """
    The retrieval query the pipeline asks, derived from the campaign brief.

    Single copy on purpose. evals/run_evals.py used to pass the golden dataset's
    hand-written `query` column straight to retrieve(), which meant the eval was
    scoring retrieval on a question the production path never asks — the app only
    ever has a brief, and builds the query from it with this template.
    """
    return f"What brand facts, certifications, and approved claims support this campaign: {brief}"


def retrieve(
    query: str,
    include_poisoned: bool = False,
    simulate_low_confidence: bool = False,
    n_results: int = 4,
) -> RetrievalResult:
    """Query the brand documents. No model call, no branching on model output."""
    effective_n = 1 if simulate_low_confidence else n_results

    collection = _get_collection(include_poisoned=include_poisoned)
    results = collection.query(
        query_texts=[query],
        n_results=min(effective_n, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    raw_distances = results["distances"][0] if results["distances"] else []
    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        distance = raw_distances[i] if i < len(raw_distances) else 1.0
        similarity = max(0.0, 1.0 - (distance / 2.0))
        chunks.append({
            "text": doc,
            "doc_name": results["metadatas"][0][i].get("source", "unknown"),
            "relevance_score": round(similarity, 3),
        })

    score = _grounding_score(chunks, n_results)
    return RetrievalResult(
        query=query,
        chunks=chunks,
        sources=sorted({c["doc_name"] for c in chunks}),
        grounding_score=score,
        metadata={
            "n_chunks_retrieved": len(chunks),
            "avg_relevance": round(
                sum(c["relevance_score"] for c in chunks) / max(len(chunks), 1), 3
            ),
            "retrieval_quality": "low" if score < RETRIEVAL_QUALITY_THRESHOLD else "high",
            "injection_risk": "high" if include_poisoned else "low",
            "simulated_low_confidence": simulate_low_confidence,
        },
    )
