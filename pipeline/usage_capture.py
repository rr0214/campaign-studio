"""
Capturing token usage from calls we don't own.
==============================================

The grounding judge runs through phoenix's ClassificationEvaluator, which builds
its own OpenAI client and returns a Score carrying only
`{name, label, explanation, metadata:{model}, kind, direction}` — no token usage
at all. Verified against phoenix 20.14 on 2026-09-19.

So the judge's cost is invisible through the API we're required to reuse. Rather
than estimate it (an invented number on a cost dashboard is exactly the failure
this project exists to catch), this module briefly wraps the OpenAI SDK's create
methods for the duration of the judge call and records the `usage` block off
responses as they pass.

Deliberately narrow: patched on entry, restored on exit, no global state left
behind. If nothing is captured — a future phoenix routing calls some other way —
the step records `measured=False` and the cost is reported as unavailable, never
as zero.
"""

from contextlib import contextmanager


@contextmanager
def capture_openai_usage():
    """
    Yields a list that fills with `usage` objects from any OpenAI SDK completion
    made inside the block, including calls made by third-party libraries.
    """
    captured = []
    patches = []

    def wrap(cls, attr):
        original = getattr(cls, attr, None)
        if original is None:
            return

        def wrapped(self, *args, **kwargs):
            response = original(self, *args, **kwargs)
            usage = getattr(response, "usage", None)
            if usage is not None:
                # The request is captured alongside the usage so the exact text
                # sent can be shown later. For calls made by a third party —
                # phoenix builds the judge request itself — this is the only way
                # to see what actually went over the wire rather than guessing
                # from the template.
                captured.append({
                    "usage": usage,
                    "model": kwargs.get("model", ""),
                    "messages": kwargs.get("messages") or kwargs.get("input"),
                })
            return response

        setattr(cls, attr, wrapped)
        patches.append((cls, attr, original))

    try:
        from openai.resources.chat.completions import Completions
        wrap(Completions, "create")
    except Exception:
        pass

    try:
        from openai.resources.responses import Responses
        wrap(Responses, "create")
    except Exception:
        pass

    # Embeddings bill too — ChromaDB's OpenAIEmbeddingFunction routes here, not
    # through chat completions. Missing this reported retrieval as a measured
    # zero, which is worse than reporting it as unmeasured.
    try:
        from openai.resources.embeddings import Embeddings
        wrap(Embeddings, "create")
    except Exception:
        pass

    try:
        yield captured
    finally:
        for cls, attr, original in patches:
            setattr(cls, attr, original)


def sum_usage(usages):
    """
    Collapse captured usage blocks into one tuple.
    Returns (prompt, completion, reasoning, cached, measured).
    """
    if not usages:
        return 0, 0, 0, 0, False

    prompt = completion = reasoning = cached = 0
    for entry in usages:
        u = entry["usage"] if isinstance(entry, dict) else entry
        prompt += getattr(u, "prompt_tokens", 0) or 0
        completion += getattr(u, "completion_tokens", 0) or 0
        det_out = getattr(u, "completion_tokens_details", None)
        det_in = getattr(u, "prompt_tokens_details", None)
        if det_out:
            reasoning += getattr(det_out, "reasoning_tokens", 0) or 0
        if det_in:
            cached += getattr(det_in, "cached_tokens", 0) or 0
    return prompt, completion, reasoning, cached, True


def rendered_prompts(captured) -> list:
    """
    Flatten captured requests into readable prompt text, in call order.
    Returns [{"model", "role", "text"}].
    """
    out = []
    for entry in captured:
        if not isinstance(entry, dict):
            continue
        messages = entry.get("messages")
        if not messages:
            continue
        if isinstance(messages, str):
            out.append({"model": entry.get("model", ""), "role": "input", "text": messages})
            continue
        for m in messages:
            if isinstance(m, dict):
                content = m.get("content")
                if isinstance(content, list):  # multi-part content
                    content = " ".join(
                        c.get("text", "") for c in content if isinstance(c, dict)
                    )
                if content:
                    out.append({"model": entry.get("model", ""),
                                "role": m.get("role", ""), "text": str(content)})
    return out
