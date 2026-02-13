# agents/multi_query.py
"""Generate query variants via LLM for multi-query retrieval."""

from typing import List

_PROMPT_TEMPLATE = (
    "You are a helpful assistant that generates alternative search queries.\n"
    "Given the original question below, generate {n} alternative versions that "
    "capture the same information need but use different wording, emphasis, or "
    "decomposition. Return ONLY the alternative queries, one per line.\n\n"
    "Original question: {query}\n\n"
    "Alternative queries:"
)


def generate_query_variants(
    query: str,
    llm,
    n_variants: int = 2,
) -> List[str]:
    """
    Use *llm* to produce ``n_variants`` alternative phrasings of *query*.

    Returns:
        A list starting with the **original query**, followed by up to
        ``n_variants`` generated alternatives (duplicates / blanks removed).
    """
    prompt = _PROMPT_TEMPLATE.format(n=n_variants, query=query)

    try:
        response = llm.invoke(prompt)
        # LangChain ChatModel returns AIMessage; plain LM returns str
        text = getattr(response, "content", None) or str(response)
    except Exception as e:
        print(f"[MultiQuery] LLM call failed ({e}); falling back to original query only.")
        return [query]

    # Parse lines — skip blanks, numbering prefixes like "1." or "- "
    variants: List[str] = []
    for line in text.strip().splitlines():
        line = line.strip()
        # strip leading numbering / bullets
        for prefix in ("1.", "2.", "3.", "4.", "-", "*"):
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
                break
        if line and line != query:
            variants.append(line)

    # Dedupe while preserving order
    seen = {query}
    unique: List[str] = []
    for v in variants[:n_variants]:
        if v not in seen:
            seen.add(v)
            unique.append(v)

    result = [query] + unique
    print(f"[MultiQuery] {len(result)} queries: {result}")
    return result
