from .protocols import ExtractFn


def passthrough_extractor() -> ExtractFn:
    """Return an extractor that passes highlights through unchanged."""

    def extract(highlight: str, context: str | None = None) -> str:
        return highlight

    return extract


def claude_extractor(model: str = "claude-haiku-4-5-20251001") -> ExtractFn:
    """Return an extractor that uses Claude to distil the core information from a highlight.

    Args:
        model: Claude model ID to use.
    """
    import anthropic

    client = anthropic.Anthropic()

    def extract(highlight: str, context: str | None = None) -> str:
        user_content = highlight
        if context:
            user_content = f"Context:\n{context}\n\nHighlight:\n{highlight}"

        message = client.messages.create(
            model=model,
            max_tokens=512,
            system=(
                "You are a concise knowledge extractor. Given a highlight from a book or article, "
                "return only the core information or concept it contains. "
                "Be brief and precise. Do not add commentary."
            ),
            messages=[{"role": "user", "content": user_content}],
        )
        return message.content[0].text

    return extract
