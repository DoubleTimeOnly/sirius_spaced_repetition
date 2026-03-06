import subprocess

from .protocols import ExtractFn


def passthrough_extractor() -> ExtractFn:
    """Return an extractor that passes highlights through unchanged."""

    def extract(highlight: str, context: str | None = None) -> str:
        return highlight

    return extract


def claude_extractor(model: str = "claude-haiku-4-5-20251001", api_key: str | None = None) -> ExtractFn:
    """Return an extractor that uses Claude to distil the core information from a highlight.

    Args:
        model: Claude model ID to use.
        api_key: Anthropic API key. Defaults to ANTHROPIC_API_KEY env var.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    def extract(highlight: str, context: str | None = None) -> str:
        user_content = highlight
        if context:
            user_content = f"Context:\n{context}\n\nHighlight:\n{highlight}"

        message = client.messages.create(
            model=model,
            max_tokens=512,
            system=(
                "You are a concise knowledge extractor. Given a highlight from a book or article, "
                "return only the core information or concept(s) it contains. "
                "Be brief and precise. Do not add commentary."
            ),
            messages=[{"role": "user", "content": user_content}],
        )
        return message.content[0].text

    return extract


def claude_code_extractor(model: str = "haiku") -> ExtractFn:
    """Return an extractor that uses Claude to distil the core information from a highlight, outputting in code format.

    Args:
        model: Claude model ID to use.
    """
    def extract(highlight: str, context: str | None = None) -> str:
        
        system_prompt = (
            "You are a concise knowledge extractor. Given a highlight from a book or article, "
            "return only the core information or concept(s) it contains. "
            "Be brief and precise. Format the output as a Python dictionary with a single key 'core_info'."
        )
        user_content = highlight
        if context:
            user_content = f"{system_prompt}\n\nContext:\n{context}\n\nHighlight:\n{highlight}"

        command = ["claude", "-p", user_content, "--model", model]
        result = subprocess.run(
            command,
            capture_output=True, 
            text=True
        )
        return result.stdout.strip()

    return extract


def local_llm_extractor(
    model: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    device: str = "auto",
) -> ExtractFn:
    """Return an extractor that uses a local HuggingFace model.

    Downloads the model on first use (cached to ~/.cache/huggingface/hub/).

    Args:
        model: HuggingFace model ID to use.
        device: Device to run the model on. "auto" uses GPU if available,
                otherwise falls back to CPU. Explicit values: "cpu", "cuda",
                "cuda:0", "mps", etc.
    """
    from transformers import pipeline
    from huggingface_hub import login
    login()

    kwargs = {"device_map": "auto"} if device == "auto" else {"device": device}
    pipe = pipeline("text-generation", model=model, **kwargs)

    def extract(highlight: str, context: str | None = None) -> str:
        system_prompt = (
            "You are a concise knowledge extractor. Given a highlight from a book or article, "
            "return only the core information or concept(s) it contains. "
            "Be brief and precise."
        )
        user_content = highlight
        if context:
            user_content = f"Context:\n{context}\n\nHighlight:\n{highlight}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        result = pipe(messages, max_new_tokens=256)
        return result[0]["generated_text"][-1]["content"]

    return extract
