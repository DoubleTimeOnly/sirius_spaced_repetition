import re
from pathlib import Path

from .protocols import HighlightParserFn, Highlights


def readwise_markdown_parser() -> HighlightParserFn:
    """Return a parser that extracts highlights from a Readwise markdown export."""

    def parse(filepath: str) -> Highlights:
        text = Path(filepath).read_text()
        highlights = []
        for line in text.splitlines():
            if not line.startswith("> "):
                continue
            content = line[2:]
            # Remove ([View Highlight](...)) link
            content = re.sub(r'\s*\(\[View Highlight\]\([^)]+\)\)', '', content)
            # Remove image markdown ![...](...)
            content = re.sub(r'!\[.*?\]\([^)]+\)', '', content)
            content = content.strip()
            if content:
                highlights.append(content)
        return highlights

    return parse
