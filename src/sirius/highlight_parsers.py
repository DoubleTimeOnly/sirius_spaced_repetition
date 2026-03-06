import re
from pathlib import Path

from .protocols import Highlight, HighlightParserFn, Highlights


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
                highlights.append(Highlight(text=content, context=None))
        return highlights

    return parse


def readwise_api_parser(context_sentences: int = 4) -> HighlightParserFn:
    """Return a parser that fetches highlights from the Readwise API.

    The ``filepath`` argument (per the HighlightParserFn protocol) is repurposed
    as the book query: a numeric string is treated as a Readwise book ID; any
    other string is used as a title search term.

    Requires the ``READWISE_API_KEY`` environment variable to be set.

    Args:
        context_sentences: Number of surrounding sentences to include as context
            for each highlight (fetched via Readwise Reader v3 if available).
    """
    import html.parser
    import io
    import os
    import time
    import zipfile

    import httpx

    api_key = os.environ["READWISE_API_KEY"]
    headers = {"Authorization": f"Token {api_key}"}

    def _get(url: str, params: dict | None = None) -> httpx.Response:
        """GET with automatic retry on 429, honouring Retry-After."""
        for attempt in range(6):
            resp = httpx.get(url, headers=headers, params=params or {})
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 2 ** attempt))
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        resp.raise_for_status()  # raise on the final attempt
        return resp  # unreachable, satisfies type checker

    def _resolve_book_id(query: str) -> int:
        if query.strip().isdigit():
            return int(query)
        query_lower = query.lower()
        url: str | None = "https://readwise.io/api/v2/books/"
        params: dict = {"search": query}
        while url:
            data = _get(url, params=params).json()
            for result in data["results"]:
                if (result.get("title") or "").lower() == query_lower:
                    return result["id"]
            url, params = data.get("next"), {}
        raise ValueError(f"No Readwise book found for query: {query!r}")

    def _fetch_highlights(book_id: int) -> list[dict]:
        all_highlights: list[dict] = []
        url: str | None = "https://readwise.io/api/v2/highlights/"
        params: dict = {"book_id": book_id, "page_size": 1000}
        while url:
            data = _get(url, params=params).json()
            all_highlights.extend(data["results"])
            url, params = data.get("next"), {}
        return all_highlights

    def _fetch_book_metadata(book_id: int) -> dict:
        """GET /api/v2/books/{book_id}/ → dict with title, source_url, category."""
        return _get(f"https://readwise.io/api/v2/books/{book_id}/").json()

    def _find_reader_document(title: str, source_url: str | None) -> dict:
        """Find a Readwise Reader (v3) document matching the given title/source_url.

        Fast path: if source_url is a Reader-private URL (``private://read/<id>``),
        the path segment IS the v3 document ID — fetch it directly without pagination.

        Otherwise: paginate /api/v3/list/ (cursor-based) matching source_url first
        (exact), then title (case-insensitive) over all pages.
        Raises ValueError if not found.
        """
        # Fast path for EPUBs uploaded to Reader — source_url encodes the doc ID.
        if source_url and source_url.startswith("private://read/"):
            doc_id = source_url.split("/")[-1]
            results = _get(
                "https://readwise.io/api/v3/list/", params={"id": doc_id}
            ).json().get("results", [])
            if results:
                return results[0]

        # General path: paginate v3 list with cursor-based pagination.
        all_docs: list[dict] = []
        cursor: str | None = None
        while True:
            params: dict = {"page_size": 100}
            if cursor:
                params["pageCursor"] = cursor
            data = _get("https://readwise.io/api/v3/list/", params=params).json()
            for doc in data.get("results", []):
                if source_url and doc.get("source_url") == source_url:
                    return doc
                all_docs.append(doc)
            cursor = data.get("nextPageCursor")
            if not cursor:
                break

        # Fallback: title match over all collected docs
        title_lower = (title or "").lower()
        for doc in all_docs:
            if (doc.get("title") or "").lower() == title_lower:
                return doc
        raise ValueError(
            f"Document {title!r} not found in Readwise Reader. "
            "Ensure the document is imported into Reader (articles and epubs are supported)."
        )

    def _strip_html(html_str: str) -> str:
        """Extract plain text from HTML, inserting spaces at block-level tags."""
        BLOCK_TAGS = {"p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"}
        SKIP_TAGS = {"script", "style"}

        class _Parser(html.parser.HTMLParser):
            def __init__(self):
                super().__init__()
                self.parts: list[str] = []
                self._skip_depth = 0

            def handle_starttag(self, tag, attrs):
                if tag in SKIP_TAGS:
                    self._skip_depth += 1
                elif tag in BLOCK_TAGS:
                    self.parts.append(" ")

            def handle_endtag(self, tag):
                if tag in SKIP_TAGS:
                    self._skip_depth = max(0, self._skip_depth - 1)

            def handle_data(self, data):
                if self._skip_depth == 0:
                    self.parts.append(data)

        parser = _Parser()
        parser.feed(html_str)
        return re.sub(r'\s+', ' ', "".join(parser.parts)).strip()

    def _parse_epub(epub_bytes: bytes) -> str:
        """Extract plain text from EPUB bytes using stdlib only."""
        zf = zipfile.ZipFile(io.BytesIO(epub_bytes))

        # Find OPF file via META-INF/container.xml
        container_xml = zf.read("META-INF/container.xml").decode("utf-8", errors="replace")
        m = re.search(r'full-path="([^"]+\.opf)"', container_xml)
        if not m:
            raise ValueError("Could not find OPF path in container.xml")
        opf_path = m.group(1)
        opf_dir = opf_path.rsplit("/", 1)[0] if "/" in opf_path else ""
        opf = zf.read(opf_path).decode("utf-8", errors="replace")

        # Build manifest: id → href (html/xhtml items only)
        manifest: dict[str, str] = {}
        for item_m in re.finditer(r'<item\b([^>]+)/>', opf):
            attrs_str = item_m.group(1)
            id_m = re.search(r'\bid="([^"]+)"', attrs_str)
            href_m = re.search(r'\bhref="([^"]+)"', attrs_str)
            mt_m = re.search(r'\bmedia-type="([^"]+)"', attrs_str)
            if not (id_m and href_m):
                continue
            href = href_m.group(1)
            media_type = mt_m.group(1) if mt_m else ""
            if "html" in media_type or href.endswith((".xhtml", ".html")):
                manifest[id_m.group(1)] = href

        # Spine reading order
        spine_ids = re.findall(r'<itemref\b[^>]+\bidref="([^"]+)"', opf)

        chapters = []
        for item_id in spine_ids:
            href = manifest.get(item_id)
            if not href:
                continue
            chapter_path = f"{opf_dir}/{href}" if opf_dir else href
            try:
                chapter_html = zf.read(chapter_path).decode("utf-8", errors="replace")
                chapters.append(_strip_html(chapter_html))
            except KeyError:
                continue

        return " ".join(chapters)

    def _fetch_article_text(doc_id: str) -> str:
        """GET /api/v3/list/?id={doc_id}&withHtmlContent=true → plain text."""
        results = _get(
            "https://readwise.io/api/v3/list/",
            params={"id": doc_id, "withHtmlContent": "true"},
        ).json().get("results", [])
        if not results:
            raise ValueError(f"Reader document {doc_id!r} not found")
        return _strip_html(results[0]["html_content"])

    def _fetch_epub_text(doc_id: str) -> str:
        """GET /api/v3/list/?id={doc_id}&withRawSourceUrl=true → download + parse EPUB."""
        results = _get(
            "https://readwise.io/api/v3/list/",
            params={"id": doc_id, "withRawSourceUrl": "true"},
        ).json().get("results", [])
        if not results:
            raise ValueError(f"Reader document {doc_id!r} not found")
        raw_url = results[0]["raw_source_url"]
        epub_resp = httpx.get(raw_url)
        epub_resp.raise_for_status()
        return _parse_epub(epub_resp.content)

    def _fetch_document_text(reader_doc: dict) -> str:
        """Dispatch to article or EPUB fetcher based on doc category."""
        doc_id = reader_doc["id"]
        category = reader_doc.get("category", "")
        if category == "epub":
            return _fetch_epub_text(doc_id)
        else:
            return _fetch_article_text(doc_id)

    def _sentences(text: str) -> list[str]:
        return re.split(r'(?<=[.!?])\s+', text)

    def _extract_context(doc_text: str, highlight_text: str, n: int) -> str | None:
        sentences = _sentences(doc_text)
        for i, sent in enumerate(sentences):
            if highlight_text[:60] in sent or sent[:60] in highlight_text:
                before = sentences[max(0, i - n):i]
                after = sentences[i + 1:i + 1 + n]
                return " ".join(before + after) or None
        return None

    def parse(query: str) -> Highlights:
        book_id = _resolve_book_id(query)
        raw = _fetch_highlights(book_id)
        meta = _fetch_book_metadata(book_id)
        reader_doc = _find_reader_document(meta["title"], meta.get("source_url"))
        doc_text = _fetch_document_text(reader_doc)
        return [
            Highlight(
                text=h["text"],
                context=_extract_context(doc_text, h["text"], context_sentences),
            )
            for h in raw
        ]

    return parse
