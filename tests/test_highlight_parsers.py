"""Tests for highlight_parser factory functions."""

import io
import textwrap
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sirius.highlight_parsers import readwise_api_parser, readwise_markdown_parser
from sirius.protocols import Highlight, Highlights


# ---------------------------------------------------------------------------
# readwise_markdown_parser
# ---------------------------------------------------------------------------


def test_markdown_parser_returns_callable():
    assert callable(readwise_markdown_parser())


def test_markdown_parser_returns_list_of_highlights(tmp_path):
    md = tmp_path / "book.md"
    md.write_text(textwrap.dedent("""\
        # Book Title

        > First highlight.
        > Second highlight.
    """))
    parse = readwise_markdown_parser()
    result = parse(str(md))
    assert isinstance(result, list)
    assert all(isinstance(h, Highlight) for h in result)


def test_markdown_parser_highlight_has_none_context(tmp_path):
    md = tmp_path / "book.md"
    md.write_text("> A highlight.\n")
    parse = readwise_markdown_parser()
    result = parse(str(md))
    assert result[0].context is None


def test_markdown_parser_strips_view_highlight_link(tmp_path):
    md = tmp_path / "book.md"
    md.write_text("> Some text. ([View Highlight](https://example.com/h/123))\n")
    parse = readwise_markdown_parser()
    result = parse(str(md))
    assert result[0].text == "Some text."


def test_markdown_parser_strips_image_markdown(tmp_path):
    md = tmp_path / "book.md"
    md.write_text("> ![alt](https://example.com/img.png) Text after image.\n")
    parse = readwise_markdown_parser()
    result = parse(str(md))
    assert result[0].text == "Text after image."


def test_markdown_parser_skips_non_highlight_lines(tmp_path):
    md = tmp_path / "book.md"
    md.write_text(textwrap.dedent("""\
        # Title
        Some prose.
        > Highlight only.
        More prose.
    """))
    parse = readwise_markdown_parser()
    result = parse(str(md))
    assert len(result) == 1
    assert result[0].text == "Highlight only."


def test_markdown_parser_empty_file_returns_empty_list(tmp_path):
    md = tmp_path / "book.md"
    md.write_text("")
    parse = readwise_markdown_parser()
    assert parse(str(md)) == []


# ---------------------------------------------------------------------------
# Helpers for readwise_api_parser tests
# ---------------------------------------------------------------------------


def _make_httpx_response(json_data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


def _make_epub_bytes(chapters: list[str]) -> bytes:
    """Create a minimal valid EPUB zip with the given chapter HTML strings."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("META-INF/container.xml", textwrap.dedent("""\
            <?xml version="1.0"?>
            <container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
              <rootfiles>
                <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
              </rootfiles>
            </container>
        """))
        manifest_items = "\n".join(
            f'    <item id="ch{i}" href="chapter{i}.xhtml" media-type="application/xhtml+xml"/>'
            for i in range(len(chapters))
        )
        spine_items = "\n".join(
            f'    <itemref idref="ch{i}"/>'
            for i in range(len(chapters))
        )
        opf = textwrap.dedent(f"""\
            <?xml version="1.0"?>
            <package xmlns="http://www.idpf.org/2007/opf">
              <manifest>
            {manifest_items}
              </manifest>
              <spine>
            {spine_items}
              </spine>
            </package>
        """)
        zf.writestr("OEBPS/content.opf", opf)
        for i, ch_html in enumerate(chapters):
            zf.writestr(f"OEBPS/chapter{i}.xhtml", ch_html)
    return buf.getvalue()


def _api_parser_with_env(monkeypatch, **kwargs):
    """Create a readwise_api_parser with READWISE_API_KEY set."""
    monkeypatch.setenv("READWISE_API_KEY", "test-key")
    return readwise_api_parser(**kwargs)


# ---------------------------------------------------------------------------
# readwise_api_parser — factory
# ---------------------------------------------------------------------------


def test_api_parser_requires_env_key():
    """KeyError raised if READWISE_API_KEY is not set."""
    import os
    env = {k: v for k, v in os.environ.items() if k != "READWISE_API_KEY"}
    with patch.dict("os.environ", env, clear=True):
        with pytest.raises(KeyError):
            readwise_api_parser()


def test_api_parser_returns_callable(monkeypatch):
    monkeypatch.setenv("READWISE_API_KEY", "test-key")
    assert callable(readwise_api_parser())


# ---------------------------------------------------------------------------
# readwise_api_parser — numeric query resolves to book ID directly
# ---------------------------------------------------------------------------


def test_api_parser_numeric_query_skips_search(monkeypatch):
    parse = _api_parser_with_env(monkeypatch)

    highlights_resp = _make_httpx_response({"results": [{"text": "Hello world."}], "next": None})
    book_meta_resp = _make_httpx_response({
        "id": 42, "title": "My Book",
        "source_url": "https://example.com/article", "category": "article",
    })
    reader_list_resp = _make_httpx_response({
        "results": [{"id": "doc-42", "source_url": "https://example.com/article",
                     "title": "My Book", "category": "article"}],
        "next": None,
    })
    article_resp = _make_httpx_response({"results": [{"html_content": "<p>Hello world.</p>"}]})

    with patch("httpx.get", side_effect=[highlights_resp, book_meta_resp, reader_list_resp, article_resp]) as mock_get:
        result = parse("42")

    # First call should be highlights endpoint with book_id=42, not search
    first_call = mock_get.call_args_list[0]
    assert "highlights" in first_call[0][0]
    assert first_call[1]["params"]["book_id"] == 42


# ---------------------------------------------------------------------------
# readwise_api_parser — string query triggers title search
# ---------------------------------------------------------------------------


def test_api_parser_string_query_searches_by_title(monkeypatch):
    parse = _api_parser_with_env(monkeypatch)

    search_resp = _make_httpx_response({"results": [{"id": 99, "title": "How We Learn"}]})
    highlights_resp = _make_httpx_response({"results": [{"text": "A highlight."}], "next": None})
    book_meta_resp = _make_httpx_response({
        "id": 99, "title": "How We Learn",
        "source_url": "https://example.com/how-we-learn", "category": "article",
    })
    reader_list_resp = _make_httpx_response({
        "results": [{"id": "doc-99", "source_url": "https://example.com/how-we-learn",
                     "title": "How We Learn", "category": "article"}],
        "next": None,
    })
    article_resp = _make_httpx_response({"results": [{"html_content": "<p>A highlight.</p>"}]})

    with patch("httpx.get", side_effect=[search_resp, highlights_resp, book_meta_resp, reader_list_resp, article_resp]):
        result = parse("How We Learn")

    assert len(result) == 1
    assert result[0].text == "A highlight."


def test_api_parser_raises_when_no_book_found(monkeypatch):
    parse = _api_parser_with_env(monkeypatch)

    search_resp = _make_httpx_response({"results": []})

    with patch("httpx.get", return_value=search_resp):
        with pytest.raises(ValueError, match="No Readwise book found"):
            parse("Unknown Title")


# ---------------------------------------------------------------------------
# readwise_api_parser — pagination
# ---------------------------------------------------------------------------


def test_api_parser_follows_pagination(monkeypatch):
    parse = _api_parser_with_env(monkeypatch)

    page1 = _make_httpx_response({
        "results": [{"text": "First."}],
        "next": "https://readwise.io/api/v2/highlights/?page=2",
    })
    page2 = _make_httpx_response({"results": [{"text": "Second."}], "next": None})
    book_meta_resp = _make_httpx_response({
        "id": 42, "title": "My Book",
        "source_url": "https://example.com/my-book", "category": "article",
    })
    reader_list_resp = _make_httpx_response({
        "results": [{"id": "doc-42", "source_url": "https://example.com/my-book",
                     "title": "My Book", "category": "article"}],
        "next": None,
    })
    article_resp = _make_httpx_response({"results": [{"html_content": "<p>First. Second.</p>"}]})

    with patch("httpx.get", side_effect=[page1, page2, book_meta_resp, reader_list_resp, article_resp]):
        result = parse("42")

    assert len(result) == 2
    assert result[0].text == "First."
    assert result[1].text == "Second."


# ---------------------------------------------------------------------------
# readwise_api_parser — context extraction
# ---------------------------------------------------------------------------


def test_api_parser_raises_when_reader_unavailable(monkeypatch):
    parse = _api_parser_with_env(monkeypatch, context_sentences=2)

    highlights_resp = _make_httpx_response({"results": [{"text": "A highlight."}], "next": None})
    book_meta_resp = _make_httpx_response({
        "id": 42, "title": "My Book",
        "source_url": "https://example.com/my-book", "category": "article",
    })
    reader_list_resp = _make_httpx_response({"results": [], "next": None})

    with patch("httpx.get", side_effect=[highlights_resp, book_meta_resp, reader_list_resp]):
        with pytest.raises(ValueError, match="not found in Readwise Reader"):
            parse("42")


def test_api_parser_article_context_extraction(monkeypatch):
    parse = _api_parser_with_env(monkeypatch, context_sentences=1)

    doc_html = "<p>Before sentence.</p><p>The actual highlight text here.</p><p>After sentence.</p>"
    highlights_resp = _make_httpx_response({
        "results": [{"text": "The actual highlight text here."}],
        "next": None,
    })
    book_meta_resp = _make_httpx_response({
        "id": 42, "title": "My Book",
        "source_url": "https://example.com/my-book", "category": "article",
    })
    reader_list_resp = _make_httpx_response({
        "results": [{"id": "doc-42", "source_url": "https://example.com/my-book",
                     "title": "My Book", "category": "article"}],
        "next": None,
    })
    article_resp = _make_httpx_response({"results": [{"html_content": doc_html}]})

    with patch("httpx.get", side_effect=[highlights_resp, book_meta_resp, reader_list_resp, article_resp]):
        result = parse("42")

    assert result[0].context is not None
    assert "Before sentence" in result[0].context or "After sentence" in result[0].context


def test_api_parser_epub_context_extraction(monkeypatch):
    parse = _api_parser_with_env(monkeypatch, context_sentences=1)

    epub_bytes = _make_epub_bytes([
        "<html><body><p>Before sentence.</p><p>The actual highlight text here.</p><p>After sentence.</p></body></html>"
    ])
    highlights_resp = _make_httpx_response({
        "results": [{"text": "The actual highlight text here."}],
        "next": None,
    })
    book_meta_resp = _make_httpx_response({
        "id": 42, "title": "My Epub",
        "source_url": None, "category": "epub",
    })
    reader_list_resp = _make_httpx_response({
        "results": [{"id": "doc-epub-42", "source_url": None,
                     "title": "My Epub", "category": "epub"}],
        "next": None,
    })
    epub_url_resp = _make_httpx_response({"results": [{"raw_source_url": "https://s3.example.com/epub42.epub"}]})
    s3_resp = MagicMock()
    s3_resp.raise_for_status = MagicMock()
    s3_resp.content = epub_bytes

    with patch("httpx.get", side_effect=[highlights_resp, book_meta_resp, reader_list_resp, epub_url_resp, s3_resp]):
        result = parse("42")

    assert result[0].context is not None
    assert "Before sentence" in result[0].context or "After sentence" in result[0].context


def test_api_parser_context_none_when_highlight_not_in_doc(monkeypatch):
    parse = _api_parser_with_env(monkeypatch, context_sentences=2)

    doc_html = "<p>Completely unrelated content. Nothing matches.</p>"
    highlights_resp = _make_httpx_response({
        "results": [{"text": "A highlight not in the document."}],
        "next": None,
    })
    book_meta_resp = _make_httpx_response({
        "id": 42, "title": "My Book",
        "source_url": "https://example.com/my-book", "category": "article",
    })
    reader_list_resp = _make_httpx_response({
        "results": [{"id": "doc-42", "source_url": "https://example.com/my-book",
                     "title": "My Book", "category": "article"}],
        "next": None,
    })
    article_resp = _make_httpx_response({"results": [{"html_content": doc_html}]})

    with patch("httpx.get", side_effect=[highlights_resp, book_meta_resp, reader_list_resp, article_resp]):
        result = parse("42")

    assert result[0].context is None


# ---------------------------------------------------------------------------
# readwise_api_parser — reader document matching
# ---------------------------------------------------------------------------


def test_api_parser_matches_reader_doc_by_source_url(monkeypatch):
    """source_url match takes priority over title match."""
    parse = _api_parser_with_env(monkeypatch)

    highlights_resp = _make_httpx_response({"results": [{"text": "A highlight."}], "next": None})
    book_meta_resp = _make_httpx_response({
        "id": 42, "title": "Ambiguous Title",
        "source_url": "https://example.com/specific-article", "category": "article",
    })
    # Two docs: one with matching title but wrong source_url, one with correct source_url
    reader_list_resp = _make_httpx_response({
        "results": [
            {"id": "wrong-doc", "source_url": "https://example.com/other",
             "title": "Ambiguous Title", "category": "article"},
            {"id": "correct-doc", "source_url": "https://example.com/specific-article",
             "title": "Different Title", "category": "article"},
        ],
        "next": None,
    })
    article_resp = _make_httpx_response({"results": [{"html_content": "<p>A highlight.</p>"}]})

    with patch("httpx.get", side_effect=[highlights_resp, book_meta_resp, reader_list_resp, article_resp]) as mock_get:
        result = parse("42")

    # Check that the article content call used the correct doc id
    article_call = mock_get.call_args_list[3]
    assert article_call[1]["params"]["id"] == "correct-doc"


def test_api_parser_falls_back_to_title_match(monkeypatch):
    """When source_url is None, title match should work."""
    parse = _api_parser_with_env(monkeypatch)

    highlights_resp = _make_httpx_response({"results": [{"text": "A highlight."}], "next": None})
    book_meta_resp = _make_httpx_response({
        "id": 42, "title": "My EPUB Book",
        "source_url": None, "category": "epub",
    })
    reader_list_resp = _make_httpx_response({
        "results": [{"id": "epub-doc", "source_url": None,
                     "title": "My EPUB Book", "category": "epub"}],
        "next": None,
    })
    epub_url_resp = _make_httpx_response({"results": [{"raw_source_url": "https://s3.example.com/book.epub"}]})
    epub_bytes = _make_epub_bytes(["<html><body><p>A highlight.</p></body></html>"])
    s3_resp = MagicMock()
    s3_resp.raise_for_status = MagicMock()
    s3_resp.content = epub_bytes

    with patch("httpx.get", side_effect=[highlights_resp, book_meta_resp, reader_list_resp, epub_url_resp, s3_resp]):
        result = parse("42")

    assert len(result) == 1
    assert result[0].text == "A highlight."


def test_api_parser_epub_parsing(monkeypatch):
    """EPUB multi-chapter text is extracted and used for context."""
    monkeypatch.setenv("READWISE_API_KEY", "test-key")

    epub_bytes = _make_epub_bytes([
        "<html><body><p>Chapter one content.</p></body></html>",
        "<html><body><p>Chapter two content.</p></body></html>",
    ])
    highlights_resp = _make_httpx_response({
        "results": [{"text": "Chapter one content."}],
        "next": None,
    })
    book_meta_resp = _make_httpx_response({
        "id": 1, "title": "Test Book",
        "source_url": None, "category": "epub",
    })
    reader_list_resp = _make_httpx_response({
        "results": [{"id": "epub-1", "source_url": None,
                     "title": "Test Book", "category": "epub"}],
        "next": None,
    })
    epub_url_resp = _make_httpx_response({"results": [{"raw_source_url": "https://s3.example.com/test.epub"}]})
    s3_resp = MagicMock()
    s3_resp.raise_for_status = MagicMock()
    s3_resp.content = epub_bytes

    parse = readwise_api_parser(context_sentences=1)
    with patch("httpx.get", side_effect=[highlights_resp, book_meta_resp, reader_list_resp, epub_url_resp, s3_resp]):
        result = parse("1")

    assert result[0].text == "Chapter one content."
    # With context_sentences=1, context should include the adjacent sentence from ch2
    assert result[0].context is not None
    assert "Chapter two content" in result[0].context
