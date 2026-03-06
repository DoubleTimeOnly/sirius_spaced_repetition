"""Integration tests for readwise_api_parser — makes real Readwise API calls.

Run with:
    READWISE_API_KEY=<key> uv run pytest tests/test_highlight_parsers_integration.py -v -m integration

Skipped automatically when READWISE_API_KEY is not set.
"""

import os

import httpx
import pytest

from sirius.highlight_parsers import readwise_api_parser
from sirius.protocols import Highlight


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def api_key():
    key = os.environ.get("READWISE_API_KEY")
    if not key:
        pytest.skip("READWISE_API_KEY environment variable not set")
    return key


@pytest.fixture(scope="module")
def headers(api_key):
    return {"Authorization": f"Token {api_key}"}


def _first_book_of_category(headers: dict, category: str) -> dict | None:
    """Return the first v2 book with the given category, or None."""
    resp = httpx.get(
        "https://readwise.io/api/v2/books/",
        headers=headers,
        params={"category": category, "page_size": 10},
    )
    resp.raise_for_status()
    results = resp.json()["results"]
    # Only return books that actually have highlights
    for book in results:
        if book.get("num_highlights", 0) > 0:
            return book
    return None


# ---------------------------------------------------------------------------
# Article integration test
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_readwise_api_parser_article(api_key, headers):
    """Parse a real article from Readwise Reader, verify highlights and context."""
    book = _first_book_of_category(headers, "articles")
    if book is None:
        pytest.skip("No articles with highlights found in Readwise library")

    book_id = book["id"]
    title = book.get("title", "<unknown>")
    parse = readwise_api_parser(context_sentences=3)

    highlights = parse(str(book_id))

    # Basic structure checks
    assert isinstance(highlights, list), "parse() must return a list"
    assert len(highlights) > 0, f"Expected highlights for article {title!r} (id={book_id})"
    assert all(isinstance(h, Highlight) for h in highlights)
    assert all(isinstance(h.text, str) and h.text for h in highlights), \
        "Every highlight must have non-empty text"

    # At least one highlight should have context (document text was fetched successfully)
    highlights_with_context = [h for h in highlights if h.context is not None]
    assert len(highlights_with_context) > 0, (
        f"Expected at least one highlight with context for article {title!r} (id={book_id}). "
        f"Got {len(highlights)} highlights, all with context=None. "
        "This may indicate the document text could not be fetched or the context matching failed."
    )

    # Spot-check: context strings should be non-empty
    for h in highlights_with_context:
        assert h.context.strip(), "context must not be blank"


# ---------------------------------------------------------------------------
# EPUB integration test
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_readwise_api_parser_epub(api_key, headers):
    """Parse a real EPUB from Readwise Reader, verify highlights and context."""
    book = _first_book_of_category(headers, "books")
    if book is None:
        pytest.skip("No books (EPUBs) with highlights found in Readwise library")

    book_id = book["id"]
    title = book.get("title", "<unknown>")
    parse = readwise_api_parser(context_sentences=3)

    highlights = parse(str(book_id))

    # Basic structure checks
    assert isinstance(highlights, list)
    assert len(highlights) > 0, f"Expected highlights for epub {title!r} (id={book_id})"
    assert all(isinstance(h, Highlight) for h in highlights)
    assert all(isinstance(h.text, str) and h.text for h in highlights)

    highlights_with_context = [h for h in highlights if h.context is not None]
    assert len(highlights_with_context) > 0, (
        f"Expected at least one highlight with context for epub {title!r} (id={book_id}). "
        f"Got {len(highlights)} highlights, all with context=None."
    )

    for h in highlights_with_context:
        assert h.context.strip(), "context must not be blank"


# ---------------------------------------------------------------------------
# String query (title search) integration test
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_readwise_api_parser_string_query(api_key, headers):
    """Resolving a book by title string works end-to-end."""
    # Use the first available book title as the search term
    resp = httpx.get(
        "https://readwise.io/api/v2/books/",
        headers=headers,
        params={"page_size": 10},
    )
    resp.raise_for_status()
    books = [b for b in resp.json()["results"] if b.get("num_highlights", 0) > 0]
    if not books:
        pytest.skip("No books with highlights found in Readwise library")

    book = books[0]
    title = book["title"]

    parse = readwise_api_parser(context_sentences=2)
    highlights = parse(title)

    assert isinstance(highlights, list)
    assert len(highlights) > 0, f"Expected highlights when searching by title {title!r}"
    assert all(isinstance(h, Highlight) for h in highlights)


# ---------------------------------------------------------------------------
# Book ID resolution via exact title match integration test
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_resolve_book_id_exact_title_match(api_key, headers):
    """Paginating the v2 books search for 'How We Learn' finds an exact title match."""
    search_title = "How We Learn"

    url = "https://readwise.io/api/v2/books/"
    params: dict = {"search": search_title, "page_size": 100}
    book_id = None

    while url:
        resp = httpx.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()

        for book in data.get("results", []):
            if book.get("title", "").lower() == search_title.lower():
                book_id = book["id"]
                break

        if book_id is not None:
            break

        url = data.get("next")
        params = {}

    assert book_id is not None, (
        f"Expected to find a book titled {search_title!r} (case-insensitive) "
        "via the Readwise v2 books search API, but none was found."
    )

    meta_resp = httpx.get(
        f"https://readwise.io/api/v2/books/{book_id}/",
        headers=headers,
    )
    meta_resp.raise_for_status()
    meta = meta_resp.json()

    assert meta["title"] == search_title
