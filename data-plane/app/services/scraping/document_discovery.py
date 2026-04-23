from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from app.utils.logger import get_logger

log = get_logger(__name__)

DOCUMENT_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "doc",
    ".xlsx": "xlsx",
    ".xls": "xls",
    ".pptx": "pptx",
    ".ppt": "ppt",
    ".odt": "odt",
    ".ods": "ods",
    ".rtf": "rtf",
    ".csv": "csv",
}


class DiscoveredDoc:
    """A document URL found on a page."""

    __slots__ = ("url", "type", "link_text", "found_on")

    def __init__(self, url: str, type: str, link_text: str | None = None, found_on: str | None = None):
        self.url = url
        self.type = type
        self.link_text = link_text
        self.found_on = found_on


def document_type(url: str) -> str | None:
    path = urlparse(url).path.lower()
    for ext, doc_type in DOCUMENT_EXTENSIONS.items():
        if path.endswith(ext):
            return doc_type
    return None


def discover_documents(html: str, base_url: str) -> list[DiscoveredDoc]:
    """Extract document URLs from HTML by extension and normalize to absolute URLs."""
    soup = BeautifulSoup(html or "", "lxml")
    seen: set[str] = set()
    docs: list[DiscoveredDoc] = []

    for a_tag in soup.find_all("a", href=True):
        href = a_tag.get("href")
        if not isinstance(href, str):
            continue
        href = href.strip()
        if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
            continue

        abs_url = urljoin(base_url, href)
        doc_kind = document_type(abs_url)
        if not doc_kind:
            continue
        if abs_url in seen:
            continue

        seen.add(abs_url)
        link_text = a_tag.get_text(strip=True) or None
        docs.append(
            DiscoveredDoc(
                url=abs_url,
                type=doc_kind,
                link_text=link_text,
                found_on=base_url,
            )
        )

    log.debug("documents_discovered", base_url=base_url, count=len(docs))
    return docs


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg", ".tiff", ".tif", ".ico"}


class DiscoveredImage:
    """An image URL found on a page."""

    __slots__ = ("url", "alt", "title")

    def __init__(self, url: str, alt: str | None = None, title: str | None = None):
        self.url = url
        self.alt = alt
        self.title = title


def discover_images(html: str, base_url: str) -> list[DiscoveredImage]:
    """Extract image URLs from HTML <img> tags and normalize to absolute URLs."""
    soup = BeautifulSoup(html or "", "lxml")
    seen: set[str] = set()
    images: list[DiscoveredImage] = []

    for img_tag in soup.find_all("img", src=True):
        src = img_tag.get("src")
        if not isinstance(src, str):
            continue
        src = src.strip()
        if not src or src.startswith("data:"):
            continue

        abs_url = urljoin(base_url, src)
        parsed = urlparse(abs_url)
        if parsed.scheme not in {"http", "https"}:
            continue
        if abs_url in seen:
            continue

        # Only include URLs with a recognizable image extension
        path_lower = parsed.path.lower()
        if not any(path_lower.endswith(ext) for ext in IMAGE_EXTENSIONS):
            continue

        seen.add(abs_url)
        alt = img_tag.get("alt", "").strip() or None
        title = img_tag.get("title", "").strip() or None
        images.append(DiscoveredImage(url=abs_url, alt=alt, title=title))

    log.debug("images_discovered", base_url=base_url, count=len(images))
    return images


def extract_documents_and_links(html: str, base_url: str, found_on: str | None = None) -> tuple[list[DiscoveredDoc], list[str]]:
    """Return both discovered documents and normal links."""
    soup = BeautifulSoup(html or "", "lxml")
    docs = discover_documents(html, base_url)

    doc_urls = {doc.url for doc in docs}
    links: list[str] = []
    seen_links: set[str] = set()

    for a_tag in soup.find_all("a", href=True):
        href = a_tag.get("href")
        if not isinstance(href, str):
            continue
        href = href.strip()
        if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
            continue

        abs_url = urljoin(base_url, href)
        parsed = urlparse(abs_url)
        if parsed.scheme not in {"http", "https"}:
            continue
        if abs_url in doc_urls:
            continue
        if abs_url in seen_links:
            continue

        seen_links.add(abs_url)
        links.append(abs_url)

    if found_on:
        for doc in docs:
            doc.found_on = found_on

    return docs, links


def split_documents_and_links(urls: list[str], found_on: str | None = None) -> tuple[list[DiscoveredDoc], list[str]]:
    """Split a flat list of absolute URLs into documents and non-document page links."""
    docs: list[DiscoveredDoc] = []
    links: list[str] = []
    seen_docs: set[str] = set()
    seen_links: set[str] = set()

    for url in urls:
        if not isinstance(url, str):
            continue
        normalized = url.strip()
        if not normalized:
            continue

        parsed = urlparse(normalized)
        if parsed.scheme not in {"http", "https"}:
            continue

        doc_kind = document_type(normalized)
        if doc_kind:
            if normalized in seen_docs:
                continue
            seen_docs.add(normalized)
            docs.append(
                DiscoveredDoc(
                    url=normalized,
                    type=doc_kind,
                    link_text=None,
                    found_on=found_on,
                )
            )
            continue

        if normalized in seen_links:
            continue
        seen_links.add(normalized)
        links.append(normalized)

    return docs, links
