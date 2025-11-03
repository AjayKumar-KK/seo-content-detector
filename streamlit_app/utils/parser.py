import re
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/119.0.0.0 Safari/537.36"
    )
}


def fetch_and_parse(url: str, timeout: int = 10) -> str | None:
    """
    Fetch raw HTML from a URL.

    Returns:
        HTML string on success, or None if the request fails or the
        response status code is not 200.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        if resp.status_code != 200:
            return None
        return resp.text
    except Exception:
        return None


def clean_text(s: str) -> str:
    """
    Normalise whitespace and remove non-breaking spaces.
    """
    if not isinstance(s, str):
        return ""
    s = s.replace("\xa0", " ")
    return re.sub(r"\s+", " ", s).strip()


def extract_text_from_html(html: str) -> tuple[str, str]:
    """
    Extract (title, body_text) from raw HTML using a simple heuristic.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content tags
    for t in soup(["script", "style", "noscript"]):
        t.extract()

    # Title
    title = soup.title.get_text(separator=" ", strip=True) if soup.title else ""

    # Try <main> or <article> as primary content blocks
    candidates = []
    for tag_name in ["main", "article"]:
        tag = soup.find(tag_name)
        if tag:
            candidates.append(tag.get_text(separator=" ", strip=True))

    # Fallback: concatenate all <p> tags or full text
    if not candidates:
        ps = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
        if len(ps) >= 2:
            candidates.append(" ".join(ps))
        else:
            candidates.append(soup.get_text(separator=" ", strip=True))

    body = max(candidates, key=len) if candidates else ""
    return clean_text(title), clean_text(body)
