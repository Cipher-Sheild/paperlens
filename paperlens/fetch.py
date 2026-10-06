"""Download a paper from a URL / arXiv id. Includes basic SSRF protection for public deployments."""
import ipaddress
import re
import socket
import tempfile
import urllib.request
from urllib.parse import urljoin, urlparse

MAX_BYTES = 30 * 1024 * 1024
UA = "PaperLens/1.1 (research paper summarizer)"

_ARXIV_NEW = re.compile(r"(?:^|arxiv\.org/(?:abs|pdf|html)/)(\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?$", re.I)
_ARXIV_OLD = re.compile(r"arxiv\.org/(?:abs|pdf)/([a-z\-]+(?:\.[A-Za-z]{2})?/\d{7})(?:v\d+)?(?:\.pdf)?$", re.I)
_META_A = re.compile(r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)', re.I)
_META_B = re.compile(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']citation_pdf_url["\']', re.I)
_DOI = re.compile(r"^(?:doi:\s*|https?://(?:dx\.)?doi\.org/)?(10\.\d{4,9}/\S+)$", re.I)
_HOSTLIKE = re.compile(r"^(?:[a-z0-9-]+\.)+[a-z]{2,}(?::\d+)?(?:/|$)", re.I)
# publishers that normally need a login / block scripted downloads
_PAYWALL_HOSTS = {"ieeexplore.ieee.org": "IEEE Xplore", "www.sciencedirect.com": "ScienceDirect",
                  "dl.acm.org": "ACM Digital Library", "onlinelibrary.wiley.com": "Wiley",
                  "www.tandfonline.com": "Taylor & Francis", "link.springer.com": "Springer",
                  "www.nature.com": "Nature", "www.jstor.org": "JSTOR", "ieeexplore.ieee.org:443": "IEEE Xplore"}
_HREF_PDF = re.compile(r'href=["\']([^"\']+\.pdf(?:\?[^"\']*)?)["\']', re.I)


def normalize_source(s: str) -> str:
    """Turn an arXiv id / abs link / pdf link / plain URL into a downloadable URL."""
    s = (s or "").strip().strip("<>")
    if not s:
        raise ValueError("Please enter a paper link or arXiv id.")
    core = re.sub(r"^arxiv:\s*", "", s, flags=re.I).split("?")[0].split("#")[0].rstrip("/")
    m = _ARXIV_NEW.search(core)
    if m:
        return f"https://arxiv.org/pdf/{m.group(1)}"
    m = _ARXIV_OLD.search(core)
    if m:
        return f"https://arxiv.org/pdf/{m.group(1)}"
    m = _DOI.match(core)
    if m:                                             # DOI such as 10.1109/ABC.2026.123 -> resolver link
        return "https://doi.org/" + m.group(1)
    if not re.match(r"^https?://", s, re.I):
        if _HOSTLIKE.match(s) and " " not in s:
            s = "https://" + s
        else:
            raise ValueError("That doesn't look like a link, DOI or arXiv id (examples: 1706.03762, "
                             "10.1109/..., https://example.org/paper.pdf).")
    return s


_DOI_PUBLISHERS = {"10.1109/": "IEEE Xplore", "10.1016/": "ScienceDirect", "10.1145/": "ACM Digital Library",
                   "10.1002/": "Wiley", "10.1111/": "Wiley", "10.1080/": "Taylor & Francis"}


def paywall_hint(url: str) -> str:
    """Friendly explanation when the publisher site needs a login."""
    host = (urlparse(url).hostname or "").lower()
    path = urlparse(url).path.lstrip("/")
    if host in ("doi.org", "dx.doi.org"):
        for prefix, name in _DOI_PUBLISHERS.items():
            if path.startswith(prefix):
                return (f"{name} papers usually need a login and block automatic downloads. Open the paper in your "
                        "browser, download the PDF, then use the 'Upload PDF' tab.")
    for h, name in _PAYWALL_HOSTS.items():
        if host == h.split(":")[0] or host.endswith("." + h.split(":")[0]):
            return (f"{name} needs a login and blocks automatic downloads. Open the paper in your browser, "
                    "download the PDF, then use the 'Upload PDF' tab.")
    return ""


def _check_public(url: str, allow_private: bool = False):
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise ValueError("Only http(s) links are supported.")
    if allow_private:
        return
    try:
        infos = socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme == "https" else 80))
    except socket.gaierror:
        raise ValueError(f"Could not resolve the host '{p.hostname}'.")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ValueError("Links to private/internal addresses are not allowed.")


class _Guard(urllib.request.HTTPRedirectHandler):
    def __init__(self, allow_private):
        self.allow_private = allow_private

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _check_public(newurl, self.allow_private)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _get(url, allow_private, timeout):
    _check_public(url, allow_private)
    opener = urllib.request.build_opener(_Guard(allow_private))
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/pdf,text/html;q=0.8,*/*;q=0.5"})
    with opener.open(req, timeout=timeout) as r:
        final_url = r.geturl()
        chunks, total = [], 0
        while True:
            b = r.read(1 << 16)
            if not b:
                break
            total += len(b)
            if total > MAX_BYTES:
                raise ValueError("The file is larger than 30 MB.")
            chunks.append(b)
        return b"".join(chunks), final_url


def download_pdf(source: str, allow_private: bool = False, timeout: int = 30) -> str:
    """Return the path of a temporary PDF downloaded from `source`."""
    url = normalize_source(source)
    final = url
    try:
        data, final = _get(url, allow_private, timeout)
        if not data.startswith(b"%PDF"):                      # maybe an HTML landing page that links to the PDF
            page = data.decode("utf-8", "ignore")
            m = _META_A.search(page) or _META_B.search(page) or _HREF_PDF.search(page)
            if not m:
                raise ValueError(paywall_hint(final) or
                                 "The link did not return a PDF. Try a direct PDF link or an arXiv link, "
                                 "or download the PDF and use the 'Upload PDF' tab.")
            data, final = _get(urljoin(final, m.group(1)), allow_private, timeout)
            if not data.startswith(b"%PDF"):
                raise ValueError(paywall_hint(final) or paywall_hint(url) or
                                 "Could not find a downloadable PDF at that link (it may be paywalled). "
                                 "Download it manually and use the 'Upload PDF' tab.")
    except ValueError:
        raise
    except Exception as e:
        hint = paywall_hint(final) or paywall_hint(url)
        raise ValueError(hint or f"Download failed: {str(e)[:120]}. If the paper is behind a login, "
                                 "download the PDF and use the 'Upload PDF' tab.")
    f = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    f.write(data)
    f.close()
    return f.name
