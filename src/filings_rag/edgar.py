"""Download 10-K filings from SEC EDGAR and turn the HTML into clean text.

EDGAR fair-access rules: at most 10 requests per second, and every request
must carry a User-Agent naming you and a contact email
(https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data).
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
from selectolax.lexbor import LexborHTMLParser as HTMLParser
from selectolax.lexbor import LexborNode as Node

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
DAY = 24 * 3600


@dataclass
class Filing:
    ticker: str
    company: str
    cik: int
    form: str
    accession: str
    filing_date: str
    report_date: str
    fiscal_year: int
    primary_doc: str

    @property
    def url(self) -> str:
        return ARCHIVE_URL.format(cik=self.cik, acc=self.accession.replace("-", ""), doc=self.primary_doc)


class EdgarClient:
    """Small EDGAR client with a polite rate limit (8 requests/second) and an on-disk cache."""

    def __init__(self, user_agent: str, cache_dir: str | Path = "data/raw", max_rps: float = 8.0):
        if not re.search(r"\S+@\S+", user_agent or ""):
            raise ValueError("SEC_USER_AGENT must include a name and contact email, e.g. 'Jane Doe jane@example.com'")
        self.http = httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=30.0,
            follow_redirects=True,
        )
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._min_interval = 1.0 / max_rps
        self._last = 0.0
        self._lock = threading.Lock()

    def _get(self, url: str) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(4):
            with self._lock:  # every attempt, retries included, respects the rate limit
                wait = self._min_interval - (time.monotonic() - self._last)
                if wait > 0:
                    time.sleep(wait)
                self._last = time.monotonic()
            try:
                resp = self.http.get(url)
            except httpx.TransportError as exc:  # timeouts, resets
                last_error = exc
                time.sleep(2**attempt)
                continue
            if resp.status_code in (429, 500, 502, 503, 504):
                retry_after = resp.headers.get("retry-after", "")
                time.sleep(float(retry_after) if retry_after.isdigit() else 2**attempt)
                last_error = httpx.HTTPStatusError(f"{resp.status_code} for {url}", request=resp.request, response=resp)
                continue
            resp.raise_for_status()
            return resp
        assert last_error is not None
        raise last_error

    def _cached_json(self, name: str, url: str, max_age_s: float | None = None) -> dict:
        """Fetch JSON through the disk cache; max_age_s=None means the file never goes stale."""
        path = self.cache_dir / name
        if path.exists() and (max_age_s is None or time.time() - path.stat().st_mtime < max_age_s):
            return json.loads(path.read_text())
        data = self._get(url).json()
        path.write_text(json.dumps(data))
        return data

    def cik_for(self, ticker: str) -> tuple[int, str]:
        table = self._cached_json("company_tickers.json", TICKERS_URL, max_age_s=7 * DAY)
        for row in table.values():
            if row["ticker"].upper() == ticker.upper():
                return int(row["cik_str"]), row["title"]
        raise KeyError(f"Unknown ticker: {ticker}")

    def latest_10ks(self, ticker: str, count: int = 1) -> list[Filing]:
        cik, company = self.cik_for(ticker)
        subs = self._cached_json(f"submissions_{cik}.json", SUBMISSIONS_URL.format(cik=cik), max_age_s=DAY)
        out = filings_from_submissions(subs, ticker.upper(), company, cik, count)
        # "recent" holds the last ~1,000 filings; frequent filers (banks) push older 10-Ks
        # into paginated files listed under filings.files.
        for page in subs["filings"].get("files", []):
            if len(out) >= count:
                break
            older = self._cached_json(page["name"], SUBMISSIONS_PAGE_URL.format(name=page["name"]))
            out += filings_from_submissions(
                {"name": subs.get("name"), "filings": {"recent": older}}, ticker.upper(), company, cik, count - len(out)
            )
        return out

    def filing_text(self, filing: Filing) -> str:
        path = self.cache_dir / f"{filing.ticker}_{filing.accession}.txt"
        if path.exists():
            return path.read_text()
        text = html_to_text(self._get(filing.url).text)
        path.write_text(text)
        (self.cache_dir / f"{filing.ticker}_{filing.accession}.json").write_text(json.dumps(asdict(filing)))
        return text


def filings_from_submissions(subs: dict, ticker: str, company: str, cik: int, count: int) -> list[Filing]:
    """Pick the newest original 10-K filings (not 10-K/A amendments) from a submissions JSON."""
    recent = subs["filings"]["recent"]
    out: list[Filing] = []
    for i, form in enumerate(recent["form"]):
        if form != "10-K":
            continue
        report_date = recent["reportDate"][i] or recent["filingDate"][i]
        out.append(
            Filing(
                ticker=ticker,
                company=subs.get("name") or company,
                cik=cik,
                form=form,
                accession=recent["accessionNumber"][i],
                filing_date=recent["filingDate"][i],
                report_date=report_date,
                fiscal_year=int(report_date[:4]),
                primary_doc=recent["primaryDocument"][i],
            )
        )
        if len(out) >= count:
            break
    return out


_BLOCK_TAGS = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section"}


def _owner(node: Node, tag: str) -> Node | None:
    """Nearest ancestor with this tag."""
    parent = node.parent
    while parent is not None and parent.tag != tag:
        parent = parent.parent
    return parent


def _table_to_text(table: Node) -> str:
    """Render an HTML table as 'cell | cell' rows so numbers keep their labels.

    Only this table's own rows and cells are visited; a nested table's text comes
    through once, inside the cell that holds it.
    """
    rows = []
    for tr in table.css("tr"):
        owner = _owner(tr, "table")
        if owner is None or owner.mem_id != table.mem_id:
            continue
        cells = []
        for td in tr.css("td, th"):
            row = _owner(td, "tr")
            if row is None or row.mem_id != tr.mem_id:
                continue
            t = re.sub(r"\s+", " ", td.text(separator=" ")).strip()
            if t and t not in {"$", ")", "%"}:
                cells.append(t)
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def html_to_text(html: str) -> str:
    """Convert a 10-K HTML (often inline XBRL) document to readable plain text."""
    tree = HTMLParser(html)
    for sel in ("script", "style", "ix\\:header", "head"):
        for node in tree.css(sel):
            node.decompose()
    # Hidden XBRL blocks
    for node in tree.css('[style*="display:none"], [style*="display: none"]'):
        node.decompose()
    for table in tree.css("table"):
        if _owner(table, "table") is not None:
            continue  # nested: rendered as part of its outer table
        text = _table_to_text(table)
        table.replace_with(f"\n{text}\n" if text else "\n")
    for tag in _BLOCK_TAGS - {"table"}:
        for node in tree.css(tag):
            node.insert_after("\n")
    body = tree.body or tree.root
    raw = body.text(separator="") if body else ""
    raw = raw.replace("\xa0", " ").replace("​", "")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in raw.splitlines()]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
