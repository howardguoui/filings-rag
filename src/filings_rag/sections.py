"""Split a 10-K into its standard Items (1, 1A, 7, 7A, 8, ...).

A 10-K mentions each Item at least twice: once in the table of contents and once as
the real section heading (some filers also repeat it as a header on every page). For
every Item we keep the occurrence whose section is longest, which is the real one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

ITEM_TITLES = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity",
    "6": "Reserved",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}

# A heading line: "Item 7." / "ITEM 1A:" / "Item 9A - Controls..." with at most a title after it.
# - (?![0-9a-z]) stops "Item 1" matching the start of "Item 1A" text like "Item 1ABC".
# - [^\n]{0,160}$ keeps cross-references in prose ("Item 7 of this report describes ...")
#   from being treated as headings, since those sit inside long paragraph lines.
_ITEM_RE = re.compile(
    r"^[ \t]*item[ \t\u00a0]+(1a|1b|1c|7a|9a|9b|9c|1[0-6]|[1-9])(?![0-9a-z])[ \t]*[\.:\-\u2014\u2013]?[^\n]{0,160}$",
    re.IGNORECASE | re.MULTILINE,
)

# Some 10-Ks (JPMorgan, Bank of America, ...) are the annual report with a "Form 10-K
# cross-reference index" instead of Item headings: the index rows are table-of-contents
# stubs, so Risk Factors and MD&A never come out as sections. When either is missing
# the split is unreliable and the caller indexes the whole document (see looks_itemized).
CORE_ITEMS = ("1A", "7")
FULL_DOCUMENT = "0"


@dataclass
class Section:
    item: str  # e.g. "7A"
    title: str
    text: str

    @property
    def label(self) -> str:
        return self.title if self.item == FULL_DOCUMENT else f"Item {self.item}. {self.title}"


def split_items(text: str) -> list[Section]:
    matches = list(_ITEM_RE.finditer(text))
    if not matches:
        return [Section(item=FULL_DOCUMENT, title="Full document", text=text.strip())]

    # Consecutive headings for the same Item are one section: some filers (Microsoft) repeat
    # "Item 1A" as a running header on every page, which would otherwise cut the section
    # into page-sized pieces and keep only the longest page.
    runs: list[tuple[str, int]] = []
    for m in matches:
        item = m.group(1).upper()
        if not runs or runs[-1][0] != item:
            runs.append((item, m.start()))

    best: dict[str, tuple[int, int]] = {}
    for i, (item, start) in enumerate(runs):
        end = runs[i + 1][1] if i + 1 < len(runs) else len(text)
        if item not in best or (end - start) > (best[item][1] - best[item][0]):
            best[item] = (start, end)

    sections = []
    for item, (start, end) in sorted(best.items(), key=lambda kv: kv[1][0]):
        body = text[start:end].strip()
        if len(body) < 200:  # table-of-contents stub with no real section behind it
            continue
        sections.append(Section(item=item, title=ITEM_TITLES.get(item, ""), text=body))
    return sections or [Section(item=FULL_DOCUMENT, title="Full document", text=text.strip())]


def looks_itemized(sections: list[Section]) -> bool:
    """True when the split found real Risk Factors and MD&A sections."""
    found = {sec.item for sec in sections}
    return all(item in found for item in CORE_ITEMS)
