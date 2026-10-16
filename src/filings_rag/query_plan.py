"""Split questions about several companies into one retrieval per company.

"Compare Apple and Microsoft on supply chain risk" searched as one query returns the top k chunks
overall, and those usually all come from whichever filing matches the wording best: the answer then
covers one company and has nothing to say about the other. Instead, find the companies the question
names (by ticker or by name, matched against the filings in the index), search each company's filings
separately, and interleave the per-company rankings so every company gets a fair share of the k slots.

Matching is deterministic (no LLM call): it uses the ticker and a short form of the company name
EDGAR reports ("BANK OF AMERICA CORP /DE/" -> "bank of america", "JPMORGAN CHASE & CO" -> "jpmorgan
chase" and "jpmorgan").
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

# Words that end a legal name and say nothing about which company it is.
_LEGAL_SUFFIXES = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "companies",
    "group",
    "holdings",
    "holding",
    "ltd",
    "limited",
    "plc",
    "llc",
    "lp",
    "sa",
    "ag",
    "nv",
    "the",
    "com",
}
# First words too common to stand for a company on their own ("bank" of America, "general" Motors).
_GENERIC_FIRST_WORDS = {
    "american",
    "bank",
    "first",
    "general",
    "global",
    "international",
    "national",
    "new",
    "united",
}


def name_aliases(company: str) -> set[str]:
    """Lower-case names a question might use for a company, from its EDGAR name."""
    name = re.sub(r"/[A-Z]{2}/", " ", company)  # state of incorporation: "/DE/"
    words = re.findall(r"[a-z0-9]+", name.lower().replace("&", " "))
    while words and words[-1] in _LEGAL_SUFFIXES:
        words.pop()
    if words and words[0] == "the":
        words = words[1:]
    if not words:
        return set()
    aliases = {" ".join(words)}
    if len(words) > 1 and len(words[0]) >= 4 and words[0] not in _GENERIC_FIRST_WORDS:
        aliases.add(words[0])  # "goldman" for Goldman Sachs, "jpmorgan" for JPMorgan Chase
    return aliases


@dataclass(frozen=True)
class Company:
    ticker: str
    name: str


def mentioned_tickers(question: str, companies: Iterable[Company]) -> list[str]:
    """Tickers of the companies a question names, in the order it first names them.

    A ticker counts only when written in capitals as its own word ("BAC") and at least two letters long,
    so tickers don't match ordinary words; a name counts in any case, with or without a possessive ("Apple's").
    """
    lowered = question.lower()
    first_seen: dict[str, int] = {}
    for c in companies:
        ticker = c.ticker.upper()
        positions = []
        if len(ticker) >= 2:  # a one-letter ticker ("A", "F") would match the start of a sentence
            pattern = rf"(?<![A-Za-z0-9]){re.escape(ticker)}(?![A-Za-z0-9])"
            positions = [m.start() for m in re.finditer(pattern, question)]
        for alias in name_aliases(c.name):
            pattern = r"(?<![a-z0-9])" + r"\s+".join(map(re.escape, alias.split())) + r"(?![a-z0-9])"
            positions += [m.start() for m in re.finditer(pattern, lowered)]
        if positions:
            first_seen[ticker] = min(min(positions), first_seen.get(ticker, len(question)))
    return sorted(first_seen, key=lambda t: first_seen[t])


def split_tickers(question: str, companies: Iterable[Company], filter_tickers: list[str] | None) -> list[str]:
    """The companies to search one at a time, or [] to run a single search.

    - No company filter: split when the question names two or more indexed companies.
    - With a filter: the companies named among the filtered ones decide; if the question names none of
      them (e.g. two companies picked in the UI, then "compare their cybersecurity governance"), split
      across every filtered company. One named company means a normal search within the filter.
    """
    named = mentioned_tickers(question, companies)
    if filter_tickers:
        allowed = list(dict.fromkeys(t.upper() for t in filter_tickers))
        named = [t for t in named if t in allowed]
        if not named:
            named = allowed
    return named if len(named) >= 2 else []


def interleave(rankings: list[list], k: int) -> list:
    """Round-robin merge: the best item of each list, then the second best of each, ... up to k.

    A list that runs out leaves its remaining slots to the others.
    """
    merged: list = []
    for depth in range(max((len(r) for r in rankings), default=0)):
        for ranking in rankings:
            if depth < len(ranking):
                merged.append(ranking[depth])
                if len(merged) == k:
                    return merged
    return merged
