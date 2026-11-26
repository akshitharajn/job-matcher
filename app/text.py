"""Small text helpers shared by retrieval, filters and explanations."""
import re

_TOKEN = re.compile(r"[a-z0-9][a-z0-9+#.\-]*")


def tokenize(s: str) -> list[str]:
    toks = (t.strip(".-") for t in _TOKEN.findall(s.lower()))
    return [t for t in toks if t]


def contains_term(text: str, term: str) -> bool:
    """Whole-word, case-insensitive match (so 'java' does not match 'javascript')."""
    term = term.strip().lower()
    if not term:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])"
    return re.search(pattern, text.lower()) is not None
