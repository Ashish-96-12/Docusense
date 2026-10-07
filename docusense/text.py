"""Small text helpers: tokenizing and sentence splitting, no downloads needed."""
import re
from typing import List

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:['\-][a-z0-9]+)*")

STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further had
    has have having he her here hers herself him himself his how i if in into is it its itself just
    me more most my myself no nor not now of off on once only or other our ours ourselves out over own
    same she should so some such than that the their theirs them themselves then there these they this
    those through to too under until up very was we were what when where which while who whom why will
    with would you your yours yourself yourselves""".split()
)


def tokenize(text: str, drop_stopwords: bool = True) -> List[str]:
    """Lowercase word tokens with punctuation stripped."""
    tokens = _TOKEN_RE.findall(text.lower())
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


# Sentence boundaries. Blank lines and bullet markers count too, so lists
# without periods don't become one giant "sentence".
_SENT_BOUNDARY = re.compile(
    r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9])"  # normal sentence end
    r"|(?<=[a-z0-9)][.!?])(?=[A-Z][a-z])"  # PDFs that drop the space: "standards.Data"
    r"|\n\s*\n+"  # blank line
    r"|\n(?=\s*(?:[-*•]|\d+[.)])\s)"  # bullet item
)


_BULLET_LINE = re.compile(r"^(\s*(?:[-*•]|\d+[.)])\s.*)$", re.MULTILINE)


def split_sentences(text: str) -> List[str]:
    # A bullet item ends at its line break, even without a period
    text = _BULLET_LINE.sub(r"\1\n", text)
    parts = _SENT_BOUNDARY.split(text)
    sentences = []
    for part in parts:
        cleaned = " ".join(part.split())
        cleaned = re.sub(r"^(?:[-*•]|\d+[.)])\s+", "", cleaned)
        if len(cleaned) > 1:
            sentences.append(cleaned)
    return sentences


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r", "")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
