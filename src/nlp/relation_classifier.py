import re


REQUEST = [
    r"\bplease\b",
    r"\bcould you\b",
    r"\bcan you\b",
    r"\bi need\b",
    r"\blet me know\b",
]

COORDINATION = [
    r"\bmeeting\b",
    r"\bschedule\b",
    r"\bcall\b",
    r"\bdiscuss\b",
    r"\btomorrow\b",
]

INFORMATION = [
    r"\bfyi\b",
    r"\battached\b",
    r"\breport\b",
    r"\bdata\b",
    r"\bupdate\b",
]


def match_any(text, patterns):
    return any(re.search(p, text) for p in patterns)


def classify_relation(text: str) -> str:
    if not text:
        return "other"

    text = text.lower()

    # PRIORYTETY (ważne!)
    if match_any(text, REQUEST):
        return "request"

    if match_any(text, COORDINATION):
        return "coordination"

    if match_any(text, INFORMATION):
        return "information"

    return "other"
