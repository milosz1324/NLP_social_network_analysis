from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import pandas as pd


def normalize_email(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def canonicalize_person(value: Any) -> str:
    if not isinstance(value, str):
        return ""

    value = value.lower().strip()
    value = value.replace("@enron.com", "")
    value = value.split("@")[0]
    value = value.replace("_", " ").replace(".", " ")
    value = re.sub(r"[^a-z0-9 ]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def split_recipients(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    return [
        normalize_email(part)
        for part in str(value).replace(";", ",").split(",")
        if normalize_email(part)
    ]


def parse_people(value: Any) -> list[str]:
    if isinstance(value, list):
        return [canonicalize_person(item) for item in value if canonicalize_person(item)]

    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = [item.strip() for item in text.split(",")]

    if not isinstance(parsed, (list, tuple, set)):
        return []

    return [canonicalize_person(item) for item in parsed if canonicalize_person(item)]


def truncate_text(text: Any, limit: int) -> str:
    text = "" if pd.isna(text) else str(text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def load_top_edges(edges_path: Path, top_k: int) -> pd.DataFrame:
    edges = pd.read_csv(edges_path)
    required = {"source", "target", "weight"}
    missing = required - set(edges.columns)
    if missing:
        raise ValueError(f"Missing required edge column(s): {', '.join(sorted(missing))}")

    return (
        edges.sort_values(["weight", "source", "target"], ascending=[False, True, True])
        .head(top_k)
        .reset_index(drop=True)
    )


def row_matches_metadata_edge(row: pd.Series, source: str, target: str) -> bool:
    sender = normalize_email(row.get("sender"))
    return sender == source and target in split_recipients(row.get("recipients"))


def row_matches_ner_edge(row: pd.Series, source: str, target: str) -> bool:
    sender = canonicalize_person(row.get("sender"))
    mentions = parse_people(row.get("mentioned_people"))
    return sender == source and target in mentions


def collect_contexts(
    emails: pd.DataFrame,
    source: str,
    target: str,
    relation_source: str,
    max_messages: int,
    body_chars: int,
) -> list[dict[str, str]]:
    if relation_source == "ner":
        mask = emails.apply(lambda row: row_matches_ner_edge(row, source, target), axis=1)
    else:
        mask = emails.apply(lambda row: row_matches_metadata_edge(row, source, target), axis=1)

    matched = emails.loc[mask].head(max_messages)
    contexts = []

    for _, row in matched.iterrows():
        body = row.get("clean_body", row.get("body", ""))
        contexts.append(
            {
                "date": truncate_text(row.get("date_utc", row.get("date", "")), 80),
                "subject": truncate_text(row.get("subject", ""), 160),
                "body": truncate_text(body, body_chars),
            }
        )

    return contexts
