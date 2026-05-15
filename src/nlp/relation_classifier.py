from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import PROCESSED_DATA_DIR, TABLES_DIR
from src.nlp.ollama_client import call_ollama
from src.nlp.relation_context import (
    canonicalize_person,
    collect_contexts,
    load_top_edges,
    normalize_email,
    truncate_text,
)
from src.nlp.relation_prompt import RELATION_LABELS, build_prompt


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
    """Small rule-based classifier kept for the existing full_pipeline module."""
    if not text:
        return "other"

    text = text.lower()

    if match_any(text, REQUEST):
        return "request"

    if match_any(text, COORDINATION):
        return "coordination"

    if match_any(text, INFORMATION):
        return "information"

    return "other"


def parse_llm_json(raw_response: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", raw_response, flags=re.DOTALL)
    if not match:
        return parse_label_from_text(raw_response)

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return parse_label_from_text(raw_response)

    relation_type_value = parsed.get("relation_type", "niejednoznaczna")
    if isinstance(relation_type_value, dict):
        relation_type_value = relation_type_value.get("type", "niejednoznaczna")

    relation_type = str(relation_type_value).strip()
    if relation_type not in RELATION_LABELS:
        relation_type = parse_label_from_text(raw_response)["relation_type"]

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))

    evidence = parsed.get("evidence", "")
    if isinstance(evidence, (dict, list)):
        evidence = json.dumps(evidence, ensure_ascii=False)

    return {
        "relation_type": relation_type,
        "confidence": confidence,
        "evidence": truncate_text(evidence, 300),
    }


def parse_label_from_text(raw_response: str) -> dict[str, Any]:
    text = raw_response.lower()
    matches = [label for label in RELATION_LABELS if label in text]

    if len(matches) == 1:
        return {
            "relation_type": matches[0],
            "confidence": 0.35,
            "evidence": "Etykieta odczytana z odpowiedzi tekstowej modelu.",
        }

    return {
        "relation_type": "niejednoznaczna",
        "confidence": 0.0,
        "evidence": "Model nie zwrocil jednoznacznego poprawnego JSON.",
    }


def classify_top_edges(
    edges: pd.DataFrame,
    emails: pd.DataFrame,
    model: str,
    host: str,
    timeout: int,
    max_messages: int,
    body_chars: int,
    dry_run: bool = False,
) -> pd.DataFrame:
    rows = []

    for idx, edge in edges.iterrows():
        source = normalize_email(edge["source"])
        target = normalize_email(edge["target"])
        relation_source = str(edge.get("relation_source", "metadata")).strip() or "metadata"

        if relation_source == "ner":
            source = canonicalize_person(source)
            target = canonicalize_person(target)

        contexts = collect_contexts(
            emails=emails,
            source=source,
            target=target,
            relation_source=relation_source,
            max_messages=max_messages,
            body_chars=body_chars,
        )
        prompt = build_prompt(source, target, edge["weight"], contexts)

        if dry_run:
            parsed = {
                "relation_type": "niejednoznaczna",
                "confidence": 0.0,
                "evidence": "Dry run: prompt prepared, LLM not called.",
            }
            raw_response = ""
        else:
            raw_response = call_ollama(prompt, model=model, host=host, timeout=timeout)
            parsed = parse_llm_json(raw_response)

        rows.append(
            {
                "rank": idx + 1,
                "source": edge["source"],
                "target": edge["target"],
                "weight": edge["weight"],
                "relation_source": relation_source,
                "relation_type_llm": parsed["relation_type"],
                "confidence": parsed["confidence"],
                "evidence": parsed["evidence"],
                "context_messages": len(contexts),
                "model": model,
                "raw_response": raw_response,
            }
        )

        print(
            f"[{idx + 1}/{len(edges)}] {edge['source']} -> {edge['target']}: "
            f"{parsed['relation_type']} ({parsed['confidence']:.2f})",
            flush=True,
        )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify the strongest graph edges with a local LLM through Ollama."
    )
    parser.add_argument(
        "--edges",
        type=Path,
        default=TABLES_DIR / "metadata_edges.csv",
        help="CSV edge list with source, target, weight and relation_source columns.",
    )
    parser.add_argument(
        "--emails",
        type=Path,
        default=PROCESSED_DATA_DIR / "emails_preprocessed.csv",
        help="Preprocessed emails CSV used to collect message context for each edge.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=TABLES_DIR / "llm_relation_labels.csv",
        help="Output CSV with LLM relation labels.",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--model", default="llama3")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-messages", type=int, default=5)
    parser.add_argument("--body-chars", type=int, default=900)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare contexts and output rows without calling the local LLM.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    edges = load_top_edges(args.edges, top_k=args.top_k)
    emails = pd.read_csv(args.emails)

    print(f"Loaded edges: {len(edges)}")
    print(f"Loaded emails: {len(emails)}")
    print(f"Model: {args.model}")

    results = classify_top_edges(
        edges=edges,
        emails=emails,
        model=args.model,
        host=args.ollama_host,
        timeout=args.timeout,
        max_messages=args.max_messages,
        body_chars=args.body_chars,
        dry_run=args.dry_run,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"Saved LLM relation labels to: {args.output}")


if __name__ == "__main__":
    main()
