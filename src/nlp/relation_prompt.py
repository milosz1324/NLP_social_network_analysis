from __future__ import annotations

from typing import Any


RELATION_LABELS = [
    "formal_sluzbowa",
    "wspolpraca",
    "konflikt",
    "delegowanie_zadan",
    "informacyjna",
    "towarzyska",
    "niejednoznaczna",
]


def build_prompt(source: str, target: str, weight: Any, contexts: list[dict[str, str]]) -> str:
    messages = "\n\n".join(
        (
            f"Wiadomosc {idx}\n"
            f"Data: {item['date']}\n"
            f"Temat: {item['subject']}\n"
            f"Tresc: {item['body']}"
        )
        for idx, item in enumerate(contexts, start=1)
    )
    if not messages:
        messages = "Brak tresci wiadomosci dla tej krawedzi. Ocen tylko na podstawie metadanych."

    return f"""Classify one Enron communication graph edge.
Return only one JSON object. Do not use Markdown. Do not explain the task.

Edge: {source} -> {target}
Edge weight: {weight}

Choose exactly one relation_type:
- formal_sluzbowa: professional hierarchy or administrative work relation.
- wspolpraca: people coordinate, discuss, or work together.
- konflikt: disagreement, pressure, criticism, escalation, or negative tone.
- delegowanie_zadan: one person asks, orders, or assigns a task.
- informacyjna: status update, report, attachment, data, FYI, or forwarding information.
- towarzyska: private, social, jokes, sports, events, family, non-work conversation.
- niejednoznaczna: use only when the messages are missing or truly contradictory.

Material dowodowy:
{messages}

Required JSON schema:
{{
  "relation_type": "formal_sluzbowa|wspolpraca|konflikt|delegowanie_zadan|informacyjna|towarzyska|niejednoznaczna",
  "confidence": 0.0,
  "evidence": "short reason in Polish"
}}"""
