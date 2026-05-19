from __future__ import annotations

from typing import Any


RELATION_LABELS = [
    "logistics",
    "task",
    "update",
    "discussion",
    "conflict",
]


def build_prompt(source: str, target: str, weight: Any, contexts: list[dict[str, str]]) -> str:
    messages = "\n\n".join(
        (
            f"Email {idx}\n"
            f"Date: {item['date']}\n"
            f"Subject: {item['subject']}\n"
            f"Body: {item['body']}"
        )
        for idx, item in enumerate(contexts, start=1)
    )
    if not messages:
        messages = "No email body available. Classify based on metadata only."

    return f"""You are classifying emails exchanged between two Enron employees.
Read the emails and pick the label that best describes what is happening in them.

PERSON A: {source}
PERSON B: {target}
NUMBER OF EMAILS: {weight}

--- EMAILS ---
{messages}
--- END ---

Pick exactly one label:

LOGISTICS
  The emails are about arranging time or place.
  Scheduling a meeting, confirming a call, sharing availability, or travel plans.
  The content is about WHEN or WHERE, not about the work itself.
  Example: "Are you free Thursday at 2?" or "I'll be in Houston next week."

TASK
  One person asks the other to do something, or delivers something that was requested.
  There is a clear work obligation: a request expecting action, or a delivery of completed work.
  Example: "Can you prepare the gas report by Friday?" or "Here is the analysis you asked for."

UPDATE
  One person sends information to the other without asking for anything in return.
  Status reports, market data, news, forwarded emails, FYI messages.
  The receiver is being informed — no action is expected.
  Example: "FYI — the deal closed this morning." or "Attached is the weekly summary."

DISCUSSION
  Both people exchange ideas, opinions, or analysis.
  Both sides contribute substance — they debate, negotiate, or work through a problem together.
  Example: "I think we should restructure the contract — what's your view?" followed by a substantive reply.

CONFLICT
  There is clear tension, pressure, or disagreement between the two people.
  One person challenges, criticizes, complains, or pushes back against the other.
  Example: "This is unacceptable and needs to be fixed immediately." or a defensive reply to criticism.

How to decide:
- Is it about scheduling time? → LOGISTICS
- Is someone being asked to do work, or delivering work? → TASK
- Is someone just sending information with no ask? → UPDATE
- Are both people exchanging ideas or opinions? → DISCUSSION
- Is there tension or frustration? → CONFLICT

CONFLICT overrides the others — if there is clear tension anywhere in the emails, use CONFLICT regardless of the topic.
If the emails contain both a task and an update, pick whichever is more dominant.

Return only this JSON, nothing else:
{{
  "relation_type": "logistics|task|update|discussion|conflict",
  "confidence": 0.0,
  "evidence": "One sentence: what specific signal led you to this label?"
}}"""
