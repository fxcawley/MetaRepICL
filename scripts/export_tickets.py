#!/usr/bin/env python3
"""
Parse PROJECT_PLAN.md tickets and export structured JSON artifacts:
- build/tickets.json
- build/labels.json
- build/milestones.json

No external deps; relies on stdlib only.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any


PLAN_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "PROJECT_PLAN.md")
BUILD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "build")


TICKET_HEADER_RE = re.compile(r"^### Ticket: (?P<title>.+)$", re.MULTILINE)

FIELD_PATS = {
    "Description": re.compile(r"^- \*\*Description\*\*: (?P<val>.+)$", re.MULTILINE),
    "Acceptance Criteria": re.compile(r"^- \*\*Acceptance Criteria\*\*: (?P<val>.+)$", re.MULTILINE),
    "Deliverables": re.compile(r"^- \*\*Deliverables\*\*: (?P<val>.+)$", re.MULTILINE),
    "Dependencies": re.compile(r"^- \*\*Dependencies\*\*: (?P<val>.+)$", re.MULTILINE),
    "Estimate": re.compile(r"^- \*\*Estimate\*\*: (?P<val>.+)$", re.MULTILINE),
    "Owner": re.compile(r"^- \*\*Owner\*\*: (?P<val>.+)$", re.MULTILINE),
    "Labels": re.compile(r"^- \*\*Labels\*\*: (?P<val>.+)$", re.MULTILINE),
}

MILESTONES = [
    {"title": "M1", "due_on": "2025-09-18T23:59:59Z"},
    {"title": "M2", "due_on": "2025-09-22T23:59:59Z"},
    {"title": "M3", "due_on": "2025-10-10T23:59:59Z"},
    {"title": "M4", "due_on": "2025-10-31T23:59:59Z"},
    {"title": "M5", "due_on": "2025-11-18T23:59:59Z"},
    {"title": "M6", "due_on": "2025-12-12T23:59:59Z"},
]


@dataclass
class Ticket:
    title: str
    description: str
    acceptance_criteria: str
    deliverables: str
    dependencies: str
    estimate: str
    owner: str
    labels: List[str]
    milestone: str | None
    body: str


def extract_field(pattern: re.Pattern, block: str) -> str:
    m = pattern.search(block)
    return m.group("val").strip() if m else ""


def parse_tickets(md: str) -> List[Ticket]:
    tickets: List[Ticket] = []
    headers = list(TICKET_HEADER_RE.finditer(md))
    for idx, h in enumerate(headers):
        start = h.end()
        end = headers[idx + 1].start() if idx + 1 < len(headers) else len(md)
        block = md[start:end]
        title = h.group("title").strip()

        desc = extract_field(FIELD_PATS["Description"], block)
        ac = extract_field(FIELD_PATS["Acceptance Criteria"], block)
        deliv = extract_field(FIELD_PATS["Deliverables"], block)
        deps = extract_field(FIELD_PATS["Dependencies"], block)
        est = extract_field(FIELD_PATS["Estimate"], block)
        owner = extract_field(FIELD_PATS["Owner"], block)
        labels_line = extract_field(FIELD_PATS["Labels"], block)

        def clean_label(s: str) -> str:
            s = s.strip()
            # remove trailing punctuation
            while s and s[-1] in ".,;:":
                s = s[:-1]
                s = s.strip()
            return s

        labels = [clean_label(s) for s in labels_line.split(",") if s.strip()] if labels_line else []
        milestone = None
        for m in ["M1", "M2", "M3", "M4", "M5", "M6"]:
            if m in labels:
                milestone = m
                break

        body_lines = [
            f"Description:\n{desc}",
            f"\nAcceptance Criteria:\n{ac}",
            f"\nDeliverables:\n{deliv}",
            f"\nDependencies:\n{deps}",
            f"\nEstimate: {est}",
            f"\nOwner: {owner}",
        ]
        body = "\n".join(body_lines).strip() + "\n"

        tickets.append(
            Ticket(
                title=title,
                description=desc,
                acceptance_criteria=ac,
                deliverables=deliv,
                dependencies=deps,
                estimate=est,
                owner=owner,
                labels=labels,
                milestone=milestone,
                body=body,
            )
        )
    return tickets


def main() -> None:
    if not os.path.exists(PLAN_PATH):
        raise FileNotFoundError(f"PROJECT_PLAN.md not found at {PLAN_PATH}")
    with open(PLAN_PATH, "r", encoding="utf-8") as f:
        md = f.read()

    tickets = parse_tickets(md)
    all_labels = sorted({lbl for t in tickets for lbl in t.labels})

    os.makedirs(BUILD_DIR, exist_ok=True)
    with open(os.path.join(BUILD_DIR, "tickets.json"), "w", encoding="utf-8") as f:
        json.dump([asdict(t) for t in tickets], f, indent=2)
    with open(os.path.join(BUILD_DIR, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(all_labels, f, indent=2)
    with open(os.path.join(BUILD_DIR, "milestones.json"), "w", encoding="utf-8") as f:
        json.dump(MILESTONES, f, indent=2)

    print(f"Exported {len(tickets)} tickets, {len(all_labels)} labels, {len(MILESTONES)} milestones to 'build/'.")


if __name__ == "__main__":
    main()


