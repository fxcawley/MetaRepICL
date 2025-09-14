#!/usr/bin/env python3
"""
Publish tickets to GitHub using the GitHub CLI (gh).

Prereqs:
- `gh auth login` completed
- Environment variables:
  GH_REPO: owner/name (e.g., org/metarep)

Outputs:
- Creates labels, milestones, a classic project (optional), and issues with labels & milestones.

Note: This uses subprocess calls to gh to avoid direct GitHub API tokens in code.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(__file__))
BUILD_DIR = os.path.join(ROOT, "build")


def run(cmd: List[str]) -> None:
    subprocess.check_call(cmd, cwd=ROOT)


def ensure_labels(labels: List[str]) -> None:
    for lbl in labels:
        # Create if not exists; ignore errors
        try:
            run(["gh", "label", "create", lbl, "--color", "ededed", "--description", lbl])
        except subprocess.CalledProcessError:
            pass


def ensure_milestones(milestones: List[Dict[str, Any]]) -> Dict[str, int]:
    name_to_number: Dict[str, int] = {}
    for ms in milestones:
        title = ms["title"]
        due_on = ms.get("due_on")
        # Try to create
        args = ["gh", "api", "-X", "POST", f"repos/${{GH_REPO}}/milestones", "-f", f"title={title}"]
        if due_on:
            args += ["-f", f"due_on={due_on}"]
        try:
            out = subprocess.check_output(args, cwd=ROOT)
            num = json.loads(out.decode("utf-8"))['number']
            name_to_number[title] = num
        except subprocess.CalledProcessError:
            # Fetch existing
            out = subprocess.check_output(["gh", "api", f"repos/${{GH_REPO}}/milestones"], cwd=ROOT)
            existing = json.loads(out.decode("utf-8"))
            for e in existing:
                if e.get("title") == title:
                    name_to_number[title] = e.get("number")
                    break
    return name_to_number


def create_issue(ticket: Dict[str, Any], milestone_number: int | None) -> None:
    title = ticket["title"]
    body = ticket.get("body") or ""
    labels = ticket.get("labels", [])
    cmd = ["gh", "issue", "create", "--title", title, "--body", body]
    for lbl in labels:
        cmd += ["--label", lbl]
    if milestone_number is not None:
        cmd += ["--milestone", str(milestone_number)]
    run(cmd)


def main() -> None:
    gh_repo = os.environ.get("GH_REPO")
    if not gh_repo:
        print("Environment variable GH_REPO must be set to 'owner/name'.", file=sys.stderr)
        sys.exit(1)

    tickets_path = os.path.join(BUILD_DIR, "tickets.json")
    labels_path = os.path.join(BUILD_DIR, "labels.json")
    milestones_path = os.path.join(BUILD_DIR, "milestones.json")

    for p in [tickets_path, labels_path, milestones_path]:
        if not os.path.exists(p):
            print(f"Missing {p}. Run scripts/export_tickets.py first.", file=sys.stderr)
            sys.exit(1)

    tickets = json.load(open(tickets_path, "r", encoding="utf-8"))
    labels = json.load(open(labels_path, "r", encoding="utf-8"))
    milestones = json.load(open(milestones_path, "r", encoding="utf-8"))

    # Scope GH_REPO for gh cli
    os.environ["GH_REPO"] = gh_repo

    ensure_labels(labels)
    name_to_number = ensure_milestones(milestones)

    # Create issues
    for t in tickets:
        milestone_number = name_to_number.get(t.get("milestone") or "")
        create_issue(t, milestone_number)

    print(f"Published {len(tickets)} issues to {gh_repo}.")


if __name__ == "__main__":
    main()


