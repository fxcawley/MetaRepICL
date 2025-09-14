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


def ensure_labels(labels: List[str], gh_repo: str) -> None:
    for lbl in labels:
        # Create if not exists; ignore errors
        try:
            run(["gh", "label", "create", lbl, "--color", "ededed", "--description", lbl, "--repo", gh_repo])
        except subprocess.CalledProcessError:
            # Might already exist
            pass


def ensure_milestones(milestones: List[Dict[str, Any]], gh_repo: str) -> Dict[str, int]:
    name_to_number: Dict[str, int] = {}
    for ms in milestones:
        title = ms["title"]
        due_on = ms.get("due_on")
        # Try to create (scoped to repo)
        args = ["gh", "api", "-X", "POST", f"repos/{gh_repo}/milestones", "-f", f"title={title}"]
        if due_on:
            args += ["-f", f"due_on={due_on}"]
        try:
            out = subprocess.check_output(args, cwd=ROOT)
            num = json.loads(out.decode("utf-8"))['number']
            name_to_number[title] = num
        except subprocess.CalledProcessError:
            # Fetch existing
            out = subprocess.check_output(["gh", "api", f"repos/{gh_repo}/milestones"], cwd=ROOT)
            existing = json.loads(out.decode("utf-8"))
            for e in existing:
                if e.get("title") == title:
                    name_to_number[title] = e.get("number")
                    break
    return name_to_number


def create_issue(ticket: Dict[str, Any], milestone_number: int | None, gh_repo: str) -> None:
    title = ticket["title"]
    body = ticket.get("body") or ""
    labels = ticket.get("labels", [])
    cmd = ["gh", "issue", "create", "--title", title, "--body", body, "--repo", gh_repo]
    for lbl in labels:
        cmd += ["--label", lbl]
    # gh issue create expects milestone TITLE, not number; use the ticket's milestone label as title
    milestone_title = ticket.get("milestone")
    if milestone_title:
        cmd += ["--milestone", milestone_title]
    run(cmd)


def is_logged_in() -> bool:
    try:
        subprocess.check_call(["gh", "auth", "status"], cwd=ROOT)
        return True
    except subprocess.CalledProcessError:
        return False


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

    if not is_logged_in():
        print("GitHub CLI is not authenticated. Please run: gh auth login -w (choose HTTPS, grant repo scope)", file=sys.stderr)
        sys.exit(1)

    ensure_labels(labels, gh_repo)
    name_to_number = ensure_milestones(milestones, gh_repo)

    # Create issues
    for t in tickets:
        milestone_number = name_to_number.get(t.get("milestone") or "")
        create_issue(t, milestone_number, gh_repo)

    print(f"Published {len(tickets)} issues to {gh_repo}.")


if __name__ == "__main__":
    main()


