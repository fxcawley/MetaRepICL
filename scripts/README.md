## Scripts

### 1) Export tickets from PROJECT_PLAN.md
```bash
python scripts/export_tickets.py
```
Outputs JSON artifacts in `build/`:
- `tickets.json`
- `labels.json`
- `milestones.json`

### 2) Publish to GitHub (requires gh)
```bash
setx GH_REPO yourowner/yourrepo
# Restart shell if needed so GH_REPO is visible to gh
python scripts/publish_github.py
```

Prereqs:
- Install GitHub CLI (`gh`) and authenticate: `gh auth login`
- Create an empty repo on GitHub and set GH_REPO to `owner/name`

Notes:
- This creates labels, milestones, and issues with labels/milestones attached.
- Classic Projects are not created here; use GitHub Projects (Beta) to organize issues into a board.


