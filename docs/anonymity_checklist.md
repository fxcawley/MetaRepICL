# Anonymity and Hygiene Checklist

This document tracks the pre-submission audit to ensure double-blind compliance (e.g., AISTATS, NeurIPS).

## Codebase Hygiene
- [ ] **Remove Author Names**: Ensure no headers contain author names (e.g., "Author: John Doe").
- [ ] **Remove Usernames**: Scan for absolute paths containing user home directories (e.g., `/Users/lcawley/`).
- [ ] **Remove Institution Names**: Ensure no comments or docstrings mention the affiliation.
- [ ] **Remove Git Metadata**: When packaging for supplementary material, remove `.git/` folder.
- [ ] **Sanitize Configs**: Ensure default paths in `configs/` do not point to specific user directories.

## Artifacts and Links
- [ ] **Project URL**: Do not link to the public GitHub repo in the main paper. Use "Anonymous Github" or "Supplementary Material".
- [ ] **Preprint**: If a preprint exists (arXiv), do not cite it as "Ours" or link to it.
- [ ] **W&B / Trackers**: Ensure experiment tracking links (Weights & Biases) are either removed or set to anonymous mode/teams.

## PDF Metadata
- [ ] **PDF Properties**: Strip Author/Creator metadata from the generated PDF (using `exiftool` or Adobe Acrobat).

## Automated Check
Run `python scripts/check_anonymity.py` to scan for common violations.

