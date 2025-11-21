import os
import sys
import argparse

FORBIDDEN_STRINGS = [
    "lcawley",           # User path
    "fxcawley",          # Github username
    "Author:",           # Metadata
    "Created by",
]

IGNORE_DIRS = [
    ".git",
    "__pycache__",
    "outputs",
    "wandb",
    ".pytest_cache",
    "venv",
    "env"
]

IGNORE_FILES = [
    "check_anonymity.py",
    "anonymity_checklist.md"
]

def scan_file(filepath):
    issues = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                for s in FORBIDDEN_STRINGS:
                    if s in line:
                        issues.append((i + 1, s, line.strip()))
    except Exception as e:
        pass # Binary file or permission issue
    return issues

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".")
    args = parser.parse_args()

    root_dir = os.path.abspath(args.root)
    print(f"Scanning for anonymity violations in {root_dir}...")
    print(f"Forbidden strings: {FORBIDDEN_STRINGS}")

    violation_count = 0
    
    for root, dirs, files in os.walk(root_dir):
        # Filter ignored dirs
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        for file in files:
            if file in IGNORE_FILES:
                continue
                
            filepath = os.path.join(root, file)
            # Skip binary/image extensions roughly
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.pyc', '.git', '.idx', '.pack')):
                continue
                
            issues = scan_file(filepath)
            if issues:
                print(f"\n[FAIL] {os.path.relpath(filepath, root_dir)}")
                for line_num, match, content in issues:
                    print(f"  Line {line_num}: Found '{match}' -> {content[:60]}...")
                violation_count += len(issues)

    if violation_count == 0:
        print("\n[PASS] No anonymity violations found.")
        sys.exit(0)
    else:
        print(f"\n[FAIL] Found {violation_count} potential violations.")
        sys.exit(1)

if __name__ == "__main__":
    main()

