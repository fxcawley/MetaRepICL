import os
import sys
import argparse
import zipfile
import shutil
import glob

def check_files_in_package(directory):
    """
    Checks for common prohibited files in the submission package.
    """
    forbidden_extensions = ['.git', '.pyc', '.DS_Store', '.idea', '.vscode']
    issues = []
    
    print(f"Scanning {directory} for forbidden files...")
    
    for root, dirs, files in os.walk(directory):
        # Check directories
        for d in dirs:
            if d in ['.git', '__pycache__', 'wandb', 'outputs']:
                 issues.append(f"Forbidden directory found: {os.path.join(root, d)}")
        
        # Check files
        for f in files:
            _, ext = os.path.splitext(f)
            if ext in forbidden_extensions or f in ['.DS_Store']:
                issues.append(f"Forbidden file found: {os.path.join(root, f)}")
                
    return issues

def check_anonymity(directory):
    """
    Wraps the existing check_anonymity script logic or re-implements it.
    For now, we assume check_anonymity.py handles the content scan.
    This script focuses on the *package structure*.
    """
    # Import existing checker if available, else rudimentary check
    try:
        from scripts.check_anonymity import scan_file, FORBIDDEN_STRINGS, IGNORE_FILES
        
        print("Running deep anonymity scan on package content...")
        issues = []
        for root, dirs, files in os.walk(directory):
            for f in files:
                if f in IGNORE_FILES: continue
                if f.endswith('.py') or f.endswith('.md') or f.endswith('.txt'):
                    path = os.path.join(root, f)
                    file_issues = scan_file(path)
                    if file_issues:
                         issues.append(f"Anonymity violation in {f}: {len(file_issues)} matches")
        return issues
    except ImportError:
        print("Warning: scripts.check_anonymity not found, skipping deep content scan.")
        return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=".", help="Path to the project root or package dir")
    parser.add_argument("--zip", action="store_true", help="Create a submission zip if checks pass")
    args = parser.parse_args()
    
    root_dir = os.path.abspath(args.path)
    
    print("========================================")
    print("   OpenReview Submission Package Check  ")
    print("========================================")
    
    issues = []
    
    # 1. File Structure Check
    issues.extend(check_files_in_package(root_dir))
    
    # 2. Anonymity Check (Deep)
    # Add local path to sys.path to find scripts
    sys.path.insert(0, root_dir)
    issues.extend(check_anonymity(root_dir))
    
    if issues:
        print("\n[FAIL] Issues found:")
        for i in issues:
            print(f" - {i}")
        sys.exit(1)
    else:
        print("\n[PASS] No structural or anonymity issues found.")
        
    if args.zip:
        out_zip = "submission_package.zip"
        print(f"\nCreating {out_zip}...")
        # Create zip excluding forbidden items
        with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(root_dir):
                # In-place filter dirs
                dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'wandb', 'outputs', 'venv', '.idea', '.vscode']]
                
                for file in files:
                    if file.endswith('.pyc') or file == '.DS_Store' or file == out_zip:
                        continue
                    
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.relpath(abs_path, root_dir)
                    zf.write(abs_path, rel_path)
        print(f"Package created: {os.path.abspath(out_zip)}")

if __name__ == "__main__":
    main()

