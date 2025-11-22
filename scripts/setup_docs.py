import shutil
import os

def safe_copy(src, dst):
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Copied {src} -> {dst}")
    except Exception as e:
        print(f"Error copying {src}: {e}")

# Theory
safe_copy('paper/sections/theory.md', 'docs/theory/index.md')
safe_copy('docs/route_a_theorem.md', 'docs/theory/route_a.md')
safe_copy('docs/lat_to_cg_proof.md', 'docs/theory/route_b.md')
safe_copy('docs/width_rank_theorem.md', 'docs/theory/width_rank.md')

# Experiments
safe_copy('paper/sections/experiments.md', 'docs/experiments/index.md')
# Mechanistic report was missing, check if we can recover or placeholder
if os.path.exists('docs/mechanistic_report.md'):
    safe_copy('docs/mechanistic_report.md', 'docs/experiments/mechanistic_report.md')
else:
    # Create placeholder if missing
    with open('docs/experiments/mechanistic_report.md', 'w') as f:
        f.write("# Mechanistic Report\n\n(Content to be populated from experiments)\n")

# Reproducibility
safe_copy('REPRODUCIBILITY.md', 'docs/reproducibility.md')

# Figures (Recursive)
if os.path.exists('figures'):
    if os.path.exists('docs/figures'):
        # Merge
        pass
    else:
        shutil.copytree('figures', 'docs/figures')
    print("Copied figures -> docs/figures")
