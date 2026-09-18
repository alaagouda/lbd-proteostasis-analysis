"""
E7 -- All graph axes must be labelled with units.

Since axis labels are set in your plotting scripts, not discoverable from
a rendered image alone, this scans every .py file in a folder for
matplotlib axis-labelling calls and flags:
  (a) any ax.plot/scatter/bar call with no matching set_xlabel/set_ylabel
      anywhere in the same function, and
  (b) any set_xlabel/set_ylabel whose text has no unit-like content
      (no parentheses, no known unit word) -- a label like "Expression"
      or "Score" often needs a unit or normalisation method appended,
      e.g. "Expression (log-normalised)".

This is a heuristic aid to speed up a manual check, not a substitute for
actually looking at each figure -- point it at the folder containing all
your plotting scripts (the ones that generated Figs 1-8 and supplementary
figures) and read the flagged list.
"""
import re
from pathlib import Path

SCRIPT_DIR = Path(".")  # run from the folder containing your plotting scripts

UNIT_HINTS = [
    "(", ")", "log", "score", "rho", "p-value", "p value", "FDR", "NES",
    "%", "count", "AUROC", "AUC", "rank", "Z-score", "expression",
]

LABEL_RE = re.compile(r'(set_[xy]label)\s*\(\s*[\'"]([^\'"]*)[\'"]', re.IGNORECASE)
PLOTCALL_RE = re.compile(r'\.(plot|scatter|bar|barh|hist|boxplot|violinplot)\s*\(')


def check_file(path):
    text = path.read_text(errors="ignore")
    labels = LABEL_RE.findall(text)
    plot_calls = PLOTCALL_RE.findall(text)

    issues = []
    if plot_calls and not labels:
        issues.append(f"{len(plot_calls)} plotting call(s) found but NO set_xlabel/set_ylabel "
                       f"calls anywhere in this file")

    for kind, label_text in labels:
        has_unit_hint = any(hint.lower() in label_text.lower() for hint in UNIT_HINTS)
        if not has_unit_hint and len(label_text.strip()) > 0:
            issues.append(f"{kind}(\"{label_text}\") -- no unit or method hint detected, "
                           f"consider e.g. \"{label_text} (units here)\"")

    return labels, issues


if __name__ == "__main__":
    py_files = sorted(SCRIPT_DIR.glob("*.py"))
    if not py_files:
        print(f"No .py files found in {SCRIPT_DIR.resolve()} -- run this from the folder "
              f"containing your plotting scripts.")
        raise SystemExit

    total_issues = 0
    for f in py_files:
        labels, issues = check_file(f)
        if not labels and not issues:
            continue
        print(f"\n{f.name}")
        for kind, text in labels:
            print(f"    {kind}: \"{text}\"")
        for issue in issues:
            print(f"    FLAG: {issue}")
            total_issues += 1

    print(f"\n{'='*70}")
    print(f"{total_issues} potential issue(s) flagged across {len(py_files)} script(s).")
    print("Remember: this only catches labels with no unit-like text by a crude keyword")
    print("heuristic -- read each flagged label yourself; some (e.g. \"Spearman rho\") are")
    print("genuinely fine without parentheses and will show as false positives.")
