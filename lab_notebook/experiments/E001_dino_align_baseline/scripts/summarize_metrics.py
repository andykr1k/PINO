"""Prints the validation curve of E001 (or any run's metrics.jsonl) as a table.

    uv run python lab_notebook/experiments/E001_dino_align_baseline/scripts/summarize_metrics.py [metrics.jsonl]
"""
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "logs" / "metrics.jsonl"
cols = ["val/render/cos", "val/render/cos_raw", "val/point/cos", "val/point/cos_raw", "val/baseline/cos"]
print("step   " + "  ".join(f"{c.removeprefix('val/'):>15}" for c in cols))
for line in path.read_text().splitlines():
    row = json.loads(line)
    if "val/loss" in row:
        print(f"{row['step']:<6} " + "  ".join(f"{row.get(c, float('nan')):>15.4f}" for c in cols))
