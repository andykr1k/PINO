"""Plots E004's validation curves (E001, trained on the pre-fix data, as a faint reference).

Reads the live run dir if present, else the committed logs; writes logs/val_curves.png.

    uv run python lab_notebook/experiments/E004_dino_align_fixed_opacity/scripts/plot_val_curves.py
"""
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_DIR = Path(__file__).resolve().parent.parent
REPO = Path(subprocess.check_output(["git", "-C", str(EXP_DIR), "rev-parse", "--show-toplevel"], text=True).strip())
MAIN = Path(
    subprocess.check_output(["git", "-C", str(REPO), "worktree", "list", "--porcelain"], text=True)
    .splitlines()[0]
    .removeprefix("worktree ")
)

# Reference palette (dataviz skill): categorical slots 1-2, neutral text/grid inks.
E004_COLOR, E001_COLOR = "#2a78d6", "#eb6834"
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load_val(run_dir: Path, fallback: Path) -> list[dict]:
    path = run_dir / "metrics.jsonl" if (run_dir / "metrics.jsonl").exists() else fallback
    return [r for r in map(json.loads, path.read_text().splitlines()) if "val/loss" in r]


e004 = load_val(MAIN / "outputs/e004_dino_align_fixed_opacity", EXP_DIR / "logs/metrics.jsonl")
# Raw cosine as logged during training was distorted by unconstrained prediction norms (see README); the
# corrected values come from scripts/revalidate.py, which re-scores every checkpoint.
revalidated_path = EXP_DIR / "logs/metrics_revalidated.jsonl"
revalidated = load_val(Path("/nonexistent"), revalidated_path) if revalidated_path.exists() else []
runs = [
    ("E004 (fixed data)", E004_COLOR, 2.0, 1.0, e004),
    ("E001 (pre-fix data, stopped)", E001_COLOR, 1.25, 0.55,
     load_val(MAIN / "outputs/dino_align_2026-10-27", REPO / "lab_notebook/experiments/E001_dino_align_baseline/logs/metrics.jsonl")),
]
panels = [
    ("val/loss", "Validation loss", "lower is better", None, runs),
    ("val/render/cos_raw", "Cosine to DINO tokens (raw, norm-corrected)", "dashed: predict-the-mean baseline",
     "val/baseline/cos", [runs[0][:4] + (revalidated or e004,)]),
    ("val/render/cos", "Cosine in standardised space", "constant prediction ≈ 0", None, runs),
]

plt.rcParams.update({"font.size": 10, "text.color": TEXT, "axes.labelcolor": TEXT_2,
                     "xtick.color": TEXT_2, "ytick.color": TEXT_2})
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, facecolor=SURFACE)
for ax, (key, title, note, baseline_key, panel_runs) in zip(axes, panels):
    ax.set_facecolor(SURFACE)
    for label, color, width, alpha, rows in panel_runs:
        steps = [r["step"] for r in rows]
        ax.plot(steps, [r[key] for r in rows], color=color, lw=width, alpha=alpha, marker="o", ms=4, label=label)
        if baseline_key:
            ax.plot(steps, [r[baseline_key] for r in rows], color=color, lw=1, alpha=alpha, ls="--")
        if rows and label.startswith("E004"):
            ax.annotate(f"{rows[-1][key]:.3f}", (steps[-1], rows[-1][key]), xytext=(6, 0),
                        textcoords="offset points", va="center", color=TEXT, fontsize=9)
    ax.set_title(title, loc="left", fontsize=11, color=TEXT, pad=18)
    ax.text(0, 1.015, note, transform=ax.transAxes, fontsize=8.5, color=TEXT_2, va="bottom")
    ax.set_xlabel("training step")
    ax.grid(True, color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
axes[0].legend(frameon=False, fontsize=9, labelcolor=TEXT)
fig.suptitle("Splat → DINOv3 alignment on 20 held-out scenes", x=0.01, ha="left", fontsize=12, color=TEXT)
fig.tight_layout()
out = EXP_DIR / "logs/val_curves.png"
fig.savefig(out, dpi=150, facecolor=SURFACE)
print("wrote", out)
