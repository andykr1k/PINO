"""Raw PLY attribute distributions for a few InteriorGS scenes, to check encoding conventions
(opacity as logit vs linear, log vs linear scales, quaternion order, SH layout).

    uv run python lab_notebook/experiments/E003_render_fidelity/scripts/attribute_stats.py [scene ...]
"""
import sys

import numpy as np

from pino import INTERIOR_GS_PATH
from pino.data.scene import PLY_NAME, _read_ply_header

scenes = sys.argv[1:] or ["0001_839920", "0101_839973"]
qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
for name in scenes:
    with open(INTERIOR_GS_PATH / name / PLY_NAME, "rb") as f:
        n, props = _read_ply_header(f)
        data = np.fromfile(f, dtype=np.float32, count=n * len(props)).reshape(n, len(props))
    col = {p: i for i, p in enumerate(props)}
    print(f"=== {name}: {n} splats")
    print("properties:", " ".join(props[:14]), "...", props[-1])
    for key in ["opacity", "scale_0", "rot_0", "rot_1", "f_dc_0", "f_rest_0", "f_rest_15", "f_rest_30"]:
        v = data[:, col[key]]
        print(f"{key:>10}: " + " ".join(f"p{q}={np.percentile(v, q):.3g}" for q in qs))
    op = data[:, col["opacity"]]
    others = np.delete(data, col["opacity"], axis=1)
    print(
        f"opacity +inf={int(np.isposinf(op).sum())} -inf={int(np.isneginf(op).sum())} nan={int(np.isnan(op).sum())}",
        f"finite max={float(op[np.isfinite(op)].max()):.3g};",
        f"rows with non-finite values in other columns={int((~np.isfinite(others)).any(axis=1).sum())}",
    )
    sig = 1 / (1 + np.exp(-op))
    print("fraction sigmoid(opacity) < 0.005:", float((sig < 0.005).mean()), " < 0.05:", float((sig < 0.05).mean()))
    quat = data[:, [col[f"rot_{i}"] for i in range(4)]]
    print("quat norm percentiles:", np.percentile(np.linalg.norm(quat, axis=1), [1, 50, 99]))
    print("|rot_0| largest component fraction:", float((np.abs(quat).argmax(1) == 0).mean()))
