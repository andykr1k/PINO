"""Surveys InteriorGS labels.json: label vocabulary and frequency, and the corner layout of the 3D boxes.

    uv run python lab_notebook/experiments/E006_object_probe/scripts/label_stats.py
"""
import collections
import json

import numpy as np

from pino import INTERIOR_GS_PATH
from pino.data.scene import list_scene_dirs

scenes = list_scene_dirs(INTERIOR_GS_PATH)
instances = collections.Counter()
scenes_with = collections.Counter()
axis_aligned, total, layouts = 0, 0, collections.Counter()
no_box = collections.Counter()
for scene in scenes:
    labels = json.loads((scene / "labels.json").read_text())
    seen = set()
    for obj in labels:
        if "bounding_box" not in obj:
            no_box[obj["label"]] += 1
            continue
        instances[obj["label"]] += 1
        seen.add(obj["label"])
        c = np.array([[p["x"], p["y"], p["z"]] for p in obj["bounding_box"]])
        total += 1
        e1, e2, e3 = c[1] - c[0], c[3] - c[0], c[4] - c[0]
        # Box edges from corner 0 should be mutually orthogonal and c[2] = c0 + e1 + e2, c[6] = c0 + e1 + e2 + e3
        ok = np.allclose(c[2], c[0] + e1 + e2, atol=1e-3) and np.allclose(c[6], c[0] + e1 + e2 + e3, atol=1e-3)
        layouts[ok] += 1
        axis_aligned += int(np.allclose(e1[[1, 2]], 0, atol=1e-4) or np.allclose(e1[[0, 2]], 0, atol=1e-4))
    scenes_with.update(seen)

print(f"{len(scenes)} scenes, {total} boxed instances, {len(instances)} distinct labels")
print("labels without boxes:", dict(no_box.most_common(10)))
print(f"corner layout c2=c0+e1+e2 and c6=c0+e1+e2+e3 holds for {layouts[True]}/{total}")
print(f"first edge axis-aligned for {axis_aligned}/{total}")
print("top labels (instances, scenes):")
for label, n in instances.most_common(60):
    print(f"  {label:<28} {n:>6} {scenes_with[label]:>5}")
