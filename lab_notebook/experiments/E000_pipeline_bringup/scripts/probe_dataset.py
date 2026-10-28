"""Surveys InteriorGS: scene count, PLY header consistency, splat counts, room layout and raw load speed.

    uv run python lab_notebook/experiments/E000_pipeline_bringup/scripts/probe_dataset.py
"""
import collections
import json
import time

import numpy as np

from pino import INTERIOR_GS_PATH

scenes = sorted(p for p in INTERIOR_GS_PATH.iterdir() if p.is_dir())
uncompressed = [p for p in scenes if (p / "3dgs_uncompressed.ply").exists()]
print("scenes", len(scenes), "uncompressed", len(uncompressed))

headers = collections.Counter()
for p in uncompressed[::50]:
    with open(p / "3dgs_uncompressed.ply", "rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            header += f.readline()
    lines = header.decode().splitlines()
    props = tuple(l for l in lines if l.startswith("property"))
    headers[props] += 1
    n = int(next(l for l in lines if l.startswith("element vertex")).split()[-1])
    print(p.name, "splats", n, "props", len(props))
print("distinct headers", len(headers))

structure = json.loads((uncompressed[0] / "structure.json").read_text())
print("rooms", len(structure["rooms"]), "walls", len(structure["walls"]), "holes", len(structure["holes"]))

p = uncompressed[3] / "3dgs_uncompressed.ply"
start = time.time()
with open(p, "rb") as f:
    header = b""
    while not header.endswith(b"end_header\n"):
        header += f.readline()
    lines = header.decode().splitlines()
    n = int(next(l for l in lines if l.startswith("element vertex")).split()[-1])
    num_props = sum(l.startswith("property") for l in lines)
    data = np.fromfile(f, dtype=np.float32, count=n * num_props).reshape(n, num_props)
print(f"np.fromfile load {time.time() - start:.2f}s", data.shape)
