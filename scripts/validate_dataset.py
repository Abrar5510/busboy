# Validate a LeRobot v3 dataset against the failure modes that crashed training:
#   1. orphan rows: data parquet rows not covered by meta/episodes (shifts every later episode)
#   2. frame-index overshoot: any frame whose video lookup lands past the end of its video file
#   3. per-file: sum(episode lengths) == decoder frame count
# Usage: python validate_dataset.py <root>   (exit 1 if anything fails)
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from torchcodec.decoders import VideoDecoder

root = sys.argv[1]
info = json.load(open(f"{root}/meta/info.json"))
fps = info["fps"]
eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/meta/episodes/*/*.parquet"))]).set_index("episode_index")
data = pd.concat([pd.read_parquet(f, columns=["episode_index", "timestamp"]) for f in sorted(glob.glob(f"{root}/data/*/*.parquet"))])
video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
fail = []

rows, meta_frames = len(data), int(eps["length"].sum())
print(f"{root}: {len(eps)} episodes, {rows} data rows, {meta_frames} frames in metadata")
if rows != meta_frames:
    fail.append(f"orphan rows: {rows} data rows vs {meta_frames} in metadata (diff {rows - meta_frames})")
orphans = set(data["episode_index"].unique()) - set(eps.index)
if orphans:
    fail.append(f"episodes present in data but not metadata: {sorted(orphans)}")

nframes = {}
for k in video_keys:
    over_total = 0
    for ep_idx, g in data[data["episode_index"].isin(eps.index)].groupby("episode_index"):
        e = eps.loc[ep_idx]
        p = Path(root) / info["video_path"].format(video_key=k, chunk_index=int(e[f"videos/{k}/chunk_index"]),
                                                   file_index=int(e[f"videos/{k}/file_index"]))
        if p not in nframes:
            md = VideoDecoder(str(p)).metadata
            nframes[p] = (md.num_frames, md.average_fps)
        n, avg = nframes[p]
        idx = np.round((e[f"videos/{k}/from_timestamp"] + g["timestamp"].to_numpy()) * avg).astype(int)
        over_total += int((idx >= n).sum())
    if over_total:
        fail.append(f"{k}: {over_total} frames request an index past the end of their video file")

for k in video_keys:
    for (c, fi), g in eps.groupby([f"videos/{k}/chunk_index", f"videos/{k}/file_index"]):
        p = Path(root) / info["video_path"].format(video_key=k, chunk_index=int(c), file_index=int(fi))
        n = nframes[p][0] if p in nframes else VideoDecoder(str(p)).metadata.num_frames
        if int(g["length"].sum()) != n:
            fail.append(f"{k} file {c}/{fi}: sum(lengths) {int(g['length'].sum())} != decoder frames {n}")

print("FAIL" if fail else "OK")
for f in fail:
    print("  -", f)
sys.exit(1 if fail else 0)
