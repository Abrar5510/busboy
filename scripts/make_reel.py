"""Tile eval episode videos into one grid video, e.g. 10 randomized seeds as 2x5, for the demo.

    python -m scripts.make_reel results/videos/smolvla_torch_nominal_train_set_table_ep*.mp4 \
        --out results/videos/reel_smolvla_set_table.mp4

Streams frame by frame (constant memory); finished clips hold their last frame (the outcome banner).
"""

import argparse
import re

import cv2
import imageio.v2 as imageio
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--scale", type=float, default=0.25, help="per-tile scale of the 1536x512 eval frames")
    args = ap.parse_args()

    paths = sorted(args.videos, key=lambda p: int(m.group(1)) if (m := re.search(r"ep(\d+)", p)) else 0)
    readers = [iter(imageio.get_reader(p)) for p in paths]
    tiles = [None] * len(paths)
    blank = None
    writer = imageio.get_writer(args.out, fps=30, macro_block_size=1)
    while True:
        alive = False
        for i, r in enumerate(readers):
            try:
                tiles[i] = cv2.resize(next(r), None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
                alive = True
            except StopIteration:
                pass
        if not alive:
            break
        blank = np.zeros_like(tiles[0]) if blank is None else blank
        cells = tiles + [blank] * (-len(tiles) % args.cols)
        rows = [np.hstack(cells[i:i + args.cols]) for i in range(0, len(cells), args.cols)]
        writer.append_data(np.vstack(rows))
    writer.close()
    print(f"wrote {args.out} from {len(paths)} clips")


if __name__ == "__main__":
    main()
