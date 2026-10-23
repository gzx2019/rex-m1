#!/usr/bin/env python3
"""Head-camera thumbnails for the history-frame input.

For every converted episode JSON, decode its ego (head-camera) video once and store uint8 thumbnails
of shape (num_kept, H, W, 3) next to the dataset:

    <data_root>/thumbs/<chunk>/<episode>.ego<RES>s<STRIDE>.npy      # frames 0, STRIDE, 2*STRIDE, ...

JsonDataset reads a history anchor t - history_stride*k at floor(anchor / STRIDE), i.e. at most
STRIDE-1 frames early. Thumbnails are resized with mibot.utils.io.resize_image (factor 32,
max_pixels = RES*RES), the same resize the dataset applies.

Usage (the Q2 recipe uses --res 400 --stride 5):
  python tools/precompute_thumbs.py --data-root /path/to/RoboDojo-all-arx_x5-ee-v2 --res 400 --stride 5 \
      [--chunks chunk-0000,chunk-0001] [--workers 8] [--limit N]
Existing outputs with the expected frame count are skipped.
"""
import argparse, glob, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mibot.utils.io import resize_image


def _resize(img, res):
    from PIL import Image
    return np.asarray(resize_image(Image.fromarray(img), factor=32, max_pixels=res * res))


def process_episode(args):
    json_path, data_root, res, stride = args
    from decord import VideoReader
    with open(json_path) as f:
        traj = json.load(f)
    ego = traj["observations"]["ego"][0]
    vpath, start = ego["path"], int(ego.get("start", 0))
    n = int(traj["num_frames"])
    rel = os.path.relpath(json_path, os.path.join(data_root, "data"))
    out = os.path.join(data_root, "thumbs", rel.replace(".json", f".ego{res}s{stride}.npy"))
    idx = list(range(0, n, stride))
    if os.path.exists(out):
        try:
            arr = np.load(out, mmap_mode="r")
            if arr.shape[0] == len(idx):
                return out, "skip"
        except Exception:
            pass
    vr = VideoReader(vpath, num_threads=2)
    frames = vr.get_batch([start + i for i in idx]).asnumpy()  # (k, H, W, 3)
    bbox = ego.get("crop_bbox")
    if bbox is not None:
        top, left, ch, cw = (int(v) for v in bbox)
        frames = frames[:, top:top + ch, left:left + cw]
    thumbs = np.stack([_resize(fr, res) for fr in frames]).astype(np.uint8)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp.npy"
    np.save(tmp, thumbs); os.replace(tmp, out)
    return out, f"{thumbs.shape}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--res", type=int, default=400); ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--chunks", default=None); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    chunks = a.chunks.split(",") if a.chunks else sorted(os.path.basename(p) for p in glob.glob(os.path.join(a.data_root, "data", "chunk-*")))
    jobs = []
    for c in chunks:
        jobs += sorted(glob.glob(os.path.join(a.data_root, "data", c, "*.json")))
    if a.limit:
        jobs = jobs[: a.limit]
    print(f"{len(jobs)} episodes, res={a.res} stride={a.stride} workers={a.workers}", flush=True)
    t0 = time.time(); done = skipped = failed = 0
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(process_episode, (j, a.data_root, a.res, a.stride)): j for j in jobs}
        for fut in as_completed(futs):
            try:
                out, status = fut.result()
                if status == "skip": skipped += 1
                else: done += 1
            except Exception as e:
                failed += 1; print("FAIL", futs[fut], repr(e)[:200], flush=True)
            if (done + skipped + failed) % 100 == 0:
                print(f"  {done+skipped+failed}/{len(jobs)} done={done} skip={skipped} fail={failed} {time.time()-t0:.0f}s", flush=True)
    print(f"finished: done={done} skip={skipped} fail={failed} in {time.time()-t0:.0f}s", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
