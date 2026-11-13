"""Make a 2-view test input (videos + intrinsics JSON) from one still image, for running the
publisher without cameras:  python make_test_video.py IMAGE OUTDIR [--seconds 20]"""
import argparse, json, os
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("image"); ap.add_argument("outdir"); ap.add_argument("--seconds", type=float, default=20)
a = ap.parse_args()
os.makedirs(a.outdir, exist_ok=True)
img = cv2.imread(a.image)
h, w = 480, 848
s = min(w / img.shape[1], h / img.shape[0])
small = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
canvas = np.zeros((h, w, 3), np.uint8)
y0, x0 = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
canvas[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
for name in ("cam_main", "cam_side"):
    vw = cv2.VideoWriter(f"{a.outdir}/{name}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    for _ in range(int(a.seconds * 30)):
        vw.write(canvas)
    vw.release()
    f = 0.9 * w
    json.dump({"camera_matrix": [[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], "gravity": [0, 1, 0]},
              open(f"{a.outdir}/{name}.json", "w"))
print("wrote", a.outdir)
