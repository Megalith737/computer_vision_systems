"""Кроки 7-9: калібрування з уточненням кутів cornerSubPix і без нього.

Запуск: python src/02_calibrate.py
Результат: таблиця в консолі і results/calibration_compare.csv.
Кадри з методом SB (кути вже субпіксельні) у порівняння не беруться.
"""
import csv

import numpy as np

from common import RESULTS_DIR, SEED, calibrate, load_detections, refine

np.random.seed(SEED)


def main():
    names, corners, methods, size = load_detections()
    keep = [i for i, m in enumerate(methods) if m != "SB"]
    names = [names[i] for i in keep]
    corners = [corners[i] for i in keep]
    print(f"Кадрів: {len(names)}, розмір {size[0]}x{size[1]}")

    runs = {"без уточнення": corners, "з cornerSubPix": refine(names, corners, size)}
    rows = []
    for label, pts in runs.items():
        rms, K, dist, _, _ = calibrate(pts, size)
        rows.append({"variant": label, "frames": len(pts), "rms_px": round(rms, 4),
                     "fx": round(K[0, 0], 2), "fy": round(K[1, 1], 2),
                     "cx": round(K[0, 2], 2), "cy": round(K[1, 2], 2),
                     **{k: round(v, 5) for k, v in
                        zip(("k1", "k2", "p1", "p2", "k3"), dist.ravel())}})

    print(f"\n{'варіант':16s} {'RMS, px':>8s} {'fx':>8s} {'fy':>8s} {'cx':>7s} {'cy':>7s}")
    for r in rows:
        print(f"{r['variant']:16s} {r['rms_px']:8.3f} {r['fx']:8.1f} {r['fy']:8.1f} "
              f"{r['cx']:7.1f} {r['cy']:7.1f}")
    print(f"\nвідношення RMS без/з cornerSubPix: {rows[0]['rms_px'] / rows[1]['rms_px']:.2f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "calibration_compare.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()