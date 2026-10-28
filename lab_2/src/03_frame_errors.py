"""Кроки 10-11: похибка репроєкції по кадрах і відсів викидів.

Запуск: python src/03_frame_errors.py
Критерій викиду (заданий заздалегідь): похибка кадра > K_OUTLIER * медіана.
Результат: figures/frame_errors.png, results/frame_errors.csv,
           results/inliers.txt (кадри без викидів, для наступних скриптів).
"""
import csv

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (FIGURES_DIR, INLIERS_FILE, RESULTS_DIR, SEED, calibrate,
                    load_points, object_points)

np.random.seed(SEED)
K_OUTLIER = 2.0


def frame_errors(pts, rvecs, tvecs, K, dist):
    """Крок 10. RMS відстані між знайденими й спроєктованими кутами для кожного кадра."""
    objp = object_points()
    errs = []
    for p, rvec, tvec in zip(pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        d2 = np.sum((p.reshape(-1, 2) - proj.reshape(-1, 2)) ** 2, axis=1)
        errs.append(np.sqrt(d2.mean()))
    return np.array(errs)


def main():
    names, methods, pts, size = load_points()
    rms, K, dist, rvecs, tvecs = calibrate(pts, size)
    errs = frame_errors(pts, rvecs, tvecs, K, dist)
    median = np.median(errs)
    threshold = K_OUTLIER * median
    is_out = errs > threshold

    print(f"RMS calibrateCamera: {rms:.3f} px; медіана по кадрах {median:.3f} px; "
          f"поріг {threshold:.3f} px")
    for n, e, o in zip(names, errs, is_out):
        print(f"{n}: {e:.3f} px{'  <-- викид' if o else ''}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(range(len(names)), errs, color=["#c0392b" if o else "#2e86c1" for o in is_out])
    ax.axhline(median, color="k", ls="--", lw=1, label=f"медіана {median:.2f}")
    ax.axhline(threshold, color="#c0392b", ls=":", lw=1.5,
               label=f"поріг {K_OUTLIER:g}×медіана = {threshold:.2f}")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n[4:6] for n in names])
    ax.set_xlabel("номер кадра")
    ax.set_ylabel("похибка репроєкції, px")
    ax.set_title("Похибка репроєкції по кадрах")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "frame_errors.png", dpi=150)

    keep = [i for i in range(len(names)) if not is_out[i]]
    if is_out.any():
        rms2, K2, dist2, _, _ = calibrate([pts[i] for i in keep], size)
        print(f"\nВикиди: {', '.join(n for n, o in zip(names, is_out) if o)}")
        print(f"без викидів: RMS {rms2:.3f} px, fx {K2[0, 0]:.1f}, fy {K2[1, 1]:.1f} "
              f"(з усіма: RMS {rms:.3f}, fx {K[0, 0]:.1f}, fy {K[1, 1]:.1f})")
    else:
        print("\nВикидів немає")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "frame_errors.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "method", "error_px", "outlier"])
        for n, m, e, o in zip(names, methods, errs, is_out):
            w.writerow([n, m, round(float(e), 4), int(o)])
    INLIERS_FILE.write_text("\n".join(names[i] for i in keep) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()