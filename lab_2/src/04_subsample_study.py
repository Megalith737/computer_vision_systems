"""Крок 12: залежність результату від кількості кадрів N.

Запуск: python src/04_subsample_study.py
Для кожного N по REPEATS випадкових підвибірок (seed 42): калібруємо, записуємо
похибку (у вибірці і на кадрах поза нею) та fx.
Результат: figures/subsample_study.png, results/subsample_study.csv.
"""
import csv

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (FIGURES_DIR, RESULTS_DIR, SEED, calibrate, load_points,
                    object_points)

np.random.seed(SEED)
SIZES = (5, 8, 10, 12, 15, 20)
REPEATS = 5


def heldout_error(pts_list, K, dist):
    """RMS на кадрах, яких не було в калібруванні (поза кадра - через solvePnP)."""
    objp = object_points()
    sq, n = 0.0, 0
    for p in pts_list:
        _, rvec, tvec = cv2.solvePnP(objp, p, K, dist)
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        d2 = np.sum((p.reshape(-1, 2) - proj.reshape(-1, 2)) ** 2, axis=1)
        sq += d2.sum()
        n += len(d2)
    return float(np.sqrt(sq / n))


def main():
    names, _, pts, size = load_points(only_inliers=True)
    idx = list(range(len(names)))
    rows = []
    for n in SIZES:
        if n > len(idx):
            continue
        for rep in range(REPEATS):
            chosen = list(np.random.choice(idx, n, replace=False))
            rest = [i for i in idx if i not in chosen]
            rms, K, dist, _, _ = calibrate([pts[i] for i in chosen], size)
            held = heldout_error([pts[i] for i in rest], K, dist) if rest else np.nan
            rows.append({"N": n, "run": rep + 1, "rms_px": rms, "heldout_px": held,
                         "fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2]})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "subsample_study.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: round(v, 4) for k, v in r.items()} for r in rows)

    Ns = sorted({r["N"] for r in rows})

    def stat(key, n):
        v = np.array([r[key] for r in rows if r["N"] == n], float)
        return np.nanmean(v), np.nanstd(v)

    print(f"{'N':>3s} {'RMS у вибірці':>16s} {'RMS поза вибіркою':>20s} {'fx':>16s}")
    for n in Ns:
        print(f"{n:3d} {stat('rms_px', n)[0]:8.3f} ± {stat('rms_px', n)[1]:5.3f} "
              f"{stat('heldout_px', n)[0]:10.3f} ± {stat('heldout_px', n)[1]:5.3f} "
              f"{stat('fx', n)[0]:9.1f} ± {stat('fx', n)[1]:6.1f}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for key, label, color in (("rms_px", "у вибірці", "#2e86c1"),
                              ("heldout_px", "поза вибіркою", "#c0392b")):
        axes[0].errorbar(Ns, [stat(key, n)[0] for n in Ns], yerr=[stat(key, n)[1] for n in Ns],
                         marker="o", capsize=3, label=label, color=color)
    axes[0].set(xlabel="кількість кадрів N", ylabel="похибка репроєкції, px",
                title="Похибка залежно від N")
    axes[0].legend()
    for n in Ns:
        axes[1].scatter([n] * REPEATS, [r["fx"] for r in rows if r["N"] == n],
                        color="#aab7b8", s=14)
    axes[1].errorbar(Ns, [stat("fx", n)[0] for n in Ns], yerr=[stat("fx", n)[1] for n in Ns],
                     marker="o", capsize=3, color="#2e86c1")
    axes[1].set(xlabel="кількість кадрів N", ylabel="fx, px",
                title="fx залежно від N (точки - окремі повтори)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "subsample_study.png", dpi=150)


if __name__ == "__main__":
    main()