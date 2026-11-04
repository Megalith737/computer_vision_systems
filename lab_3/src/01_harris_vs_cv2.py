"""ЛР3, частина A: власний Гарріс проти cv2.cornerHarris.

Обидва відгуки проходять через ОДНЕ Й ТЕ САМЕ NMS (поріг, радіус, N), тож
порівнюється саме формула відгуку, а не різні способи відбору точок.
Збіг: частка власних точок, до яких у точках OpenCV є сусід не далі за
MATCH_EPS px. Орієнтир з методички: > 90 %.

Два варіанти власного відгуку:
  gauss - за завданням (гаусіан σ = 1.5);
  box   - КОНТРОЛЬ: те саме, але з рівним вікном як у cv2.cornerHarris.
cv2.cornerHarris усереднює рівним вікном blockSize x blockSize, а не гаусіаном,
тож збіг «box» ~ 100 % доводить правильність коду, а різниця між «gauss» і
«box» показує, скільки втрачається саме через форму вікна.

Запуск (з кореня lab_3/):  python src/01_harris_vs_cv2.py
Результати: results/harris_vs_cv2.csv, figures/harris_vs_cv2.png
"""
import argparse
import csv

import cv2
import matplotlib
import numpy as np
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (FIGURES_DIR, K, MATCH_EPS, RESULTS_DIR, SEED, image_paths,
                    load_gray)
from harris import harris_response, nms_topn

np.random.seed(SEED)


def cv2_response(gray, block_size, k=K):
    return cv2.cornerHarris(gray.astype(np.float32), block_size, 3, k)


def match_fraction(a, b, eps=MATCH_EPS):
    """Частка точок з a, що мають сусіда з b у радіусі eps."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    dist, _ = cKDTree(b).query(a, k=1)
    return float((dist <= eps).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--block-size", type=int, default=5,
                    help="blockSize для cv2.cornerHarris (вікно усереднення)")
    args = ap.parse_args()

    paths = image_paths()
    if not paths:
        raise SystemExit("У data/prepared/gray/ немає PNG")
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    rows = []
    for p in paths:
        gray = load_gray(p)
        own = nms_topn(harris_response(gray))[:, :2]
        own_box = nms_topn(harris_response(gray, box=args.block_size))[:, :2]
        lib = nms_topn(cv2_response(gray, args.block_size))[:, :2]
        rows.append({
            "file": p.name,
            "n_own": len(own),
            "n_cv2": len(lib),
            "match_gauss": round(match_fraction(own, lib), 4),
            "match_box_control": round(match_fraction(own_box, lib), 4),
        })
        print(f"{p.name:16} own={len(own):4d} cv2={len(lib):4d} "
              f"gauss={rows[-1]['match_gauss']:.1%} "
              f"box(контроль)={rows[-1]['match_box_control']:.1%}")

    out = RESULTS_DIR / "harris_vs_cv2.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    mean_g = np.mean([r["match_gauss"] for r in rows])
    mean_b = np.mean([r["match_box_control"] for r in rows])
    print(f"\nСереднє: gauss {mean_g:.1%} (орієнтир > 90 %), box-контроль "
          f"{mean_b:.1%}; blockSize={args.block_size}, eps={MATCH_EPS:g} px -> {out}")

    # рисунок для першого кадру: обидва набори точок
    gray = load_gray(paths[0])
    own = nms_topn(harris_response(gray))[:, :2]
    lib = nms_topn(cv2_response(gray, args.block_size))[:, :2]
    fig, ax = plt.subplots(figsize=(9, 7), dpi=130)
    ax.imshow(gray, cmap="gray")
    ax.scatter(lib[:, 0], lib[:, 1], s=26, facecolors="none", edgecolors="tab:blue",
               linewidths=0.8, label=f"cv2.cornerHarris ({len(lib)})")
    ax.scatter(own[:, 0], own[:, 1], s=7, c="tab:red", label=f"власний ({len(own)})")
    ax.set_title(f"{paths[0].name}: збіг {rows[0]['match_gauss']:.0%} "
                 f"(eps = {MATCH_EPS:g} px)")
    ax.legend(loc="lower right")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "harris_vs_cv2.png")


if __name__ == "__main__":
    main()