"""ЛР3, частина B: Гарріс не інваріантний до масштабу.

Кадр зменшуємо вдвічі, на обох версіях запускаємо ОДНАКОВИЙ Гарріс (σ, k, поріг,
радіус NMS, N), точки зменшеного кадру переносимо назад (x2) і рахуємо, яка
частка має відповідник у точках оригіналу в межах EPS px.

Контроль: та сама процедура зі scale = 1.0 має дати 100 % (перевіряє сам
код порівняння, а не детектор).

Перенесення координат. Для зменшення вдвічі через INTER_AREA центр пікселя x'
малого кадру відповідає центру (2x' + 0.5) великого, тому x = 2x' + 0.5.
Різниця 0.5 px мала порівняно з EPS = 3 px, але так коректніше, ніж просто x2.

Запуск (з кореня lab_3/):  python src/02_harris_scale.py
Результати: results/harris_scale.csv, figures/harris_scale.png
"""
import csv

import cv2
import matplotlib
import numpy as np
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import FIGURES_DIR, RESULTS_DIR, SEED, image_paths, load_gray
from harris import harris_response, nms_topn

np.random.seed(SEED)

EPS = 3.0     # допуск збігу, px (за методичкою)
SCALE = 0.5   # зменшення вдвічі


def detect(gray):
    return nms_topn(harris_response(gray))[:, :2]


def repeatability(p_ref, p_mapped, eps=EPS):
    """Частка точок, що мають сусіда в іншому наборі в межах eps px.
    Ділимо на min(|P1|, |P2|), як у загальній формулі повторюваності."""
    if len(p_ref) == 0 or len(p_mapped) == 0:
        return 0.0
    dist, _ = cKDTree(p_ref).query(p_mapped, k=1)
    return float((dist <= eps).sum() / min(len(p_ref), len(p_mapped)))


def to_original(points_small, scale):
    """Координати точок зменшеного кадру -> система оригіналу."""
    inv = 1.0 / scale
    return (points_small + 0.5) * inv - 0.5


def main():
    paths = image_paths()
    if not paths:
        raise SystemExit("У data/prepared/gray/ немає PNG")
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    rows = []
    first = None
    for p in paths:
        gray = load_gray(p)
        h, w = gray.shape
        small = cv2.resize(gray, (round(w * SCALE), round(h * SCALE)),
                           interpolation=cv2.INTER_AREA)
        p_orig = detect(gray)
        p_small = detect(small)
        mapped = to_original(p_small, SCALE)

        # відкидаємо точки, що після перенесення випали б за кадр (на практиці 0)
        inside = (mapped[:, 0] >= 0) & (mapped[:, 0] < w) & \
                 (mapped[:, 1] >= 0) & (mapped[:, 1] < h)
        mapped = mapped[inside]

        rep = repeatability(p_orig, mapped)
        control = repeatability(p_orig, to_original(detect(gray), 1.0))
        rows.append({"file": p.name, "n_orig": len(p_orig), "n_small": len(p_small),
                     "repeatability": round(rep, 4), "control_scale1": round(control, 4)})
        print(f"{p.name:16} orig={len(p_orig):4d} small={len(p_small):4d} "
              f"повторюваність={rep:.1%} контроль={control:.0%}")
        if first is None:
            first = (p.name, gray, p_orig, mapped, rep)

    out = RESULTS_DIR / "harris_scale.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    reps = np.array([r["repeatability"] for r in rows])
    print(f"\nСередня повторюваність при масштабі {SCALE}: {reps.mean():.1%} "
          f"± {reps.std(ddof=1):.1%} (n = {len(reps)}), EPS = {EPS:g} px -> {out}")

    name, gray, p_orig, mapped, rep = first
    fig, ax = plt.subplots(figsize=(9, 7), dpi=130)
    ax.imshow(gray, cmap="gray")
    ax.scatter(p_orig[:, 0], p_orig[:, 1], s=26, facecolors="none",
               edgecolors="tab:blue", linewidths=0.8, label=f"оригінал ({len(p_orig)})")
    ax.scatter(mapped[:, 0], mapped[:, 1], s=7, c="tab:red",
               label=f"зменшений ×{SCALE:g}, перенесено ({len(mapped)})")
    ax.set_title(f"{name}: повторюваність {rep:.0%} (EPS = {EPS:g} px)")
    ax.legend(loc="lower right")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "harris_scale.png")


if __name__ == "__main__":
    main()