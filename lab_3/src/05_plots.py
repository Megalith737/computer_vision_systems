"""ЛР3, частина D: графіки повторюваності з results/repeatability.csv.

Для кожної серії - один рисунок із трьома кривими (Гарріс, SIFT, ORB): середнє по
кадрах, смуга ± σ (розкид між кадрами). Також results/repeatability_summary.csv
зі значеннями для таблиці у звіті.

Запуск (з кореня lab_3/):  python src/05_plots.py
"""
import csv
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import FIGURES_DIR, RESULTS_DIR

TITLES = {
    "rotation": ("Поворот", "Кут повороту, °"),
    "scale": ("Масштаб", "Коефіцієнт масштабу"),
    "brightness": ("Яскравість", "Множник яскравості"),
    "noise": ("Гауссів шум", "σ шуму, рівнів яскравості (0-255)"),
    "blur": ("Розмиття", "σ гауссового розмиття, px"),
    "contrast": ("Контраст", "Множник контрасту"),
}
STYLE = {"Harris": ("tab:blue", "o", "Гарріс (власний)"),
         "SIFT": ("tab:red", "s", "SIFT"),
         "ORB": ("tab:green", "^", "ORB")}


def main():
    path = RESULTS_DIR / "repeatability.csv"
    data = defaultdict(list)  # (series, detector, param) -> [repeatability по кадрах]
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            data[(r["series"], r["detector"], float(r["param"]))].append(
                float(r["repeatability"]))

    FIGURES_DIR.mkdir(exist_ok=True)
    summary = []
    for series in TITLES:
        keys = [k for k in data if k[0] == series]
        if not keys:
            continue
        title, xlabel = TITLES[series]
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
        for det, (color, marker, label) in STYLE.items():
            params = sorted({k[2] for k in keys if k[1] == det})
            if not params:
                continue
            mean = np.array([np.mean(data[(series, det, x)]) for x in params])
            std = np.array([np.std(data[(series, det, x)], ddof=1)
                            if len(data[(series, det, x)]) > 1 else 0.0 for x in params])
            n = [len(data[(series, det, x)]) for x in params]
            ax.plot(params, mean * 100, color=color, marker=marker, lw=1.8, ms=5,
                    label=label)
            ax.fill_between(params, (mean - std) * 100, np.minimum(mean + std, 1) * 100,
                            color=color, alpha=0.15, linewidth=0)
            for x, m, s, k in zip(params, mean, std, n):
                summary.append({"series": series, "detector": det, "param": x,
                                "mean": round(float(m), 4), "std": round(float(s), 4),
                                "n_images": k})
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Повторюваність, %")
        ax.set_ylim(0, 102)
        ax.set_title(f"Повторюваність: {title.lower()} (середнє ± σ по кадрах)")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"repeatability_{series}.png")
        plt.close(fig)
        print(f"figures/repeatability_{series}.png")

    with open(RESULTS_DIR / "repeatability_summary.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=["series", "detector", "param", "mean",
                                           "std", "n_images"])
        wr.writeheader()
        wr.writerows(summary)
    print("results/repeatability_summary.csv")


if __name__ == "__main__":
    main()