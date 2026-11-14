"""ЛР4, частина A: зіставлення SIFT-дескрипторів, відношення Лоу, взаємна перевірка.

Для кожної пари сусідніх кадрів серії (за замовчуванням common.SERIES) і кожного
T з RATIO_GRID (практикум, кроки 1-5):
  - knnMatch (k = 2, норма L2) → кандидати за критерієм Лоу;
  - взаємна перевірка: з кандидатів лишаємо ті, для яких найближчим сусідом
    у зворотному напрямі є той самий дескриптор;
  - «чистота» кандидатів. Еталонних відповідностей немає, тому проксі: частка
    кандидатів, узгоджених із гомографією (бібліотечний RANSAC, τ = MATCH_TAU).
    Це не істинна чистота, а оцінка знизу: правильна відповідність, що не вписалась
    в одну гомографію (паралакс), теж не буде врахована.

Умови однакові для всіх T: ті самі кадри, SIFT_NFEATURES, τ.

Запуск з кореня lab_4:
    python src/02_matching.py --series IMG_8739 IMG_8740 IMG_8741 IMG_8742 IMG_8743
(або заповніть SERIES у common.py і запустіть без аргументів)

Результати:
    results/matching_T.csv          сирі рядки: пара × T (рядок за рядком, можна перервати)
    results/matching_T_summary.csv  середнє ± σ по парах для кожного T
    figures/matching_raw.png        сирі кандидати (T = RAW_DRAW_RATIO), видно хибні лінії
    figures/matching_T.png          кількість кандидатів і частка узгоджених залежно від T
"""
import argparse

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")  # без вікон: лише збереження у файл
import matplotlib.pyplot as plt

from common import (FIGURES_DIR, MATCH_TAU, MAX_DRAW, RATIO_GRID, RAW_DRAW_RATIO,
                    RESULTS_DIR, SEED, SERIES, append_row, done_keys, estimate_h,
                    knn_two, load_frame, matches_to_points, mutual_filter,
                    nearest_index, ratio_filter, sift_features)

np.random.seed(SEED)

CSV = RESULTS_DIR / "matching_T.csv"
SUMMARY = RESULTS_DIR / "matching_T_summary.csv"
FIELDS = ["pair", "T", "n_kp_a", "n_kp_b", "n_ratio", "n_mutual",
          "n_inl_ratio", "n_inl_mutual", "share_ratio", "share_mutual"]


def inlier_count(matches, kp_a, kp_b):
    """Скільки з matches узгоджені з гомографією (RANSAC, τ = MATCH_TAU).
    Якщо пар менше 4 або гомографію не оцінено, повертає None."""
    src, dst = matches_to_points(matches, kp_a, kp_b)
    h, mask = estimate_h(src, dst)
    return None if h is None else int(mask.sum())


def share(n_inl, n):
    """Частка узгоджених від кількості кандидатів (порожньо, якщо оцінки немає)."""
    return "" if (n_inl is None or n == 0) else round(n_inl / n, 4)


def run_pair(a, b):
    """Рахує всі T для пари (a, b) і дописує рядки в CSV; готові (пара, T) пропускає."""
    name = f"{a}_{b}"
    done = done_keys(CSV, ("pair", "T"))
    todo = [t for t in RATIO_GRID if (name, str(t)) not in done]
    if not todo:
        print(f"{name}: уже пораховано")
        return
    kp_a, d_a = sift_features(a)
    kp_b, d_b = sift_features(b)
    knn = knn_two(d_a, d_b)                   # прямий напрям рахуємо один раз
    reverse = nearest_index(d_b, d_a)         # зворотний напрям для взаємної перевірки
    for t in todo:
        cand = ratio_filter(knn, t)
        mut = mutual_filter(cand, reverse)
        n_inl_r = inlier_count(cand, kp_a, kp_b)
        n_inl_m = inlier_count(mut, kp_a, kp_b)
        row = {"pair": name, "T": t, "n_kp_a": len(kp_a), "n_kp_b": len(kp_b),
               "n_ratio": len(cand), "n_mutual": len(mut),
               "n_inl_ratio": "" if n_inl_r is None else n_inl_r,
               "n_inl_mutual": "" if n_inl_m is None else n_inl_m,
               "share_ratio": share(n_inl_r, len(cand)),
               "share_mutual": share(n_inl_m, len(mut))}
        append_row(CSV, FIELDS, row)
        print(f"{name}  T={t:<4}  кандидатів={len(cand):>5}  взаємних={len(mut):>5}  "
              f"частка узгоджених={row['share_ratio'] or '-'}")


def draw_raw(a, b):
    """Рисунок: сирі кандидати (T = RAW_DRAW_RATIO). Показуємо випадкову підмножину
    з MAX_DRAW ліній, інакше нічого не видно. Хибні відповідності – довгі
    «неправильні» лінії, що перетинають решту."""
    kp_a, d_a = sift_features(a)
    kp_b, d_b = sift_features(b)
    cand = ratio_filter(knn_two(d_a, d_b), RAW_DRAW_RATIO)
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(cand), size=min(MAX_DRAW, len(cand)), replace=False)
    shown = [cand[i] for i in pick]
    img = cv2.drawMatches(load_frame(a)[0], kp_a, load_frame(b)[0], kp_b, shown, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # BGR → RGB для matplotlib
    ax.axis("off")
    ax.set_title(f"Сирі кандидати {a} – {b}, T = {RAW_DRAW_RATIO}: показано "
                 f"{len(shown)} із {len(cand)} (випадкова підмножина, seed = {SEED})")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "matching_raw.png", dpi=150)
    plt.close(fig)


def load_rows():
    import csv
    with open(CSV, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def summarize(rows, pairs):
    """Середнє ± σ (ddof = 1, як у ЛР3) по парах для кожного T; пише SUMMARY,
    повертає словник для графіків."""
    metrics = ["n_ratio", "n_mutual", "share_ratio", "share_mutual"]
    out, table = [], {}
    for t in RATIO_GRID:
        sel = [r for r in rows if float(r["T"]) == t and r["pair"] in pairs]
        rec = {"T": t}
        for m in metrics:
            vals = np.array([float(r[m]) for r in sel if r[m] != ""])
            rec[f"{m}_mean"] = round(float(vals.mean()), 4) if len(vals) else ""
            rec[f"{m}_std"] = round(float(vals.std(ddof=1)), 4) if len(vals) > 1 else ""
            rec[f"{m}_n_pairs"] = len(vals)
            table[(m, t)] = vals
        out.append(rec)
    import csv
    with open(SUMMARY, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    return table


def plot_T(table, n_pairs):
    """Два графіки: кількість кандидатів (логарифмічна вісь) і частка узгоджених.
    Великі маркери – середнє ± σ по парах, світлі точки – окремі пари."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    styles = {"ratio": ("тільки відношення Лоу", "o", "C0"),
              "mutual": ("Лоу + взаємна перевірка", "s", "C1")}
    for key, (label, marker, color) in styles.items():
        for ax, metric in ((axes[0], f"n_{key}"), (axes[1], f"share_{key}")):
            means = [table[(metric, t)].mean() if len(table[(metric, t)]) else np.nan
                     for t in RATIO_GRID]
            stds = [table[(metric, t)].std(ddof=1) if len(table[(metric, t)]) > 1 else 0
                    for t in RATIO_GRID]
            for t in RATIO_GRID:  # окремі пари
                v = table[(metric, t)]
                ax.scatter([t] * len(v), v, s=10, color=color, alpha=0.3)
            ax.errorbar(RATIO_GRID, means, yerr=stds, marker=marker, color=color,
                        capsize=3, label=label)
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())  # 2000, а не 2×10³
    axes[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set_ylabel("Кількість кандидатів, шт (логарифмічна шкала)")
    axes[1].set_ylabel(f"Частка узгоджених, % (RANSAC, τ = {MATCH_TAU} px)")
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[1].set_ylim(0, 1.05)
    for ax in axes:
        ax.set_xlabel("Поріг відношення відстаней Лоу T (безрозмірний)")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"Вплив порога Лоу, середнє ± σ по {n_pairs} парах сусідніх кадрів",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "matching_T.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", nargs="+", default=list(SERIES),
                    help="імена кадрів без розширення у порядку зйомки")
    args = ap.parse_args()
    if len(args.series) < 2:
        raise SystemExit("Задайте серію: --series IMG_.. IMG_.. або заповніть SERIES у common.py")

    pairs = [(a, b) for a, b in zip(args.series[:-1], args.series[1:])]
    for a, b in pairs:
        run_pair(a, b)
    draw_raw(*pairs[0])

    names = {f"{a}_{b}" for a, b in pairs}
    table = summarize(load_rows(), names)
    plot_T(table, len(pairs))

    print(f"\n=== Середнє ± σ по {len(pairs)} парах (τ = {MATCH_TAU} px) ===")
    print(f"{'T':>5} {'Лоу, шт':>16} {'+взаємна, шт':>16} {'узгоджені, %':>16} {'узг.+взаємна, %':>16}")
    for t in RATIO_GRID:
        cells = []
        for m, k in (("n_ratio", 1), ("n_mutual", 1), ("share_ratio", 100), ("share_mutual", 100)):
            v = table[(m, t)] * k
            if len(v) == 0:
                cells.append("-")
            else:
                s = v.std(ddof=1) if len(v) > 1 else 0.0
                cells.append(f"{v.mean():.1f} ± {s:.1f}")
        print(f"{t:>5} " + " ".join(f"{c:>16}" for c in cells))
    print(f"\nРисунки: {FIGURES_DIR / 'matching_raw.png'}, {FIGURES_DIR / 'matching_T.png'}")


if __name__ == "__main__":
    main()