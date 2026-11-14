"""ЛР4, частина B: чому потрібен робастний метод.

Для кожної пари сусідніх кадрів серії (common.SERIES) беремо робочий набір
кандидатів (відношення Лоу T = WORK_RATIO + взаємна перевірка) і оцінюємо
гомографію двома способами на ТИХ САМИХ точках:
  - МНК: cv2.findHomography(src, dst, 0), усі кандидати, без відсіву викидів;
  - RANSAC (бібліотечний, τ = MATCH_TAU), як еталон «чистої» гомографії.

Метрика: похибка репроєкції, px. Рахуємо двома способами:
  E_all – по всіх кандидатах (викиди теж рахуються, тому велика для обох H);
  E_inl – лише по точках, узгоджених із RANSAC-гомографією (маска RANSAC): це
          чесна міра того, наскільки H «потрапляє» у правильні відповідності.
Для МНК важлива саме E_inl: вона показує, що викиди зіпсували оцінку.

Контроль: кадр, зіставлений сам із собою, дає H = I, нульову похибку й 100 %
узгоджених. Якщо контроль не сходиться, далі йти не можна.

Запуск з кореня lab_4:
    python src/03_lsq_vs_ransac.py

Результати:
    results/lsq_control.csv       контроль «кадр із самим собою»
    results/lsq_vs_ransac.csv     по парах: кількість точок і похибки МНК та RANSAC
    figures/lsq_vs_ransac.png     похибки E_inl (логарифмічна вісь), кількість точок підписана
    figures/lsq_vs_ransac_warp.png  кадр A, перенесений у кадр B: МНК проти RANSAC
"""
import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (FIGURES_DIR, MATCH_TAU, MIN_PAIRS, RESULTS_DIR, SEED, SERIES,
                    WORK_MUTUAL, WORK_RATIO, append_row, done_keys, estimate_h, knn_two,
                    load_frame, matches_to_points, mutual_filter, nearest_index,
                    ratio_filter, reproj_errors, sift_features, work_matches)

np.random.seed(SEED)

CTRL_CSV = RESULTS_DIR / "lsq_control.csv"
PAIR_CSV = RESULTS_DIR / "lsq_vs_ransac.csv"
CTRL_FIELDS = ["frame", "n_matches", "max_err_px", "max_abs_H_minus_I", "share_inl"]
PAIR_FIELDS = ["pair", "n_cand", "n_inl_ransac", "E_all_lsq_mean", "E_all_lsq_median",
               "E_all_ransac_mean", "E_all_ransac_median", "E_inl_lsq_mean",
               "E_inl_lsq_median", "E_inl_ransac_mean", "E_inl_ransac_median"]


def lsq_homography(src, dst):
    """Гомографія методом найменших квадратів за ВСІМА парами (прапорець 0)."""
    h, _ = cv2.findHomography(src, dst, 0)
    return h


def run_control(stem):
    """Контроль: кадр проти самого себе. Очікуємо H = I, похибку 0, 100 % узгоджених."""
    kp, d = sift_features(stem)
    cand = ratio_filter(knn_two(d, d), WORK_RATIO)
    if WORK_MUTUAL:
        cand = mutual_filter(cand, nearest_index(d, d))
    src, dst = matches_to_points(cand, kp, kp)
    h = lsq_homography(src, dst)
    err = reproj_errors(h, src, dst)
    row = {"frame": stem, "n_matches": len(cand),
           "max_err_px": float(f"{err.max():.2e}"),
           "max_abs_H_minus_I": float(f"{np.abs(h / h[2, 2] - np.eye(3)).max():.2e}"),
           "share_inl": round(float((err < MATCH_TAU).mean()), 4)}
    if (stem,) not in done_keys(CTRL_CSV, ("frame",)):
        append_row(CTRL_CSV, CTRL_FIELDS, row)
    return row


def run_pair(a, b):
    """Оцінює дві гомографії для пари (a, b); повертає рядок метрик і самі H.
    Якщо кандидатів менше MIN_PAIRS, повертає None (гомографію не оцінити)."""
    src, dst = work_matches(a, b)
    if len(src) < MIN_PAIRS:
        return None
    h_lsq = lsq_homography(src, dst)
    h_ran, mask = estimate_h(src, dst)          # бібліотечний RANSAC, τ = MATCH_TAU
    if h_lsq is None or h_ran is None:
        raise RuntimeError(f"Не вдалося оцінити гомографію для {a}-{b}")
    e_l, e_r = reproj_errors(h_lsq, src, dst), reproj_errors(h_ran, src, dst)
    row = {"pair": f"{a}_{b}", "n_cand": len(src), "n_inl_ransac": int(mask.sum()),
           "E_all_lsq_mean": e_l.mean(), "E_all_lsq_median": np.median(e_l),
           "E_all_ransac_mean": e_r.mean(), "E_all_ransac_median": np.median(e_r),
           "E_inl_lsq_mean": e_l[mask].mean(), "E_inl_lsq_median": np.median(e_l[mask]),
           "E_inl_ransac_mean": e_r[mask].mean(), "E_inl_ransac_median": np.median(e_r[mask])}
    row = {k: (round(float(v), 3) if k != "pair" and isinstance(v, float) else v)
           for k, v in row.items()}
    if (row["pair"],) not in done_keys(PAIR_CSV, ("pair",)):
        append_row(PAIR_CSV, PAIR_FIELDS, row)
    return row, h_lsq, h_ran


def draw_warp(items):
    """Кадр A, перенесений у систему координат кадра B (50 % прозорість поверх B).
    Рядок = пара, зліва МНК, справа RANSAC. Спотворення МНК має бути видно."""
    fig, axes = plt.subplots(len(items), 2, figsize=(12, 3.3 * len(items)), squeeze=False)
    for i, (a, b, row, h_lsq, h_ran) in enumerate(items):
        img_a, img_b = load_frame(a)[0], load_frame(b)[0]
        hb, wb = img_b.shape[:2]
        for j, (name, h, e) in enumerate((("МНК", h_lsq, row["E_inl_lsq_mean"]),
                                          ("RANSAC", h_ran, row["E_inl_ransac_mean"]))):
            warped = cv2.warpPerspective(img_a, h, (wb, hb))
            blend = cv2.addWeighted(warped, 0.5, img_b, 0.5, 0)
            ax = axes[i, j]
            ax.imshow(cv2.cvtColor(blend, cv2.COLOR_BGR2RGB))   # BGR → RGB
            ax.set_title(f"{a[-4:]} → {b[-4:]}, {name}: E_inl = {e:.1f} px "
                         f"({row['n_inl_ransac']} узгоджених із {row['n_cand']})", fontsize=9)
            ax.axis("off")
    fig.suptitle("Кадр A у системі координат кадра B (50 % прозорості). "
                 "Однаковий набір точок для обох методів", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "lsq_vs_ransac_warp.png", dpi=110)
    plt.close(fig)


def draw_bars(rows):
    """Стовпчики E_inl (середнє, px) для МНК і RANSAC по парах, логарифмічна вісь.
    Під назвою пари – кількість узгоджених і кандидатів (вони впливають на інтерпретацію)."""
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - 0.2, [r["E_inl_lsq_mean"] for r in rows], 0.4, label="МНК (усі кандидати)")
    ax.bar(x + 0.2, [r["E_inl_ransac_mean"] for r in rows], 0.4, label="RANSAC")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['pair'][4:8]}→{r['pair'][13:17]}\n{r['n_inl_ransac']} з {r['n_cand']}"
                        for r in rows])
    ax.set_xlabel("Пара кадрів (внизу: узгоджених із кандидатів)")
    ax.set_ylabel("E_inl, px (логарифмічна шкала)")
    ax.set_title("Середня похибка репроєкції на узгоджених точках", fontsize=10)
    ax.set_ylim(top=ax.get_ylim()[1] * 6)   # місце під легенду
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "lsq_vs_ransac.png", dpi=150)
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Умови: T = {WORK_RATIO}, взаємна перевірка = {WORK_MUTUAL}, τ = {MATCH_TAU} px\n")

    print("=== Контроль: кадр проти самого себе (очікуємо H = I, похибка 0, 100 %) ===")
    for stem in SERIES:
        r = run_control(stem)
        print(f"{stem}: пар={r['n_matches']}, макс. похибка={r['max_err_px']} px, "
              f"|H-I|max={r['max_abs_H_minus_I']}, узгоджених={r['share_inl'] * 100:.1f} %")

    print("\n=== МНК проти RANSAC на однакових точках ===")
    rows, items = [], []
    for a, b in zip(SERIES[:-1], SERIES[1:]):
        res = run_pair(a, b)
        if res is None:
            print(f"{a}_{b}: пропущено, кандидатів менше {MIN_PAIRS}")
            continue
        row, h_lsq, h_ran = res
        rows.append(row)
        items.append((a, b, row, h_lsq, h_ran))
        print(f"{row['pair']}: кандидатів={row['n_cand']}, узгоджених={row['n_inl_ransac']}\n"
              f"    E_inl: МНК {row['E_inl_lsq_mean']:.2f} px (медіана {row['E_inl_lsq_median']:.2f}),"
              f" RANSAC {row['E_inl_ransac_mean']:.2f} px (медіана {row['E_inl_ransac_median']:.2f})\n"
              f"    E_all: МНК {row['E_all_lsq_mean']:.1f} px, RANSAC {row['E_all_ransac_mean']:.1f} px")

    if len(rows) < 2:
        raise SystemExit("Замало пар для середнього ± σ")
    for key, label in (("E_inl_lsq_mean", "E_inl МНК"), ("E_inl_ransac_mean", "E_inl RANSAC")):
        v = np.array([r[key] for r in rows])
        print(f"{label}: {v.mean():.2f} ± {v.std(ddof=1):.2f} px по {len(v)} парах")

    draw_warp(items)
    draw_bars(rows)
    print(f"\nРисунки: {FIGURES_DIR / 'lsq_vs_ransac.png'}, {FIGURES_DIR / 'lsq_vs_ransac_warp.png'}")


if __name__ == "__main__":
    main()