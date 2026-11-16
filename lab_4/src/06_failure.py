"""ЛР4, частина E: аналіз випадку відмови – хибна гомографія для несусідніх кадрів.

Пара FAIL_PAIR (8735–8737) спільного вмісту не має (за ланцюгом перекриття 0 px), а
RANSAC знаходить 226 «узгоджених» точок із нерозумною гомографією. Гіпотеза:
відповідності хибні, бо шпалери з повторюваним мотивом: у кадрі A той самий
фрагмент трапляється багато разів, тож найближчий дескриптор у кадрі B належить
іншому екземпляру мотиву. Скрипт перевіряє гіпотезу числами, а не на око.

Що робить:
  1. Для обох пар (FAIL_PAIR і CONTROL_PAIR, сусідня з тих самих кадрів) і для
     кожного T з RATIO_GRID, з взаємною перевіркою та без неї: кандидати,
     власний RANSAC (τ = MATCH_TAU), кількість і частка узгоджених, чи правдоподібна H.
     Питання: чи прибирає хибну H жорсткіший поріг T або взаємна перевірка?
  2. Тест на повторюваність. Для кожної відповідності (i → j) робочого набору
     (T = WORK_RATIO, взаємна) міряємо r_dup = d(i, найближчий ІНШИЙ дескриптор у тому
     самому кадрі A) / d(i, j). r_dup < 1 означає, що в самому кадрі A є двійник
     точки, схожіший на неї, ніж відповідник у B, тобто відповідність неоднозначна.
     Порівнюємо три групи: «узгоджені» FAIL_PAIR, «викиди» FAIL_PAIR, узгоджені CONTROL_PAIR.

Запуск з кореня lab_4:
    python src/06_failure.py

Результати (перезаписуються при кожному запуску):
    results/failure_T.csv       пари × T × взаємна: кандидати, узгоджені, правдоподібність
    results/failure_dup.csv     групи: кількість, медіана r_dup, частка r_dup < 1
    figures/failure_T.png       кількість і частка узгоджених залежно від T
    figures/failure_pair.png    узгоджені лінії FAIL_PAIR і гістограма r_dup
"""
import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt

from common import (CONTROL_PAIR, FAIL_PAIR, FIGURES_DIR, MATCH_TAU, MAX_DRAW, RANSAC_N_ITER,
                    RATIO_GRID, RESULTS_DIR, SEED, WORK_MUTUAL, WORK_RATIO, append_row,
                    knn_k, knn_two, load_frame, matches_to_points, mutual_filter, nearest_index,
                    overlap_fraction, ratio_filter, sift_features)
from ransac_core import ransac_homography

np.random.seed(SEED)

T_CSV = RESULTS_DIR / "failure_T.csv"
DUP_CSV = RESULTS_DIR / "failure_dup.csv"
T_FIELDS = ["pair", "T", "mutual", "n_cand", "n_inl", "share", "plausible"]
DUP_FIELDS = ["group", "n", "median_r_dup", "share_r_dup_below_1"]


def candidates(a, b, t, mutual):
    """Кандидати пари (a, b): відношення Лоу з порогом t і, за потреби, взаємна перевірка.
    Повертає (список DMatch, KeyPoint a, KeyPoint b)."""
    kp_a, d_a = sift_features(a)
    kp_b, d_b = sift_features(b)
    cand = ratio_filter(knn_two(d_a, d_b), t)
    if mutual:
        cand = mutual_filter(cand, nearest_index(d_b, d_a))
    return cand, kp_a, kp_b


def run_cell(a, b, t, mutual):
    """Один рядок таблиці: RANSAC на кандидатах (a, b, t, mutual)."""
    cand, kp_a, kp_b = candidates(a, b, t, mutual)
    row = {"pair": f"{a[-4:]}-{b[-4:]}", "T": t, "mutual": mutual, "n_cand": len(cand),
           "n_inl": "", "share": "", "plausible": ""}
    if len(cand) >= 4:
        src, dst = matches_to_points(cand, kp_a, kp_b)
        h, mask = ransac_homography(src, dst, MATCH_TAU, RANSAC_N_ITER, SEED)
        if h is not None:
            wa, ha = load_frame(a)[1].shape[::-1]
            wb, hb = load_frame(b)[1].shape[::-1]
            row.update(n_inl=int(mask.sum()), share=round(float(mask.mean()), 4),
                       plausible=bool(overlap_fraction(h, (wa, ha), (wb, hb))[1]))
    append_row(T_CSV, T_FIELDS, row)
    return row


def self_nearest_dist(stem):
    """Для кожного дескриптора кадру – відстань до найближчого ІНШОГО дескриптора
    того самого кадру (сам себе пропускаємо за індексом)."""
    _, d = sift_features(stem)
    knn = knn_k(d, d, 3)   # сам дескриптор + два найближчі інші
    out = np.empty(len(knn))
    for i, m in enumerate(knn):
        out[i] = next(x.distance for x in m if x.trainIdx != i)
    return out


def dup_groups():
    """r_dup для трьох груп відповідностей робочого набору (див. docstring модуля)."""
    groups, ratios = {}, {}
    for name, (a, b) in (("fail", FAIL_PAIR), ("control", CONTROL_PAIR)):
        cand, kp_a, kp_b = candidates(a, b, WORK_RATIO, WORK_MUTUAL)
        src, dst = matches_to_points(cand, kp_a, kp_b)
        _, mask = ransac_homography(src, dst, MATCH_TAU, RANSAC_N_ITER, SEED)
        d_self = self_nearest_dist(a)
        r = np.array([d_self[m.queryIdx] / max(m.distance, 1e-9) for m in cand])
        groups[name] = (src, dst, mask)
        ratios[name] = (r, mask)
    sets = {f"{FAIL_PAIR[0][-4:]}-{FAIL_PAIR[1][-4:]}, узгоджені": ratios["fail"][0][ratios["fail"][1]],
            f"{FAIL_PAIR[0][-4:]}-{FAIL_PAIR[1][-4:]}, викиди": ratios["fail"][0][~ratios["fail"][1]],
            f"{CONTROL_PAIR[0][-4:]}-{CONTROL_PAIR[1][-4:]}, узгоджені":
                ratios["control"][0][ratios["control"][1]]}
    return sets, groups["fail"]


def draw_t(rows):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for pair, color in (("fail", "C3"), ("control", "C0")):
        a, b = FAIL_PAIR if pair == "fail" else CONTROL_PAIR
        label_pair = f"{a[-4:]}-{b[-4:]}"
        for mutual, ls in ((False, "--"), (True, "-")):
            sel = [r for r in rows if r["pair"] == label_pair and r["mutual"] == mutual
                   and r["n_inl"] != ""]
            t = [r["T"] for r in sel]
            lab = f"{label_pair}, {'з взаємною' if mutual else 'лише Лоу'}"
            axes[0].plot(t, [r["n_inl"] for r in sel], ls, color=color, marker="o", ms=4, label=lab)
            axes[1].plot(t, [100 * r["share"] for r in sel], ls, color=color, marker="o", ms=4)
            for r in sel:
                if r["plausible"] is False:       # нерозумна H позначена хрестиком
                    axes[0].plot(r["T"], r["n_inl"], "kx", ms=9)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Узгоджених точок, шт. (лог. шкала)")
    axes[1].set_ylabel("Частка узгоджених, %")
    for ax in axes:
        ax.set_xlabel("Поріг відношення Лоу T")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Хибна пара проти сусідньої, τ = %g px; × – гомографія нерозумна" % MATCH_TAU,
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "failure_T.png", dpi=130)
    plt.close(fig)


def draw_pair(sets, fail_data):
    """Ліворуч – «узгоджені» лінії хибної пари; праворуч – гістограма r_dup трьох груп."""
    src, dst, mask = fail_data
    a, b = FAIL_PAIR
    img_a, img_b = load_frame(a)[0], load_frame(b)[0]
    ha, wa = img_a.shape[:2]
    canvas = np.zeros((max(ha, img_b.shape[0]), wa + img_b.shape[1], 3), np.uint8)
    canvas[:ha, :wa] = img_a
    canvas[:img_b.shape[0], wa:] = img_b
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={"width_ratios": [2, 1]})
    ax1.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    idx = np.flatnonzero(mask)
    total = len(idx)
    if total > MAX_DRAW:
        idx = np.random.default_rng(SEED).choice(idx, MAX_DRAW, replace=False)
    ax1.add_collection(LineCollection([[(src[i, 0], src[i, 1]), (dst[i, 0] + wa, dst[i, 1])]
                                       for i in idx], colors="lime", linewidths=0.6, alpha=0.8))
    ax1.set_title(f"{a[-4:]} → {b[-4:]}: «узгоджені» лінії ({total}, показано {len(idx)}); "
                  "кадри спільного вмісту не мають", fontsize=9)
    ax1.set_xlabel("x, px (кадр A зліва, кадр B справа)")
    ax1.set_ylabel("y, px")
    bins = np.linspace(0, 3, 31)
    for (name, r), c in zip(sets.items(), ("C3", "C1", "C0")):
        if len(r):
            ax2.hist(np.clip(r, 0, 3), bins=bins, histtype="step", density=True, color=c,
                     lw=1.6, label=f"{name} (n = {len(r)})")
    ax2.axvline(1.0, color="gray", ls="--", lw=1)
    ax2.set_xlabel("r_dup = d(двійник у кадрі A) / d(відповідник у B), обрізано на 3")
    ax2.set_ylabel("Густина")
    ax2.set_title("Правіше 1 – відповідність однозначна", fontsize=9)
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "failure_pair.png", dpi=130)
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for f in (T_CSV, DUP_CSV):
        if f.exists():
            f.unlink()
    print(f"Умови: τ = {MATCH_TAU} px, N = {RANSAC_N_ITER}, seed = {SEED}\n")

    print("=== Чи прибирає хибну H жорсткіший T або взаємна перевірка? ===")
    rows = []
    for a, b in (FAIL_PAIR, CONTROL_PAIR):
        for mutual in (False, True):
            for t in RATIO_GRID:
                rows.append(run_cell(a, b, t, mutual))
    print(f"{'пара':<10}{'взаємна':<9}{'T':>5}{'канд.':>7}{'узг.':>6}{'%':>7}{'прав.':>7}")
    for r in rows:
        sh = f"{100 * r['share']:.1f}" if r["share"] != "" else "–"
        print(f"{r['pair']:<10}{str(r['mutual']):<9}{r['T']:>5}{r['n_cand']:>7}"
              f"{r['n_inl'] if r['n_inl'] != '' else '–':>6}{sh:>7}"
              f"{str(r['plausible']) if r['plausible'] != '' else '–':>7}")

    print(f"\n=== Тест на повторюваність (T = {WORK_RATIO}, взаємна = {WORK_MUTUAL}) ===")
    sets, fail_data = dup_groups()
    for name, r in sets.items():
        row = {"group": name, "n": len(r),
               "median_r_dup": round(float(np.median(r)), 3) if len(r) else "",
               "share_r_dup_below_1": round(float((r < 1.0).mean()), 4) if len(r) else ""}
        append_row(DUP_CSV, DUP_FIELDS, row)
        print(f"{name:<24} n={row['n']:>5}  медіана r_dup={row['median_r_dup']}  "
              f"частка r_dup<1: {row['share_r_dup_below_1']}")

    draw_t(rows)
    draw_pair(sets, fail_data)
    print(f"\nРисунки у {FIGURES_DIR}")


if __name__ == "__main__":
    main()