"""ЛР4, частина C: власний RANSAC, дослідження порога τ, число ітерацій.

Що робить скрипт:
  1. Контроль на відомій гомографії. Кожен кадр серії деформуємо відомою H0
     (common.CONTROL_H0), знаходимо точки на деформованому кадрі, зіставляємо з
     оригіналом тим самим конвеєром і оцінюємо H власним та бібліотечним RANSAC.
     Критерій (заданий до запуску): образи кутів кадру під оціненою H відрізняються
     від образів під H0 не більше ніж на CONTROL_TOL_PX. Не пройдено – зупиняємось.
  2. Дослідження τ. Для кожної пари сусідніх кадрів і кожного τ з TAU_GRID:
     бібліотечний RANSAC (один запуск, у нього свій генератор) і власний RANSAC
     з N_SEEDS різними seed (SEED + k) і однаковим RANSAC_N_ITER. Усі методи
     працюють на ОДНАКОВОМУ наборі кандидатів (work_matches).
  3. Порівняння числа ітерацій RANSAC із формулою N = log(1-p)/log(1-(1-ε)^s),
     де ε береться з бібліотечного результату (1 - n_inl / n_cand).
  4. Рисунки: залежність від τ і приклади inliers/outliers.

Запуск з кореня lab_4:
    python src/04_ransac.py

Результати:
    results/ransac_control.csv        контроль на відомій H0
    results/ransac_tau.csv            кожен запуск (пара, τ, метод, seed); пишеться рядок за рядком
    results/ransac_tau_summary.csv    середнє ± σ по seed, N_req проти N_used
    figures/ransac_tau.png            кількість/частка узгоджених і E_inl залежно від τ
    figures/ransac_inliers_<пара>.png inliers (зелені) і outliers (червоні) за τ = MATCH_TAU
"""
import csv

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt

from common import (CONTROL_H0, CONTROL_TOL_PX, FIGURES_DIR, MATCH_TAU, MIN_PAIRS,
                    N_SEEDS, RANSAC_N_ITER, RESULTS_DIR, SEED, SERIES, SIFT_NFEATURES,
                    TAU_GRID, WORK_MUTUAL, WORK_RATIO, append_row, corner_transfer_diff,
                    done_keys, estimate_h, knn_two, load_frame, matches_to_points,
                    mutual_filter, nearest_index, ratio_filter, required_iterations,
                    reproj_errors, sift_features, work_matches, MAX_DRAW)
from ransac_core import ransac_homography

np.random.seed(SEED)

CTRL_CSV = RESULTS_DIR / "ransac_control.csv"
TAU_CSV = RESULTS_DIR / "ransac_tau.csv"
SUM_CSV = RESULTS_DIR / "ransac_tau_summary.csv"
CTRL_FIELDS = ["frame", "n_cand", "n_inl_own", "n_inl_lib", "corner_err_own_px",
               "corner_err_lib_px", "passed"]
TAU_FIELDS = ["pair", "tau", "method", "seed", "n_cand", "n_inl", "share",
              "E_inl_mean", "E_inl_median", "n_iter", "corner_diff_vs_lib_px",
              "jaccard_vs_lib"]
SUM_FIELDS = ["pair", "tau", "n_cand", "n_inl_lib", "n_inl_own_mean", "n_inl_own_std",
              "n_inl_own_min", "E_inl_lib", "E_inl_own_mean", "eps", "N_req", "N_used",
              "seeds_below_95pct", "n_seeds"]


# ================================================================ контроль
def run_control(stem):
    """Кадр → кадр, деформований відомою H0. Повертає рядок метрик і чи пройдено."""
    h0 = np.array(CONTROL_H0, dtype=np.float64)
    _, gray = load_frame(stem)
    hh, ww = gray.shape
    warped = cv2.warpPerspective(gray, h0, (ww, hh))      # піксель (x, y) → H0·(x, y)
    # SIFT на деформованому кадрі рахуємо тут, у кеш його не кладемо (це не кадр набору)
    sift = cv2.SIFT_create(nfeatures=SIFT_NFEATURES)
    kp_a, d_a = sift_features(stem)
    kp_b, d_b = sift.detectAndCompute(warped, None)
    cand = ratio_filter(knn_two(d_a, d_b), WORK_RATIO)
    if WORK_MUTUAL:
        cand = mutual_filter(cand, nearest_index(d_b, d_a))
    src, dst = matches_to_points(cand, kp_a, kp_b)
    h_own, m_own = ransac_homography(src, dst, MATCH_TAU)
    h_lib, m_lib = estimate_h(src, dst, MATCH_TAU)
    if h_own is None or h_lib is None:
        row = {"frame": stem, "n_cand": len(src), "n_inl_own": 0, "n_inl_lib": 0,
               "corner_err_own_px": "", "corner_err_lib_px": "", "passed": False}
    else:
        e_own = corner_transfer_diff(h_own, h0, (ww, hh))
        e_lib = corner_transfer_diff(h_lib, h0, (ww, hh))
        row = {"frame": stem, "n_cand": len(src), "n_inl_own": int(m_own.sum()),
               "n_inl_lib": int(m_lib.sum()), "corner_err_own_px": round(e_own, 3),
               "corner_err_lib_px": round(e_lib, 3),
               "passed": bool(e_own <= CONTROL_TOL_PX and e_lib <= CONTROL_TOL_PX)}
    if (stem,) not in done_keys(CTRL_CSV, ("frame",)):
        append_row(CTRL_CSV, CTRL_FIELDS, row)
    return row


# ================================================================ дослідження τ
def _metrics(h, mask, src, dst):
    """E_inl (середнє, медіана) на масці узгоджених."""
    e = reproj_errors(h, src, dst)[mask]
    return float(e.mean()), float(np.median(e))


def study_pair(a, b):
    """Усі τ × (бібліотечний + N_SEEDS власних) для пари (a, b). Пише CSV рядок за рядком."""
    src, dst = work_matches(a, b)
    pair = f"{a}_{b}"
    if len(src) < MIN_PAIRS:
        print(f"{pair}: пропущено, кандидатів менше {MIN_PAIRS}")
        return None
    hh, ww = load_frame(a)[1].shape
    size = (ww, hh)
    done = done_keys(TAU_CSV, ("pair", "tau", "method", "seed"))
    n = len(src)

    def share(k):
        return round(k / n, 4)

    for tau in TAU_GRID:
        h_lib, m_lib = estimate_h(src, dst, tau)
        key = (pair, str(tau), "lib", "")
        if key not in done:
            if h_lib is None:
                row = dict(pair=pair, tau=tau, method="lib", seed="", n_cand=n, n_inl=0,
                           share=0.0, E_inl_mean="", E_inl_median="", n_iter="",
                           corner_diff_vs_lib_px="", jaccard_vs_lib="")
            else:
                em, ed = _metrics(h_lib, m_lib, src, dst)
                row = dict(pair=pair, tau=tau, method="lib", seed="", n_cand=n,
                           n_inl=int(m_lib.sum()), share=share(m_lib.sum()),
                           E_inl_mean=round(em, 3), E_inl_median=round(ed, 3), n_iter="",
                           corner_diff_vs_lib_px=0.0, jaccard_vs_lib=1.0)
            append_row(TAU_CSV, TAU_FIELDS, row)
        for k in range(N_SEEDS):
            key = (pair, str(tau), "own", str(SEED + k))
            if key in done:
                continue
            h, m = ransac_homography(src, dst, tau, RANSAC_N_ITER, SEED + k)
            if h is None:
                row = dict(pair=pair, tau=tau, method="own", seed=SEED + k, n_cand=n,
                           n_inl=0, share=0.0, E_inl_mean="", E_inl_median="",
                           n_iter=RANSAC_N_ITER, corner_diff_vs_lib_px="", jaccard_vs_lib="")
            else:
                em, ed = _metrics(h, m, src, dst)
                if h_lib is None:
                    cd, jac = "", ""
                else:
                    cd = round(corner_transfer_diff(h, h_lib, size), 3)
                    jac = round(float((m & m_lib).sum() / max((m | m_lib).sum(), 1)), 4)
                row = dict(pair=pair, tau=tau, method="own", seed=SEED + k, n_cand=n,
                           n_inl=int(m.sum()), share=share(m.sum()),
                           E_inl_mean=round(em, 3), E_inl_median=round(ed, 3),
                           n_iter=RANSAC_N_ITER, corner_diff_vs_lib_px=cd, jaccard_vs_lib=jac)
            append_row(TAU_CSV, TAU_FIELDS, row)
    return src, dst


def read_tau_csv():
    with open(TAU_CSV, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _f(x):
    return float(x) if x not in ("", None) else float("nan")


def summarize(rows):
    """Зводить ransac_tau.csv у ransac_tau_summary.csv (перезаписується щоразу)."""
    out = []
    pairs = list(dict.fromkeys(r["pair"] for r in rows))
    for pair in pairs:
        for tau in TAU_GRID:
            sel = [r for r in rows if r["pair"] == pair and float(r["tau"]) == tau]
            lib = [r for r in sel if r["method"] == "lib"]
            own = [r for r in sel if r["method"] == "own"]
            if not lib or not own:
                continue
            n_cand, n_lib = int(lib[0]["n_cand"]), int(lib[0]["n_inl"])
            k = np.array([int(r["n_inl"]) for r in own])
            e = np.array([_f(r["E_inl_mean"]) for r in own])
            eps = 1.0 - n_lib / n_cand
            n_req = required_iterations(eps) if n_lib >= 1 else float("inf")
            out.append(dict(
                pair=pair, tau=tau, n_cand=n_cand, n_inl_lib=n_lib,
                n_inl_own_mean=round(k.mean(), 2),
                n_inl_own_std=round(k.std(ddof=1), 2) if len(k) > 1 else 0.0,
                n_inl_own_min=int(k.min()), E_inl_lib=lib[0]["E_inl_mean"],
                E_inl_own_mean=round(float(np.nanmean(e)), 3) if not np.all(np.isnan(e)) else "",
                eps=round(eps, 4), N_req=n_req, N_used=RANSAC_N_ITER,
                seeds_below_95pct=int((k < 0.95 * n_lib).sum()), n_seeds=len(k)))
    if SUM_CSV.exists():
        SUM_CSV.unlink()
    for r in out:
        append_row(SUM_CSV, SUM_FIELDS, r)
    return out


# ================================================================ рисунки
def draw_tau(summary):
    """Три панелі: кількість узгоджених (лог.), частка (%), E_inl; колір = пара.
    Власний RANSAC – суцільна лінія з σ по seed, бібліотечний – пунктир."""
    pairs = list(dict.fromkeys(r["pair"] for r in summary))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for i, pair in enumerate(pairs):
        s = [r for r in summary if r["pair"] == pair]
        t = [r["tau"] for r in s]
        n_cand = s[0]["n_cand"]
        label = f"{pair[4:8]}→{pair[13:17]} ({n_cand} канд.)"
        c = f"C{i}"
        mean = np.array([r["n_inl_own_mean"] for r in s])
        std = np.array([r["n_inl_own_std"] for r in s])
        lib = np.array([r["n_inl_lib"] for r in s])
        axes[0].errorbar(t, mean, yerr=std, color=c, marker="o", ms=4, capsize=2, label=label)
        axes[0].plot(t, lib, color=c, ls="--", lw=1)
        axes[1].errorbar(t, 100 * mean / n_cand, yerr=100 * std / n_cand, color=c,
                         marker="o", ms=4, capsize=2, label=label)
        axes[1].plot(t, 100 * lib / n_cand, color=c, ls="--", lw=1)
        axes[2].plot(t, [_f(r["E_inl_own_mean"]) for r in s], color=c, marker="o", ms=4,
                     label=label)
        axes[2].plot(t, [_f(r["E_inl_lib"]) for r in s], color=c, ls="--", lw=1)
    for ax, yl in zip(axes, ("Узгоджених точок, шт. (лог. шкала)", "Частка узгоджених, %",
                             "E_inl, px (середня похибка на узгоджених)")):
        ax.set_xscale("log")
        ax.set_xticks(TAU_GRID)
        ax.set_xticklabels([f"{t:g}" for t in TAU_GRID])
        ax.set_xlabel("Поріг τ, px")
        ax.set_ylabel(yl)
        ax.grid(alpha=0.3)
    axes[0].set_yscale("log")
    axes[2].set_yscale("log")
    axes[0].legend(fontsize=8, loc="center right")
    fig.suptitle(f"Власний RANSAC (суцільна, середнє ± σ по {N_SEEDS} seed) і бібліотечний "
                 f"(пунктир), N = {RANSAC_N_ITER}", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ransac_tau.png", dpi=130)
    plt.close(fig)


def draw_inliers(a, b, src, dst, tau):
    """Кадри A і B поруч; лінії пар: зелені – узгоджені, червоні – викиди (τ = tau).
    Показуємо не більше MAX_DRAW ліній кожного класу (випадкова вибірка, seed = SEED)."""
    h, mask = ransac_homography(src, dst, tau, RANSAC_N_ITER, SEED)
    if h is None:
        return
    img_a, img_b = load_frame(a)[0], load_frame(b)[0]
    ha, wa = img_a.shape[:2]
    canvas = np.zeros((max(ha, img_b.shape[0]), wa + img_b.shape[1], 3), np.uint8)
    canvas[:ha, :wa] = img_a
    canvas[:img_b.shape[0], wa:] = img_b
    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    for sel, color, name in ((mask, "lime", "узгоджені"), (~mask, "red", "викиди")):
        idx = np.flatnonzero(sel)
        total = len(idx)
        if total > MAX_DRAW:
            idx = rng.choice(idx, MAX_DRAW, replace=False)
        segs = [[(src[i, 0], src[i, 1]), (dst[i, 0] + wa, dst[i, 1])] for i in idx]
        ax.add_collection(LineCollection(segs, colors=color, linewidths=0.6, alpha=0.7,
                                         label=f"{name}: {total} (показано {len(idx)})"))
    ax.set_xlabel("x, px (кадр A зліва, кадр B справа)")
    ax.set_ylabel("y, px")
    ax.set_title(f"{a[-4:]} → {b[-4:]}, τ = {tau:g} px, N = {RANSAC_N_ITER}: "
                 f"{int(mask.sum())} узгоджених із {len(src)}", fontsize=10)
    ax.legend(loc="lower center", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"ransac_inliers_{a[-4:]}_{b[-4:]}.png", dpi=120)
    plt.close(fig)


# ================================================================ main
def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Умови: T = {WORK_RATIO}, взаємна = {WORK_MUTUAL}, N = {RANSAC_N_ITER}, "
          f"seed = {SEED}..{SEED + N_SEEDS - 1}\n")

    print(f"=== Контроль: кадр, деформований відомою H0 (допуск {CONTROL_TOL_PX} px) ===")
    ok = True
    for stem in SERIES:
        r = run_control(stem)
        print(f"{stem}: кандидатів={r['n_cand']}, узгоджених власний/бібл.="
              f"{r['n_inl_own']}/{r['n_inl_lib']}, зсув кутів власний/бібл.="
              f"{r['corner_err_own_px']}/{r['corner_err_lib_px']} px, "
              f"{'ПРОЙДЕНО' if r['passed'] else 'НЕ ПРОЙДЕНО'}")
        ok &= bool(r["passed"])
    if not ok:
        raise SystemExit("Контроль не пройдено: шукаємо причину, параметри не підганяємо.")

    print("\n=== Дослідження τ ===")
    data = {}
    for a, b in zip(SERIES[:-1], SERIES[1:]):
        res = study_pair(a, b)
        if res is not None:
            data[(a, b)] = res
    summary = summarize(read_tau_csv())

    print(f"\n{'пара':<11}{'τ':>5}{'канд.':>7}{'бібл.':>7}{'власний (сер.±σ, мін)':>26}"
          f"{'E_inl':>7}{'ε':>7}{'N_req':>8}{'N':>6}{'<95%':>6}")
    for r in summary:
        n_req = "inf" if r["N_req"] == float("inf") else int(r["N_req"])
        print(f"{r['pair'][4:8]}-{r['pair'][13:17]:<5}{r['tau']:>5g}{r['n_cand']:>7}"
              f"{r['n_inl_lib']:>7}"
              f"{r['n_inl_own_mean']:>14.1f} ± {r['n_inl_own_std']:<5.1f}{r['n_inl_own_min']:>5}"
              f"{_f(r['E_inl_own_mean']):>7.2f}{r['eps']:>7.2f}{n_req:>8}{r['N_used']:>6}"
              f"{r['seeds_below_95pct']:>4}/{r['n_seeds']}")

    draw_tau(summary)
    for (a, b), (src, dst) in data.items():
        draw_inliers(a, b, src, dst, MATCH_TAU)
    print(f"\nРисунки у {FIGURES_DIR}")


if __name__ == "__main__":
    main()