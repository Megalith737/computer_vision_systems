"""Частина A. Аудит набору: дублі, позиції камери, L/R, зсуви кадрів (ЛР5).

Запуск з кореня lab_5:  python src/01_pair_audit.py
Читає data/raw/IMG_*.jpg (імена в common.SCENES). Результати:
  results/frame_stats.csv    різкість, яскравість, розмір, EXIF кожного кадру
  results/pair_audit.csv     усі пари кадрів: узгоджені точки, медіана зсуву, клас пари
  results/primary_pairs.csv  основна стереопара кожної сцени (ліва, права) за різкістю
  figures/pair_audit_matrix.png  матриця кількості узгоджених точок між кадрами

Клас пари визначається за числами (пороги в common.py):
  «дубль»         – той самий ракурс (медіана зсуву мала);
  «стереопара»    – та сама сцена, але камеру зсунуто;
  «різні сцени»   – узгоджених точок мало.
Контроль: кадр, зіставлений сам із собою, дає нульовий зсув для всіх кандидатів.
Файли пишуться рядок за рядком: після переривання скрипт продовжить. Якщо змінили
параметри в common.py, видаліть results/pair_audit.csv і results/frame_stats.csv.
"""
import itertools

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from common import (AUDIT_RATIO, CONTROL_MIN_FRACTION, DUP_MAX_SHIFT_PX, F_RANSAC_CONF,
                    F_RANSAC_THR_PX, FIGURES_DIR, FRAMES, MAX_SIDE, RESULTS_DIR, SCENES, SEED,
                    SIFT_NFEATURES, STEREO_MIN_INLIERS, append_row, done_keys, load_camera_params,
                    load_frame, matches_to_points, native_size, np_seed_all, ratio_matches,
                    read_exif, read_rows, sharpness, sift_features)

STATS_CSV = RESULTS_DIR / "frame_stats.csv"
PAIRS_CSV = RESULTS_DIR / "pair_audit.csv"
PRIMARY_CSV = RESULTS_DIR / "primary_pairs.csv"
FIG_MATRIX = FIGURES_DIR / "pair_audit_matrix.png"

STAT_FIELDS = ["frame", "scene", "native_w", "native_h", "work_w", "work_h", "sift_points",
               "sharpness_lapvar", "mean_gray", "exif_focal_mm", "exif_focal35_mm",
               "exif_exposure_s", "exif_iso", "exif_lens"]
PAIR_FIELDS = ["a", "b", "same_scene_declared", "n_candidates", "n_inliers", "inlier_ratio",
               "median_dx_px", "median_dy_px", "dx_p5_px", "dx_p95_px", "cls", "a_is_left"]
PRIMARY_FIELDS = ["scene", "left", "right", "group_left", "group_right", "sharp_left",
                  "sharp_right", "n_inliers", "dx_left_to_right_px", "dy_left_to_right_px",
                  "lr_votes"]


def scene_of(frame):
    """Назва сцени, до якої належить кадр (за common.SCENES)."""
    return next(s for s, frames in SCENES.items() if frame in frames)


# ================================================================ кадри
def frame_row(frame):
    """Рядок статистики одного кадру."""
    _, gray = load_frame(frame)
    kp, _ = sift_features(frame)
    nw, nh = native_size(frame)
    row = {"frame": frame, "scene": scene_of(frame), "native_w": nw, "native_h": nh,
           "work_w": gray.shape[1], "work_h": gray.shape[0], "sift_points": len(kp),
           "sharpness_lapvar": round(sharpness(gray), 1), "mean_gray": round(float(gray.mean()), 1)}
    row.update(read_exif(frame))
    return row


# ================================================================ пари
def classify(n_inl, dx, dy):
    """Клас пари за числами: дубль, стереопара чи різні сцени."""
    if n_inl < STEREO_MIN_INLIERS:
        return "різні сцени"
    return "дубль" if max(abs(dx), abs(dy)) <= DUP_MAX_SHIFT_PX else "стереопара"


def pair_row(a, b):
    """Зіставлення кадрів a → b: SIFT + Лоу + RANSAC (фундаментальна матриця).
    dx = x_b − x_a, dy = y_b − y_a (px у кадрі MAX_SIDE), медіана по узгоджених точках."""
    kp_a, d_a = sift_features(a)
    kp_b, d_b = sift_features(b)
    cand = ratio_matches(d_a, d_b)
    row = {"a": a, "b": b, "same_scene_declared": int(scene_of(a) == scene_of(b)),
           "n_candidates": len(cand), "n_inliers": 0, "inlier_ratio": 0.0,
           "median_dx_px": "", "median_dy_px": "", "dx_p5_px": "", "dx_p95_px": "",
           "cls": "різні сцени", "a_is_left": ""}
    if len(cand) < 8:  # для фундаментальної матриці потрібно щонайменше 8 пар
        return row
    src, dst = matches_to_points(cand, kp_a, kp_b)
    f_mat, mask = cv2.findFundamentalMat(src, dst, cv2.FM_RANSAC, F_RANSAC_THR_PX, F_RANSAC_CONF)
    if f_mat is None or mask is None:
        return row
    m = mask.ravel().astype(bool)
    d = dst[m] - src[m]
    dx, dy = float(np.median(d[:, 0])), float(np.median(d[:, 1]))
    row.update({"n_inliers": int(m.sum()), "inlier_ratio": round(float(m.mean()), 3),
                "median_dx_px": round(dx, 2), "median_dy_px": round(dy, 2),
                "dx_p5_px": round(float(np.percentile(d[:, 0], 5)), 1),
                "dx_p95_px": round(float(np.percentile(d[:, 0], 95)), 1)})
    row["cls"] = classify(row["n_inliers"], dx, dy)
    if row["cls"] == "стереопара":
        # Права камера бачить предмети лівіше: якщо dx < 0, кадр a лівий.
        row["a_is_left"] = int(dx < 0)
    return row


def control_row(frame):
    """Контроль «тотожне перетворення»: кадр із самим собою. Усі кандидати мають
    нульовий зсув (частка = 1). Частку пишемо в inlier_ratio, кількість у n_inliers."""
    kp, d = sift_features(frame)
    cand = ratio_matches(d, d)
    src, dst = matches_to_points(cand, kp, kp)
    zero = int((np.linalg.norm(dst - src, axis=1) < 1e-3).sum()) if len(cand) else 0
    frac = zero / len(cand) if cand else 0.0
    return {"a": frame, "b": frame, "same_scene_declared": 1, "n_candidates": len(cand),
            "n_inliers": zero, "inlier_ratio": round(frac, 4), "median_dx_px": 0.0,
            "median_dy_px": 0.0, "dx_p5_px": "", "dx_p95_px": "", "cls": "контроль",
            "a_is_left": ""}


# ================================================================ основні пари
def build_primary(pair_rows, stat_rows):
    """Для кожної сцени: групи дублів (мають бути 2), яка група ліва, основна пара за різкістю."""
    sharp = {r["frame"]: float(r["sharpness_lapvar"]) for r in stat_rows}
    prow = {(r["a"], r["b"]): r for r in pair_rows if r["a"] != r["b"]}
    out, notes = [], []
    for scene, frames in SCENES.items():
        parent = {f: f for f in frames}

        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x

        for a, b in itertools.combinations(frames, 2):
            r = prow.get((a, b)) or prow.get((b, a))
            if r and r["cls"] == "дубль":
                parent[find(a)] = find(b)
        groups = {}
        for f in frames:
            groups.setdefault(find(f), []).append(f)
        groups = sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])
        if len(groups) != 2:
            notes.append(f"Сцена {scene}: груп дублів {len(groups)} (очікується 2): {groups}")
            continue
        idx = {f: i for i, g in enumerate(groups) for f in g}
        votes = [0, 0]  # скільки пар вказують, що група i ліва
        for a, b in itertools.product(groups[0], groups[1]):
            r = prow.get((a, b)) or prow.get((b, a))
            if not r or r["cls"] != "стереопара":
                notes.append(f"Сцена {scene}: пара {a}-{b} не стереопара ({r['cls'] if r else 'немає'})")
                continue
            left = r["a"] if str(r["a_is_left"]) == "1" else r["b"]
            votes[idx[left]] += 1
        left_g = 0 if votes[0] >= votes[1] else 1
        lg, rg = groups[left_g], groups[1 - left_g]
        left = max(lg, key=lambda f: sharp[f])
        right = max(rg, key=lambda f: sharp[f])
        r = prow.get((left, right)) or prow.get((right, left))
        sign = 1.0 if r["a"] == left else -1.0  # dx_L→R = x_right − x_left
        out.append({"scene": scene, "left": left, "right": right, "group_left": "|".join(lg),
                    "group_right": "|".join(rg), "sharp_left": sharp[left],
                    "sharp_right": sharp[right], "n_inliers": r["n_inliers"],
                    "dx_left_to_right_px": round(sign * float(r["median_dx_px"]), 2),
                    "dy_left_to_right_px": round(sign * float(r["median_dy_px"]), 2),
                    "lr_votes": f"{max(votes)}/{sum(votes)}"})
    return out, notes


# ================================================================ рисунок
def plot_matrix(pair_rows):
    """Матриця кількості узгоджених точок між кадрами (діагональ порожня)."""
    n = len(FRAMES)
    mat = np.full((n, n), np.nan)
    for r in pair_rows:
        if r["a"] == r["b"]:
            continue
        i, j = FRAMES.index(r["a"]), FRAMES.index(r["b"])
        mat[i, j] = mat[j, i] = max(float(r["n_inliers"]), 1.0)  # 0 не показати на лог. шкалі
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(mat, cmap="viridis", norm=LogNorm(vmin=1, vmax=np.nanmax(mat)))
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f"{int(mat[i, j])}", ha="center", va="center", fontsize=8,
                        color="white" if mat[i, j] < 100 else "black")
    ax.set_xticks(range(n), FRAMES, rotation=45, ha="right")
    ax.set_yticks(range(n), FRAMES)
    ax.set_xlabel("Кадр b")
    ax.set_ylabel("Кадр a")
    ax.set_title("Узгоджені точки між кадрами (SIFT, Лоу, F-RANSAC)")
    fig.colorbar(im, ax=ax, label="Узгоджених пар, шт. (логарифмічна шкала)")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_MATRIX, dpi=150)
    plt.close(fig)


# ================================================================ main
def main():
    np_seed_all(SEED)
    print(f"Параметри: MAX_SIDE={MAX_SIDE}, SIFT nfeatures={SIFT_NFEATURES}, Лоу T={AUDIT_RATIO}, "
          f"F-RANSAC {F_RANSAC_THR_PX} px / p={F_RANSAC_CONF}, дубль ≤ {DUP_MAX_SHIFT_PX} px, "
          f"стереопара ≥ {STEREO_MIN_INLIERS} точок")

    # 1. статистика кадрів
    have = done_keys(STATS_CSV, ("frame",))
    for f in FRAMES:
        if (f,) not in have:
            append_row(STATS_CSV, STAT_FIELDS, frame_row(f))
    stats = read_rows(STATS_CSV)

    # 2. контроль (тотожність) і всі пари
    have = done_keys(PAIRS_CSV, ("a", "b"))
    for f in FRAMES:
        if (f, f) not in have:
            append_row(PAIRS_CSV, PAIR_FIELDS, control_row(f))
    for a, b in itertools.combinations(FRAMES, 2):
        if (a, b) not in have:
            append_row(PAIRS_CSV, PAIR_FIELDS, pair_row(a, b))
            print(f"пара {a} – {b} готова")
    pairs = read_rows(PAIRS_CSV)

    # 3. основні пари й рисунок
    primary, notes = build_primary(pairs, stats)
    if PRIMARY_CSV.exists():
        PRIMARY_CSV.unlink()
    for r in primary:
        append_row(PRIMARY_CSV, PRIMARY_FIELDS, r)
    plot_matrix(pairs)

    # 4. підсумок і перевірки
    print("\n=== Контроль (кадр із самим собою) ===")
    bad = [r for r in pairs if r["cls"] == "контроль" and float(r["inlier_ratio"]) < CONTROL_MIN_FRACTION]
    for r in pairs:
        if r["cls"] == "контроль":
            print(f"{r['a']}: кандидатів {r['n_candidates']}, із нульовим зсувом {float(r['inlier_ratio'])*100:.1f} %")
    print("Контроль:", "ПРОЙДЕНО" if not bad else f"НЕ ПРОЙДЕНО ({[r['a'] for r in bad]}); шукайте причину")

    print("\n=== Класи пар ===")
    for r in pairs:
        if r["a"] == r["b"]:
            continue
        declared = int(r["same_scene_declared"])
        flag = ""
        if declared and r["cls"] == "різні сцени":
            flag = "  <-- ОЧІКУВАЛАСЬ та сама сцена"
        if not declared and r["cls"] != "різні сцени":
            flag = "  <-- ОЧІКУВАЛИСЬ різні сцени"
        print(f"{r['a']} - {r['b']}: {r['cls']:12s} узгоджених {r['n_inliers']:>5}  dx={r['median_dx_px']:>7}  "
              f"dy={r['median_dy_px']:>6}{flag}")

    print("\n=== Основні пари ===")
    for r in primary:
        print(f"Сцена {r['scene']}: ліва {r['left']}, права {r['right']} (різкість {r['sharp_left']:.0f} / "
              f"{r['sharp_right']:.0f}), узгоджених {r['n_inliers']}, dx(L→R)={r['dx_left_to_right_px']} px, "
              f"dy(L→R)={r['dy_left_to_right_px']} px, голосів за L/R {r['lr_votes']}")
    for n in notes:
        print("УВАГА:", n)

    print("\n=== Розмір і EXIF ===")
    _, _, (cw, ch) = load_camera_params()
    sizes = {(int(r["native_w"]), int(r["native_h"])) for r in stats}
    print(f"Калібрування ЛР2: {cw}x{ch} (відношення сторін {cw / ch:.4f}); кадри ЛР5: {sorted(sizes)}")
    if any(abs(w / h - cw / ch) > 0.002 for w, h in sizes):
        print("УВАГА: відношення сторін кадрів не збігається з калібруванням (кадр повернуто чи інший формат): "
              "K непридатна")
    elif sizes != {(cw, ch)}:
        # Розмір інший, але відношення сторін те саме: K масштабується разом із зображенням. У ЛР2 кадри
        # 24 Мп (4284x5712) теж зводились до 3024x4032. Припущення: поле зору в режимах 12 і 24 Мп однакове
        # (імовірно, обидва беруть усе поле сенсора; окремо не перевірено).
        print("Розмір оригіналів відрізняється від калібрування, відношення сторін те саме: K масштабується "
              "(у ЛР2 кадри 24 Мп теж зводились до розміру калібрування)")
    lenses = sorted({(r["exif_focal35_mm"], r["exif_lens"]) for r in stats})
    print("EXIF (еквівалент 35 мм, об'єктив):", lenses)
    if len(lenses) > 1:
        print("УВАГА: кадри знято різними об'єктивами або з різним зумом")
    print(f"\nГотово: {STATS_CSV.name}, {PAIRS_CSV.name}, {PRIMARY_CSV.name}, {FIG_MATRIX.name}")


if __name__ == "__main__":
    main()