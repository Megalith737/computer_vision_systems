"""Частина B. Ректифікація стереопар сцен T і F (ЛР5, практикум, кроки 4-6).

Запуск з кореня lab_5:  python src/02_rectify.py
Потрібні results/primary_pairs.csv (частина A) і ../lab_2/camera_params.yml.

Варіант 2 практикуму (одна камера, зсув убік): шахівниці в парі немає, тому stereoCalibrate
(крок 3) неможливий. R і напрям T між позиціями оцінюємо за відповідностями SIFT (істотна
матриця, undistort за параметрами ЛР2), а довжину T задає база BASELINE_MM. Для кожної сцени
перебираємо всі 4 комбінації дублів: розкид оцінок між ними показує, наскільки R можна довіряти.

Результати:
  results/rectify_combos.csv      оцінка пози й вертикальної неузгодженості для кожної комбінації
  results/rectify_<сцена>.npz     K, dist, R, T (мм), R1, R2, P1, P2, Q основної пари (для частини C)
  data/rectified/<сцена>_L|R.png  ректифіковані кадри основної пари (у git не йдуть)
  figures/rectify_grid_<сцена>.png  контроль: горизонтальні лінії кожні GRID_STEP_PX px (крок 5)

Вертикальна неузгодженість тут рахується за автоматичними SIFT-відповідностями, з яких оцінено
саму позу, тож це перевірка узгодженості, а не незалежна. Незалежна – ручні пари (наступна частина).
Контроль із бібліотечною реалізацією: stereoRectifyUncalibrated (без K і без пози) на тих самих точках.
Файл пишеться рядок за рядком; після зміни параметрів видаліть results/rectify_combos.csv.
"""
import itertools

import cv2
import numpy as np

from common import (BASELINE_MM, ESSENTIAL_CONF, ESSENTIAL_THR_PX, FIGURES_DIR, GRID_FIG_SCALE,
                    GRID_STEP_PX, RECT_ALPHA, RECT_DIR, RECT_VERT_TOL_PX, RESULTS_DIR, SCENES, SEED,
                    append_row, done_keys, load_frame, load_primary_pairs, matches_to_points,
                    np_seed_all, ratio_matches, read_rows, scaled_camera, sift_features)

COMBOS_CSV = RESULTS_DIR / "rectify_combos.csv"
FIELDS = ["scene", "left", "right", "is_primary", "n_candidates", "n_inliers", "pitch_deg", "yaw_deg",
          "roll_deg", "rot_total_deg", "bx", "by", "bz", "f_rect_px", "dy_mean_px", "dy_std_px",
          "dy_median_px", "dy_p95_px", "dy_uncal_mean_px", "d_p1_px", "d_p50_px", "d_p99_px",
          "z_far_mm", "z_near_mm"]


def estimate_pose(left, right, k, dist):
    """R, напрям T і відповідності-«inliers» для пари (left → right).

    Точки спершу перетворюємо в нормовані координати (undistortPoints: дисторсія ЛР2 + K), тож
    істотна матриця E оцінюється без K і порогом thr = 1 px / fx. recoverPose вибирає з чотирьох
    розкладів E той, за якого точки лежать перед обома камерами; t має одиничну довжину.
    Повертає (точки лівого кадру, точки правого кадру, R, t_unit, кількість кандидатів)."""
    kp_l, d_l = sift_features(left)
    kp_r, d_r = sift_features(right)
    cand = ratio_matches(d_l, d_r)
    p_l, p_r = matches_to_points(cand, kp_l, kp_r)
    n_l = cv2.undistortPoints(p_l.reshape(-1, 1, 2), k, dist).reshape(-1, 2)
    n_r = cv2.undistortPoints(p_r.reshape(-1, 1, 2), k, dist).reshape(-1, 2)
    e, mask = cv2.findEssentialMat(n_l, n_r, np.eye(3), cv2.RANSAC, ESSENTIAL_CONF, ESSENTIAL_THR_PX / k[0, 0])
    if e is None or mask is None:
        raise RuntimeError(f"Не вдалося оцінити істотну матрицю для {left} – {right}")
    e = e[:3]  # якщо знайдено кілька розв'язків, беремо перший
    m = mask.ravel().astype(bool)
    _, r_mat, t_unit, _ = cv2.recoverPose(e, n_l[m], n_r[m], np.eye(3))
    return p_l[m], p_r[m], r_mat, t_unit.ravel(), len(cand)


def rectify_points(p, k, dist, rot, proj):
    """Піксельні координати точок після дисторсії-корекції й ректифікації (rot, proj із stereoRectify)."""
    return cv2.undistortPoints(p.reshape(-1, 1, 2), k, dist, R=rot, P=proj).reshape(-1, 2)


def combo_row(scene, left, right, is_primary, size, k, dist, save=False):
    """Оцінка пози, ректифікація й статистика для однієї комбінації дублів."""
    p_l, p_r, r_mat, t_unit, n_cand = estimate_pose(left, right, k, dist)
    t_mm = (t_unit * BASELINE_MM).reshape(3, 1)  # довжину бази задаємо вручну
    r1, r2, p1, p2, q, *_ = cv2.stereoRectify(k, dist, k, dist, size, r_mat, t_mm,
                                              flags=cv2.CALIB_ZERO_DISPARITY, alpha=RECT_ALPHA)
    u_l = rectify_points(p_l, k, dist, r1, p1)
    u_r = rectify_points(p_r, k, dist, r2, p2)
    dy = np.abs(u_l[:, 1] - u_r[:, 1])
    disp = u_l[:, 0] - u_r[:, 0]  # диспаратність d = x_L − x_R; для правильної пари d > 0
    if np.median(disp) <= 0:
        raise RuntimeError(f"{left} – {right}: диспаратність від'ємна, ліва й права позиції переплутані")

    # Контроль: бібліотечна ректифікація без K і пози (за фундаментальною матрицею тих самих точок).
    f_mat, _ = cv2.findFundamentalMat(p_l, p_r, cv2.FM_LMEDS)
    dy_uncal = float("nan")
    if f_mat is not None:
        ok, h1, h2 = cv2.stereoRectifyUncalibrated(p_l.reshape(-1, 2), p_r.reshape(-1, 2), f_mat[:3], size)
        if ok:
            a = cv2.perspectiveTransform(p_l.reshape(-1, 1, 2).astype(np.float64), h1).reshape(-1, 2)
            b = cv2.perspectiveTransform(p_r.reshape(-1, 1, 2).astype(np.float64), h2).reshape(-1, 2)
            dy_uncal = float(np.mean(np.abs(a[:, 1] - b[:, 1])))

    pitch, yaw, roll = cv2.RQDecomp3x3(r_mat)[0]
    base = -r_mat.T @ t_unit  # вектор бази L → R у системі лівої камери
    f_rect = float(p1[0, 0])
    p1d, p50d, p99d = np.percentile(disp, [1, 50, 99])
    row = {"scene": scene, "left": left, "right": right, "is_primary": int(is_primary),
           "n_candidates": n_cand, "n_inliers": len(p_l), "pitch_deg": round(float(pitch), 3),
           "yaw_deg": round(float(yaw), 3), "roll_deg": round(float(roll), 3),
           "rot_total_deg": round(float(np.degrees(np.linalg.norm(cv2.Rodrigues(r_mat)[0]))), 3),
           "bx": round(float(base[0]), 4), "by": round(float(base[1]), 4), "bz": round(float(base[2]), 4),
           "f_rect_px": round(f_rect, 1), "dy_mean_px": round(float(dy.mean()), 3),
           "dy_std_px": round(float(dy.std(ddof=1)), 3), "dy_median_px": round(float(np.median(dy)), 3),
           "dy_p95_px": round(float(np.percentile(dy, 95)), 3), "dy_uncal_mean_px": round(dy_uncal, 3),
           "d_p1_px": round(float(p1d), 1), "d_p50_px": round(float(p50d), 1), "d_p99_px": round(float(p99d), 1),
           # Z = f'·B/d; B і поворот попередні, тож це орієнтир для порівняння з розмірами об'єктів
           "z_far_mm": round(f_rect * BASELINE_MM / p1d), "z_near_mm": round(f_rect * BASELINE_MM / p99d)}
    if save:
        save_primary(scene, left, right, size, k, dist, r_mat, t_mm, r1, r2, p1, p2, q)
    return row


def save_primary(scene, left, right, size, k, dist, r_mat, t_mm, r1, r2, p1, p2, q):
    """Зберігає параметри ректифікації, ректифіковані кадри й контрольний рисунок із сіткою."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(RESULTS_DIR / f"rectify_{scene}.npz", K=k, dist=dist, R=r_mat, T_mm=t_mm, R1=r1, R2=r2,
             P1=p1, P2=p2, Q=q, size=np.array(size), left=left, right=right, baseline_mm=BASELINE_MM)
    rect = []
    for stem, rot, proj in ((left, r1, p1), (right, r2, p2)):
        bgr, _ = load_frame(stem)
        mx, my = cv2.initUndistortRectifyMap(k, dist, rot, proj, size, cv2.CV_32FC1)
        rect.append(cv2.remap(bgr, mx, my, cv2.INTER_LINEAR))
    RECT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(RECT_DIR / f"{scene}_L.png"), rect[0])
    cv2.imwrite(str(RECT_DIR / f"{scene}_R.png"), rect[1])
    # Контроль (крок 5): відповідні деталі мають лежати на одній горизонтальній лінії.
    both = np.hstack(rect)
    for y in range(0, both.shape[0], GRID_STEP_PX):
        cv2.line(both, (0, y), (both.shape[1], y), (0, 255, 255), 1)
    cv2.putText(both, "L", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
    cv2.putText(both, "R", (size[0] + 20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
    small = cv2.resize(both, None, fx=GRID_FIG_SCALE, fy=GRID_FIG_SCALE, interpolation=cv2.INTER_AREA)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(FIGURES_DIR / f"rectify_grid_{scene}.png"), small)


def main():
    np_seed_all(SEED)
    primary = {r["scene"]: r for r in load_primary_pairs()}
    first = next(iter(primary.values()))
    _, gray = load_frame(first["left"])
    size = (gray.shape[1], gray.shape[0])
    k, dist = scaled_camera(size)
    print(f"Кадр {size[0]}x{size[1]}; fx={k[0, 0]:.1f}, fy={k[1, 1]:.1f}, cx={k[0, 2]:.1f}, cy={k[1, 2]:.1f} "
          f"(ЛР2, масштабовано); B = {BASELINE_MM} мм (попередня); alpha = {RECT_ALPHA}; "
          f"поріг E-RANSAC = {ESSENTIAL_THR_PX} px; крок сітки {GRID_STEP_PX} px")

    have = done_keys(COMBOS_CSV, ("scene", "left", "right"))
    for scene, prow in primary.items():
        lefts = prow["group_left"].split("|")
        rights = prow["group_right"].split("|")
        for left, right in itertools.product(lefts, rights):
            is_primary = left == prow["left"] and right == prow["right"]
            have_row = (scene, left, right) in have
            # основну пару рахуємо завжди: потрібні .npz і зображення для наступних частин
            if have_row and not is_primary:
                continue
            row = combo_row(scene, left, right, is_primary, size, k, dist, save=is_primary)
            if not have_row:
                append_row(COMBOS_CSV, FIELDS, row)
                print(f"{scene}: {left} – {right} готово")

    rows = read_rows(COMBOS_CSV)
    print("\n=== Комбінації дублів ===")
    print("сцена  ліва      права     inl   pitch   yaw    roll  |  bx     by     bz   | "
          "mean|dy| (σ)   uncal  | d p1/p50/p99, px | Z далекі/близькі, мм")
    for r in rows:
        mark = "*" if r["is_primary"] == "1" else " "
        print(f"{r['scene']}{mark}   {r['left']}  {r['right']}  {r['n_inliers']:>5}  {float(r['pitch_deg']):+6.2f} "
              f"{float(r['yaw_deg']):+6.2f} {float(r['roll_deg']):+6.2f} | {float(r['bx']):+.3f} {float(r['by']):+.3f} "
              f"{float(r['bz']):+.3f} | {float(r['dy_mean_px']):.2f} ({float(r['dy_std_px']):.2f})   "
              f"{float(r['dy_uncal_mean_px']):.2f}  | {float(r['d_p1_px']):.0f}/{float(r['d_p50_px']):.0f}/"
              f"{float(r['d_p99_px']):.0f} | {r['z_far_mm']}/{r['z_near_mm']}")
    print("(* – основна пара; mean|dy| і uncal – за автоматичними SIFT-відповідностями)")

    print("\n=== Розкид між комбінаціями дублів (середнє ± σ, n = кількість комбінацій) ===")
    for scene in primary:
        sel = [r for r in rows if r["scene"] == scene]
        if len(sel) < 2:
            continue
        yaw = np.array([float(r["yaw_deg"]) for r in sel])
        dmed = np.array([float(r["d_p50_px"]) for r in sel])
        f_rect = np.mean([float(r["f_rect_px"]) for r in sel])
        dy_mean = np.array([float(r["dy_mean_px"]) for r in sel])
        sig_yaw = yaw.std(ddof=1)
        print(f"Сцена {scene} (n={len(sel)}): yaw = {yaw.mean():+.2f} ± {sig_yaw:.2f}°, медіана d = "
              f"{dmed.mean():.1f} ± {dmed.std(ddof=1):.1f} px (σ yaw у диспаратності ≈ f'·σ = "
              f"{np.radians(sig_yaw) * f_rect:.1f} px), mean|dy| = {dy_mean.mean():.2f} ± {dy_mean.std(ddof=1):.2f} px")
        n_bad = int((dy_mean > RECT_VERT_TOL_PX).sum())
        print(f"   mean|dy| ≤ {RECT_VERT_TOL_PX} px (автоматичні точки): "
              f"{'так, у всіх комбінаціях' if n_bad == 0 else f'НІ у {n_bad} із {len(sel)}'}")
    print(f"\nГотово: {COMBOS_CSV.name}, rectify_<сцена>.npz, figures/rectify_grid_<сцена>.png")


if __name__ == "__main__":
    main()