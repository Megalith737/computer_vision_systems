"""Кроки 13-14: усунення дисторсії, alpha = 0 і alpha = 1, порівняння «до/після».

Запуск: python src/05_undistort.py
Для кожного кадра line_*.jpg відкривається вікно: клацни по двох кінцях прямого
краю предмета (ближче до країв кадра). Відрізок між точками малюється червоним на
оригіналі й (з перенесеними точками) на виправлених зображеннях.
Результат: figures/undistort_<кадр>.jpg - панель «оригінал | alpha=0 | alpha=1».
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import FIGURES_DIR, FRAMES_DIR, SEED, calibrate, load_gray, load_points

np.random.seed(SEED)


def pick_segment(img):
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title("Клацни два кінці прямого краю")
    pts = plt.ginput(2, timeout=0)
    plt.close(fig)
    return np.array(pts, np.float32)


def draw_segment(img, seg):
    out = img.copy()
    cv2.line(out, tuple(int(v) for v in seg[0]), tuple(int(v) for v in seg[1]),
             (0, 0, 255), max(3, img.shape[1] // 600), cv2.LINE_AA)
    return out


def label(img, text):
    scale = img.shape[1] / 1000
    out = img.copy()
    cv2.putText(out, text, (20, int(60 * scale) + 20), cv2.FONT_HERSHEY_SIMPLEX,
                1.5 * scale, (0, 255, 255), max(3, int(4 * scale)), cv2.LINE_AA)
    return out


def main():
    _, _, pts, size = load_points(only_inliers=True)
    rms, K, dist, _, _ = calibrate(pts, size)
    new_K = {}
    for alpha in (0, 1):
        new_K[alpha], roi = cv2.getOptimalNewCameraMatrix(K, dist, size, alpha, size)
        print(f"alpha={alpha}: roi (x, y, w, h) = {roi}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(FRAMES_DIR.glob("line_*.jpg")) + [FRAMES_DIR / "cal_01.jpg"]:
        img, _ = load_gray(path, size)
        seg = pick_segment(img) if path.name.startswith("line_") else None
        panels = [label(draw_segment(img, seg) if seg is not None else img, "original")]
        for alpha in (0, 1):
            und = cv2.undistort(img, K, dist, None, new_K[alpha])
            if seg is not None:
                mapped = cv2.undistortPoints(seg.reshape(-1, 1, 2), K, dist,
                                             P=new_K[alpha]).reshape(2, 2)
                und = draw_segment(und, mapped)
            panels.append(label(und, f"alpha={alpha}"))
        row = np.hstack(panels)
        row = cv2.resize(row, None, fx=2400 / row.shape[1], fy=2400 / row.shape[1],
                         interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(FIGURES_DIR / f"undistort_{path.stem}.jpg"), row)
        print("збережено", f"undistort_{path.stem}.jpg")


if __name__ == "__main__":
    main()