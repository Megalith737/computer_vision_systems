"""Крок 6: пошук кутів шахівниці 9x6 на калібрувальних кадрах.

Запуск: python src/01_corners.py
Результат: results/corners.npz (кути для наступних скриптів),
           figures/corners/<кадр>.jpg (кадри з намальованими кутами).
"""
from collections import Counter

import cv2
import numpy as np

from common import (FIGURES_DIR, PATTERN, SEED, find_corners, frame_paths,
                    load_gray, save_detections)

np.random.seed(SEED)


def main():
    paths = frame_paths()
    if not paths:
        raise SystemExit("У data/frames немає файлів cal_*.jpg")
    out_dir = FIGURES_DIR / "corners"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Кадри мають однаковий розмір: якщо ні - стискаємо до найпоширенішого
    native = {p.name: load_gray(p)[1].shape[::-1] for p in paths}
    size = Counter(native.values()).most_common(1)[0][0]
    for name, s in native.items():
        if s != size:
            print(f"{name}: розмір {s[0]}x{s[1]} стиснуто до {size[0]}x{size[1]}")

    names, corners, methods = [], [], []
    for path in paths:
        img, gray = load_gray(path, size)
        ok, c, method = find_corners(gray)
        print(f"{path.name}: {'знайдено' if ok else 'НЕ ЗНАЙДЕНО':12s} метод={method}")
        if ok:
            names.append(path.name)
            corners.append(c)
            methods.append(method)
            cv2.drawChessboardCorners(img, PATTERN, c, True)
            cv2.imwrite(str(out_dir / path.name), img)

    save_detections(names, corners, methods, size)
    print(f"\nкадрів {len(paths)}, знайдено {len(names)}; "
          f"за методом {dict(Counter(methods))}; розмір {size[0]}x{size[1]}")


if __name__ == "__main__":
    main()