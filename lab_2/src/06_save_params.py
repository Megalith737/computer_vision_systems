"""Крок 15: запис camera_params.yml (потрібен у ЛР5).

Запуск: python src/06_save_params.py
Калібрує на кадрах без викидів і записує через cv2.FileStorage: матрицю камери,
коефіцієнти дисторсії, розмір зображення, фінальну похибку, кількість кадрів, дату.
"""
import datetime

import cv2
import numpy as np

from common import CELL_MM, ROOT, SEED, calibrate, load_points

np.random.seed(SEED)


def main():
    names, _, pts, size = load_points(only_inliers=True)
    rms, K, dist, _, _ = calibrate(pts, size)

    out = ROOT / "camera_params.yml"
    fs = cv2.FileStorage(str(out), cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", K)
    fs.write("dist_coeffs", dist)
    fs.write("image_width", size[0])
    fs.write("image_height", size[1])
    fs.write("reprojection_error_px", float(rms))
    fs.write("num_frames", len(names))
    fs.write("square_size_mm", CELL_MM)
    fs.write("calibration_date", datetime.date.today().isoformat())
    fs.release()

    print(f"Записано {out}: {len(names)} кадрів, RMS {rms:.3f} px")
    print("K =\n", np.round(K, 2), "\ndist =", np.round(dist.ravel(), 5))


if __name__ == "__main__":
    main()