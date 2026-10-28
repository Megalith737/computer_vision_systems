"""Спільні константи й функції для ЛР2 (калібрування камери)."""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = ROOT / "data" / "frames"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
CORNERS_FILE = RESULTS_DIR / "corners.npz"
INLIERS_FILE = RESULTS_DIR / "inliers.txt"

SEED = 42
PATTERN = (9, 6)   # внутрішніх кутів: 9 по горизонталі, 6 по вертикалі (10x7 клітин)
CELL_MM = 29.8     # виміряний розмір клітини, мм (238 мм / 8 і 149 мм / 5)

FIND_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
SUBPIX_WIN = (11, 11)   # половина вікна пошуку: вікно 23x23 px
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def frame_paths():
    """Калібрувальні кадри data/frames/cal_*.jpg за порядком."""
    return sorted(FRAMES_DIR.glob("cal_*.jpg"))


def load_gray(path, size=None):
    """Повертає (BGR, відтінки сірого). Якщо задано size=(w, h) і кадр має інший
    розмір, його стискають до size (кадри 24 Мп і 12 Мп мають однакове
    співвідношення сторін)."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Не вдалося прочитати {path}")
    if size is not None and (img.shape[1], img.shape[0]) != tuple(size):
        img = cv2.resize(img, tuple(size), interpolation=cv2.INTER_AREA)
    return img, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def find_corners(gray):
    """Крок 6. Повертає (знайдено, кути (54,1,2) float32, метод).

    "classic" - cv2.findChessboardCorners; "clahe" - те саме після розмиття й
    вирівнювання контрасту (CLAHE), якщо перша спроба не вдалася; "SB" -
    cv2.findChessboardCornersSB (дає вже субпіксельні кути)."""
    ok, corners = cv2.findChessboardCorners(gray, PATTERN, FIND_FLAGS)
    if ok:
        return True, corners.reshape(-1, 1, 2).astype(np.float32), "classic"
    boosted = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(
        cv2.GaussianBlur(gray, (5, 5), 0))
    ok, corners = cv2.findChessboardCorners(boosted, PATTERN, FIND_FLAGS)
    if ok:
        return True, corners.reshape(-1, 1, 2).astype(np.float32), "clahe"
    ok, corners = cv2.findChessboardCornersSB(gray, PATTERN, 0)
    if ok:
        return True, corners.reshape(-1, 1, 2).astype(np.float32), "SB"
    return False, None, "none"


def save_detections(names, corners, methods, size):
    CORNERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CORNERS_FILE, names=np.array(names), corners=np.stack(corners),
             methods=np.array(methods), size=np.array(size))


def load_detections():
    """Читає кути, збережені скриптом 01: (імена, кути, методи, (w, h))."""
    if not CORNERS_FILE.exists():
        raise SystemExit(f"Немає {CORNERS_FILE}. Спершу запусти src/01_corners.py")
    data = np.load(CORNERS_FILE)
    return (list(data["names"]), list(data["corners"]), list(data["methods"]),
            tuple(int(v) for v in data["size"]))


def object_points():
    """Крок 8. Координати кутів шаблона, мм: (0,0,0), (s,0,0), ..., Z = 0. Форма (54, 3)."""
    grid = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    grid[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * CELL_MM
    return grid


def refine(names, corners, size):
    """Крок 7. Субпіксельне уточнення кутів (cv2.cornerSubPix)."""
    refined = []
    for name, c in zip(names, corners):
        _, gray = load_gray(FRAMES_DIR / name, size)
        out = cv2.cornerSubPix(gray, np.asarray(c, np.float32).copy(), SUBPIX_WIN,
                               (-1, -1), SUBPIX_CRITERIA)
        refined.append(out.reshape(-1, 1, 2).astype(np.float32))
    return refined


def calibrate(img_points, size):
    """Крок 9. Повертає (rms, K, dist, rvecs, tvecs)."""
    objp = object_points()
    return cv2.calibrateCamera([objp] * len(img_points), img_points, size, None, None)


def load_points(only_inliers=False):
    """Кути для калібрування: уточнені cornerSubPix (кадри SB - як є).
    only_inliers=True бере лише кадри з results/inliers.txt (після кроку 11).
    Повертає (імена, методи, кути, (w, h))."""
    names, corners, methods, size = load_detections()
    idx = list(range(len(names)))
    if only_inliers and INLIERS_FILE.exists():
        keep = set(INLIERS_FILE.read_text(encoding="utf-8").split())
        idx = [i for i in idx if names[i] in keep]
    todo = [i for i in idx if methods[i] != "SB"]
    refined = dict(zip(todo, refine([names[i] for i in todo],
                                    [corners[i] for i in todo], size)))
    pts = [refined.get(i, corners[i]) for i in idx]
    return [names[i] for i in idx], [methods[i] for i in idx], pts, size