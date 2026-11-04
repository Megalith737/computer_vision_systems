"""Спільні константи й функції для ЛР3 (детектори локальних ознак)."""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GRAY_DIR = ROOT / "data" / "prepared" / "gray"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

SEED = 42

# Параметри Гарріса (практикум, частина A)
SIGMA = 1.5        # σ гауссіана для згладжування добутків градієнтів
K = 0.04           # коефіцієнт у R = det - k * tr^2
THRESH_REL = 0.01  # поріг відгуку: R > THRESH_REL * max(R)
NMS_RADIUS = 10    # радіус придушення немаксимумів, px
N_POINTS = 500     # скільки найкращих точок лишаємо (однаково для всіх детекторів)
MATCH_EPS = 2.0    # допуск збігу з cv2.cornerHarris у частині A, px


def image_paths():
    """Усі підготовлені сірі кадри з data/prepared/gray/ за порядком імен."""
    return sorted(GRAY_DIR.glob("*.png"))


def load_gray(path):
    """Читає PNG як відтінки сірого (uint8). Через imdecode, щоб працювали шляхи
    з кирилицею у Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Не вдалося прочитати {path}")
    return img