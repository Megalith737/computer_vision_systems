"""Спільні константи й функції для ЛР4 (зіставлення ознак, RANSAC, панорама).

Усі параметри алгоритмів зібрані тут, щоб скрипти не містили «магічних» чисел.
Усі шляхи рахуються від теки lab_4 (а не від поточної директорії).
"""
import csv
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"          # оригінали кадрів (у git не йдуть)
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

SEED = 42

# ---------------------------------------------------------------- дані
MAX_SIDE = 1600        # кадри зменшуємо до 1600 px по довшій стороні (як у ЛР3)

# Усі 19 кадрів набору ЛР1. Для ЛР4 відбір за frame_stats.csv (як у ЛР3) не
# підходить: кадр 8738 (мало кутів) може бути ланкою панорами, а 8744 і темні
# 8747-8749 потрібні для випадків відмови. Серію вибираємо за аудитом пар.
FRAMES = tuple(f"IMG_{n}" for n in range(8730, 8750) if n != 8731)  # 8731 у наборі немає

# Серія для панорами (імена без розширення, у порядку зйомки зліва направо).
# Порожня, доки аудит пар (01_pair_audit.py) не покаже, які кадри сусідні.
SERIES = ("IMG_8735", "IMG_8736", "IMG_8737", "IMG_8738", "IMG_8739")

# ---------------------------------------------------------------- SIFT
SIFT_NFEATURES = 5000  # скільки найкращих за response точок лишає SIFT на кадр

# ---------------------------------------------------------------- зіставлення
RATIO_GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)  # пороги T відношення Лоу (практикум, крок 3)
AUDIT_RATIO = 0.75     # T лише для аудиту пар (01_pair_audit.py); у частині A T змінюється
MATCH_TAU = 3.0        # px; ТИМЧАСОВИЙ поріг RANSAC для аудиту й оцінки чистоти в частині A.
                       # Остаточне значення обґрунтуємо в частині C (дослідження τ).
MIN_PAIRS = 4          # менше 4 пар гомографію не оцінити

# Робочі умови для частин B, C, D (однакові для всіх пар серії). Обрані за
# результатами частини A (matching_T.csv): за T = 0.8 кількість узгоджених точок
# досягає ~86-89 % свого максимуму (T = 0.95) для трьох із чотирьох пар, а три
# пари з чотирьох мають частку узгоджених понад 40 % (орієнтир Додатка Г).
# Взаємна перевірка прибирає частину хибних пар. τ тут усе ще тимчасовий
# (MATCH_TAU): остаточне значення з'явиться в частині C.
WORK_RATIO = 0.8
WORK_MUTUAL = True

# Робочі умови частин B-E (однакові для всіх пар серії; обґрунтування в звіті).
T_WORK = 0.8           # поріг Лоу: за T = 0.8 найслабша пара серії має 135 кандидатів
                       # (99 після взаємної перевірки) проти 65 (58) за T = 0.7
USE_MUTUAL = True      # додатково вимагаємо взаємності
MIN_RELIABLE_CAND = 20 # якщо кандидатів менше, частка узгоджених не інтерпретується:
                       # 4 точки завжди ідеально задають H, тож 4 «узгоджені» є за визначенням

# Контроль на відомій гомографії (частина B): кадр деформуємо відомою H і
# перевіряємо, що конвеєр її відновлює. Критерії заданi ДО запуску.
CONTROL_H = dict(angle_deg=10.0, scale=0.9, persp=(2e-5, 1e-5))
CONTROL_MAX_RMS_PX = 1.0    # допустима СКВ відхилення відновленої H від істинної, px
CONTROL_MIN_SHARE = 0.9     # мінімальна частка узгоджених

# ---------------------------------------------------------------- рисунки
RAW_DRAW_RATIO = 0.95  # T, за якого малюємо «сирі» кандидати (найбільш забруднений набір)
MAX_DRAW = 150         # скільки ліній показуємо на рисунку (для читабельності)

# Правдоподібна гомографія: площа образу кадру відносно площі кадру
PLAUSIBLE_AREA = (0.2, 5.0)


# ================================================================ кадри
def find_frame(stem):
    """Шукає файл кадру в data/raw за іменем без розширення (jpg/jpeg/png,
    будь-який регістр розширення). Повертає Path або None."""
    for ext in (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"):
        p = RAW_DIR / (stem + ext)
        if p.exists():
            return p
    return None


_IMG_CACHE = {}


def load_frame(stem):
    """Повертає (bgr, gray) кадру, зменшеного до MAX_SIDE по довшій стороні.

    Читання через imdecode: cv2.imread не відкриває шляхи з кирилицею у Windows.
    EXIF-орієнтацію OpenCV застосовує сам. Результат кешується в пам'яті."""
    if stem in _IMG_CACHE:
        return _IMG_CACHE[stem]
    path = find_frame(stem)
    if path is None:
        raise FileNotFoundError(f"Немає кадру {stem} у {RAW_DIR}")
    data = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"Не вдалося прочитати {path}")
    scale = MAX_SIDE / max(bgr.shape[:2])
    if scale < 1.0:  # лише зменшуємо; INTER_AREA бо зменшення без аліасингу
        h, w = bgr.shape[:2]
        bgr = cv2.resize(bgr, (round(w * scale), round(h * scale)),
                         interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _IMG_CACHE[stem] = (bgr, gray)
    return bgr, gray


# ================================================================ SIFT і зіставлення
_SIFT = None
_FEAT_CACHE = {}


def sift_features(stem):
    """SIFT-точки й дескриптори кадру: (список KeyPoint, масив float32 N×128).
    Результат кешується, щоб не рахувати SIFT для того самого кадру повторно."""
    global _SIFT
    if stem not in _FEAT_CACHE:
        if _SIFT is None:
            _SIFT = cv2.SIFT_create(nfeatures=SIFT_NFEATURES)
        _, gray = load_frame(stem)
        _FEAT_CACHE[stem] = _SIFT.detectAndCompute(gray, None)
    return _FEAT_CACHE[stem]


_BF = cv2.BFMatcher(cv2.NORM_L2)  # евклідова метрика для SIFT (практикум, крок 2)


def knn_two(desc1, desc2):
    """Для кожного дескриптора desc1 – два найближчі сусіди в desc2 (k = 2)."""
    return _BF.knnMatch(desc1, desc2, k=2)


def ratio_filter(knn, t):
    """Критерій Лоу: лишаємо пару, якщо d(1-й сусід) < T · d(2-й сусід)."""
    return [p[0] for p in knn if len(p) == 2 and p[0].distance < t * p[1].distance]


def nearest_index(desc_query, desc_train):
    """Індекс найближчого дескриптора з desc_train для кожного рядка desc_query
    (потрібен для взаємної перевірки)."""
    return np.array([m.trainIdx for m in _BF.match(desc_query, desc_train)])


def mutual_filter(matches, reverse_nearest):
    """Взаємна перевірка: пара (i → j) лишається, якщо найближчий до j у першому
    наборі теж i. reverse_nearest[j] = індекс найближчого у першому наборі."""
    return [m for m in matches if reverse_nearest[m.trainIdx] == m.queryIdx]


def matches_to_points(matches, kp1, kp2):
    """Координати пар: src – у першому кадрі, dst – у другому (float32, N×2)."""
    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    return src, dst


def work_matches(a, b):
    """Робочий набір кандидатів пари кадрів (a, b): відношення Лоу з T = WORK_RATIO
    і, якщо WORK_MUTUAL, взаємна перевірка. Повертає (src, dst): координати в
    кадрах a і b (float32, N×2). Його беруть частини B, C, D, щоб умови були однакові."""
    kp_a, d_a = sift_features(a)
    kp_b, d_b = sift_features(b)
    cand = ratio_filter(knn_two(d_a, d_b), WORK_RATIO)
    if WORK_MUTUAL:
        cand = mutual_filter(cand, nearest_index(d_b, d_a))
    return matches_to_points(cand, kp_a, kp_b)


def reproj_errors(h, src, dst):
    """Похибка репроєкції кожної пари: відстань (px) між образом src під H і dst."""
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2).astype(np.float64),
                                    h.astype(np.float64)).reshape(-1, 2)
    return np.linalg.norm(proj - dst, axis=1)


def estimate_h(src, dst, tau=MATCH_TAU):
    """Гомографія src → dst бібліотечним RANSAC з порогом tau (px).
    Повертає (H, маска узгоджених bool) або (None, None), якщо пар замало
    чи оцінка не вдалася."""
    if len(src) < MIN_PAIRS:
        return None, None
    h, mask = cv2.findHomography(src, dst, cv2.RANSAC, tau)
    if h is None:
        return None, None
    return h, mask.ravel().astype(bool)


def overlap_fraction(h, size_a, size_b):
    """Частка кадру B, перекрита образом кадру A за гомографією H (A → B).

    Чотири кути A переносимо через H й перетинаємо чотирикутник з прямокутником B.
    Повертає (частка, правдоподібність). Якщо чотирикутник невипуклий або його
    площа поза PLAUSIBLE_AREA, гомографія вважається нерозумною (NaN, False)."""
    wa, ha = size_a
    wb, hb = size_b
    corners = np.float32([[0, 0], [wa, 0], [wa, ha], [0, ha]]).reshape(-1, 1, 2)
    quad = cv2.perspectiveTransform(corners, h).reshape(-1, 2).astype(np.float32)
    if not cv2.isContourConvex(quad.reshape(-1, 1, 2)):
        return float("nan"), False
    area_ratio = cv2.contourArea(quad) / (wa * ha)
    if not (PLAUSIBLE_AREA[0] <= area_ratio <= PLAUSIBLE_AREA[1]):
        return float("nan"), False
    rect = np.float32([[0, 0], [wb, 0], [wb, hb], [0, hb]])
    inter_area, _ = cv2.intersectConvexConvex(quad, rect)
    return float(inter_area) / (wb * hb), True


# ================================================================ CSV (запис рядок за рядком)
def done_keys(path, key_cols):
    """Множина ключів (кортежів рядків) уже записаних рядків CSV: щоб після
    переривання не рахувати те саме знову."""
    if not Path(path).exists():
        return set()
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {tuple(r[c] for c in key_cols) for r in csv.DictReader(f)}


def append_row(path, fieldnames, row):
    """Дописує один рядок у CSV (створює файл із заголовком, якщо його немає).
    Файл пишеться з BOM (utf-8-sig), щоб Excel правильно читав кирилицю."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "w" if new else "a", encoding="utf-8-sig" if new else "utf-8",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new:
            w.writeheader()
        w.writerow(row)