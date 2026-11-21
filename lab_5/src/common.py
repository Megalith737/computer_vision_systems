"""Спільні константи й функції для ЛР5 (стереопара: ректифікація, диспаратність, глибина).

Усі параметри алгоритмів зібрані тут, щоб скрипти не містили «магічних» чисел.
Усі шляхи рахуються від теки lab_5 (а не від поточної директорії).
"""
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"          # оригінали кадрів (у git не йдуть)
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
CAMERA_PARAMS_FILE = ROOT.parent / "lab_2" / "camera_params.yml"  # результат ЛР2

SEED = 42

# ---------------------------------------------------------------- дані
# Сцени: T – вимірювальна (підвіконня з банками, пляшками), F – сцена відмов (ТВ, штора, стіл).
# У кожній сцені 2 позиції камери (ліва й права) по 2 дублі кадру. Яка позиція ліва, а яка
# права, і які кадри дублі, визначає 01_pair_audit.py (тут це не задається вручну).
SCENES = {
    "T": tuple(f"IMG_{n}" for n in range(9549, 9553)),
    "F": tuple(f"IMG_{n}" for n in range(9553, 9557)),
}
FRAMES = tuple(f for frames in SCENES.values() for f in frames)

MAX_SIDE = 1600        # кадри зменшуємо до 1600 px по довшій стороні (як у ЛР3 і ЛР4): 1200x1600

# База стереопари, мм. ПОПЕРЕДНЯ: зі слів автора («мабуть 8.1 см», лінійка); ще не уточнена.
# У частині D перевіряємо її за відомими розмірами об'єктів.
BASELINE_MM = 81.0

# ---------------------------------------------------------------- SIFT і зіставлення (аудит)
SIFT_NFEATURES = 5000   # скільки найкращих за response точок лишає SIFT на кадр (як у ЛР4)
AUDIT_RATIO = 0.75      # поріг відношення Лоу для аудиту пар (як у ЛР4)
F_RANSAC_THR_PX = 1.0   # поріг RANSAC для фундаментальної матриці, px
F_RANSAC_CONF = 0.999   # бажана ймовірність знайти чисту вибірку

# ---------------------------------------------------------------- класи пар кадрів
DUP_MAX_SHIFT_PX = 8.0   # медіана |dx|, |dy| не більша: дубль (той самий ракурс), px у кадрі MAX_SIDE
STEREO_MIN_INLIERS = 300 # узгоджених пар не менше: та сама сцена; менше: різні сцени
CONTROL_MIN_FRACTION = 0.99  # контроль: частка кандидатів із нульовим зсувом кадру з самим собою

# ---------------------------------------------------------------- ректифікація (частина B)
RECT_DIR = ROOT / "data" / "rectified"  # ректифіковані кадри (PNG, у git не йдуть)
ESSENTIAL_THR_PX = 1.0   # поріг RANSAC для істотної матриці, px (у нормованих координатах ділиться на fx)
ESSENTIAL_CONF = 0.999   # бажана ймовірність знайти чисту вибірку
RECT_ALPHA = 0           # stereoRectify: 0 – лишаємо лише дійсні пікселі (без чорних смуг), 1 – усі
GRID_STEP_PX = 25        # крок горизонтальних ліній на контрольному рисунку (практикум, крок 5)
GRID_FIG_SCALE = 0.5     # у скільки разів зменшуємо контрольний рисунок (щоб файл був невеликий)
RECT_VERT_TOL_PX = 1.0   # допуск вертикальної неузгодженості після ректифікації (практикум, крок 6)


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
        return _IMG_CACHE[stem][:2]
    path = find_frame(stem)
    if path is None:
        raise FileNotFoundError(f"Немає кадру {stem} у {RAW_DIR}")
    data = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"Не вдалося прочитати {path}")
    native = (bgr.shape[1], bgr.shape[0])  # (ширина, висота) оригіналу з урахуванням EXIF-орієнтації
    scale = MAX_SIDE / max(bgr.shape[:2])
    if scale < 1.0:  # лише зменшуємо; INTER_AREA бо зменшення без аліасингу
        h, w = bgr.shape[:2]
        bgr = cv2.resize(bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _IMG_CACHE[stem] = (bgr, gray, native)
    return bgr, gray


def native_size(stem):
    """Розмір оригіналу (ширина, висота) у пікселях."""
    load_frame(stem)
    return _IMG_CACHE[stem][2]


def sharpness(gray):
    """Різкість кадру: дисперсія лапласіана (більше = різкіше). Порівнювати можна
    лише кадри того самого ракурсу (дублі): на різному вмісті значення не порівнянні."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def read_exif(stem):
    """Вибрані EXIF-теги кадру (None, якщо тегу немає або файл не читається через PIL)."""
    out = {"exif_focal_mm": None, "exif_focal35_mm": None, "exif_exposure_s": None,
           "exif_iso": None, "exif_lens": None}
    path = find_frame(stem)
    try:
        with Image.open(path) as im:
            sub = im.getexif().get_ifd(0x8769)  # блок Exif IFD
    except Exception:
        return out

    def num(v):
        if isinstance(v, (tuple, list)):
            v = v[0] if v else None
        return None if v is None else float(v)

    out["exif_focal_mm"] = num(sub.get(0x920A))      # FocalLength
    out["exif_focal35_mm"] = num(sub.get(0xA405))    # FocalLengthIn35mmFilm
    out["exif_exposure_s"] = num(sub.get(0x829A))    # ExposureTime
    out["exif_iso"] = num(sub.get(0x8827))           # ISO
    out["exif_lens"] = sub.get(0xA434)               # LensModel
    return out


def load_camera_params():
    """Параметри камери з lab_2/camera_params.yml: (K, dist, (ширина, висота) калібрування).
    K і dist належать розміру кадру калібрування; для іншого розміру fx, fy, cx, cy масштабуються
    так само, як зображення, а коефіцієнти дисторсії не змінюються."""
    fs = cv2.FileStorage(str(CAMERA_PARAMS_FILE), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Немає {CAMERA_PARAMS_FILE}")
    k = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("dist_coeffs").mat()
    size = (int(fs.getNode("image_width").real()), int(fs.getNode("image_height").real()))
    fs.release()
    return k, dist, size


# ================================================================ SIFT і зіставлення
_SIFT = None
_FEAT_CACHE = {}


def sift_features(stem):
    """SIFT-точки й дескриптори кадру: (список KeyPoint, масив float32 N×128). Кешується."""
    global _SIFT
    if stem not in _FEAT_CACHE:
        if _SIFT is None:
            _SIFT = cv2.SIFT_create(nfeatures=SIFT_NFEATURES)
        _, gray = load_frame(stem)
        _FEAT_CACHE[stem] = _SIFT.detectAndCompute(gray, None)
    return _FEAT_CACHE[stem]


_BF = cv2.BFMatcher(cv2.NORM_L2)  # евклідова метрика для SIFT


def ratio_matches(desc1, desc2, t=AUDIT_RATIO):
    """Критерій Лоу: лишаємо пару, якщо d(1-й сусід) < T · d(2-й сусід)."""
    knn = _BF.knnMatch(desc1, desc2, k=2)
    return [p[0] for p in knn if len(p) == 2 and p[0].distance < t * p[1].distance]


def matches_to_points(matches, kp1, kp2):
    """Координати пар: src – у першому кадрі, dst – у другому (float32, N×2)."""
    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    return src, dst


# ================================================================ CSV (запис рядок за рядком)
def done_keys(path, key_cols):
    """Множина ключів (кортежів рядків) уже записаних рядків CSV: щоб після
    переривання не рахувати те саме знову."""
    if not Path(path).exists():
        return set()
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {tuple(r[c] for c in key_cols) for r in csv.DictReader(f)}


def read_rows(path):
    """Усі рядки CSV як список словників (порожній список, якщо файлу немає)."""
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def append_row(path, fieldnames, row):
    """Дописує один рядок у CSV (створює файл із заголовком, якщо його немає).
    Файл пишеться з BOM (utf-8-sig), щоб Excel правильно читав кирилицю."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "w" if new else "a", encoding="utf-8-sig" if new else "utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new:
            w.writeheader()
        w.writerow(row)


def np_seed_all(seed=SEED):
    """Фіксує випадковий стан numpy і OpenCV (RANSAC в OpenCV бере випадкові числа з власного генератора)."""
    np.random.seed(seed)
    cv2.setRNGSeed(seed)


# ================================================================ камера й основні пари
def scaled_camera(work_size):
    """K і dist для кадру розміру work_size = (ширина, висота).

    Калібрування ЛР2 зроблено на кадрі calib_size. Для іншого розміру з тим самим відношенням сторін
    fx, fy, cx, cy масштабуються так само, як зображення (центр пікселя в OpenCV – у (0.5, 0.5),
    тому cx, cy переводимо через (c + 0.5)·s − 0.5); коефіцієнти дисторсії не змінюються."""
    k0, dist, (cw, ch) = load_camera_params()
    sx, sy = work_size[0] / cw, work_size[1] / ch
    if abs(sx - sy) > 0.002 * sx:
        raise ValueError(f"Відношення сторін кадру {work_size} не збігається з калібруванням {(cw, ch)}")
    k = k0.copy()
    k[0, 0] *= sx
    k[1, 1] *= sx
    k[0, 2] = (k0[0, 2] + 0.5) * sx - 0.5
    k[1, 2] = (k0[1, 2] + 0.5) * sx - 0.5
    return k, dist


def load_primary_pairs():
    """Основні пари сцен із results/primary_pairs.csv (частина A): список словників."""
    rows = read_rows(RESULTS_DIR / "primary_pairs.csv")
    if not rows:
        raise SystemExit("Немає results/primary_pairs.csv. Спершу запустіть src/01_pair_audit.py")
    return rows