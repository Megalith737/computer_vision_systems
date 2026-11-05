"""Три детектори з однаковим інтерфейсом і метрика повторюваності (ЛР3, частина D).

Умови чесного порівняння (зафіксовані тут і в common.py):
  - рівно N_POINTS найкращих точок за response для КОЖНОГО детектора;
  - один допуск EPS для всіх;
  - детектори бачать лише внутрішню частину кадру (маска), щоб чорні кути
    після повороту й межа кадру не давали штучних точок.
"""
import cv2
import numpy as np
from scipy.spatial import cKDTree

from common import N_POINTS
from harris import harris_response, nms_topn

EPS = 3.0      # допуск збігу, px (в системі координат перетвореного кадру)
MARGIN = 15    # відступ від межі видимої області, px
ORB_POOL = 5000


def inner_mask(valid, margin=MARGIN):
    """Маска (uint8, 1 = можна шукати точки): видима область, відсунута від межі.
    Межу самого кадру теж вважаємо межею (паддінг нулями перед ерозією)."""
    pad = cv2.copyMakeBorder(valid, margin, margin, margin, margin,
                             cv2.BORDER_CONSTANT, value=0)
    eroded = cv2.erode(pad, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
    return eroded[margin:-margin, margin:-margin].copy()


def _top_n(keypoints, n=N_POINTS):
    order = sorted(range(len(keypoints)), key=lambda i: -keypoints[i].response)
    return np.array([keypoints[i].pt for i in order[:n]], dtype=np.float64).reshape(-1, 2)


def detect_harris(gray, mask):
    r = harris_response(gray)
    r[mask == 0] = 0.0
    return nms_topn(r)[:, :2].astype(np.float64)


_SIFT = None
_ORB = None


def detect_sift(gray, mask):
    global _SIFT
    if _SIFT is None:
        _SIFT = cv2.SIFT_create()
    return _top_n(_SIFT.detect(gray, mask))


def detect_orb(gray, mask):
    global _ORB
    if _ORB is None:
        _ORB = cv2.ORB_create(nfeatures=ORB_POOL)
    return _top_n(_ORB.detect(gray, mask))


DETECTORS = {"Harris": detect_harris, "SIFT": detect_sift, "ORB": detect_orb}


def repeatability(p1, p2, H, valid_inner, eps=EPS):
    """Повторюваність точок p1 (оригінал) і p2 (перетворений кадр) при відомій H.

    1. P1 переносимо матрицею H у систему перетвореного кадру.
    2. Лишаємо лише ті з них, що потрапили у видиму область (valid_inner);
       решта точок «випала» за кадр і порівнюватись не може.
    3. a = скільки перенесених P1 мають у P2 сусіда ближче за eps;
       b = скільки точок P2 мають поблизу перенесену P1.
    4. matches = min(a, b), щоб кілька близьких точок не рахувались двічі, і
       повторюваність = matches / min(|P1_vis|, |P2|) <= 1.

    Повертає (n1_vis, n2, matches, repeatability).
    """
    if len(p1) == 0 or len(p2) == 0:
        return 0, len(p2), 0, 0.0
    q = cv2.perspectiveTransform(p1.reshape(-1, 1, 2), H).reshape(-1, 2)
    h2, w2 = valid_inner.shape
    xi, yi = np.round(q[:, 0]).astype(int), np.round(q[:, 1]).astype(int)
    inside = (xi >= 0) & (xi < w2) & (yi >= 0) & (yi < h2)
    vis = np.zeros(len(q), dtype=bool)
    vis[inside] = valid_inner[yi[inside], xi[inside]] > 0
    q = q[vis]
    if len(q) == 0:
        return 0, len(p2), 0, 0.0
    a = int((cKDTree(p2).query(q)[0] <= eps).sum())
    b = int((cKDTree(q).query(p2)[0] <= eps).sum())
    matches = min(a, b)
    return len(q), len(p2), matches, matches / min(len(q), len(p2))