"""Детектор Гарріса «власноруч» (ЛР3, частина A).

Конвеєр: Sobel -> Ix^2, Iy^2, Ix*Iy -> гаусіан -> R = det - k*tr^2
         -> поріг THRESH_REL*max(R) -> власне NMS (радіус, N найкращих).
Із бібліотеки OpenCV беремо лише Sobel і GaussianBlur як операції згортки;
сам відгук R і придушення немаксимумів написані тут.
"""
import cv2
import numpy as np

from common import K, N_POINTS, NMS_RADIUS, SIGMA, THRESH_REL


def harris_response(gray, sigma=SIGMA, k=K, box=None):
    """Карта відгуку Гарріса R (float32) того ж розміру, що й зображення.

    box: якщо задано (напр. 5), замість гаусіана береться рівне вікно box x box,
    як у cv2.cornerHarris. Потрібно лише для контрольного порівняння в частині A.

    ВАЖЛИВО: спочатку добутки градієнтів (Ix^2, Iy^2, Ix*Iy), і лише потім
    гаусове згладжування. Згладити Ix, Iy перед піднесенням до квадрата - типова
    помилка: тоді матриця M вироджується (det -> 0) і кутів не видно.
    """
    g = gray.astype(np.float32)
    ix = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    iy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)

    ixx, iyy, ixy = ix * ix, iy * iy, ix * iy
    if box:
        smooth = lambda a: cv2.boxFilter(a, -1, (box, box), normalize=True)
    else:
        smooth = lambda a: cv2.GaussianBlur(a, (0, 0), sigma)
    sxx, syy, sxy = smooth(ixx), smooth(iyy), smooth(ixy)

    det = sxx * syy - sxy * sxy
    trace = sxx + syy
    return det - k * trace * trace


def nms_topn(response, thresh_rel=THRESH_REL, radius=NMS_RADIUS, n_points=N_POINTS):
    """Жадібне придушення немаксимумів.

    1. Кандидати: R > thresh_rel * max(R).
    2. Сортуємо за спаданням R.
    3. Беремо кандидата, якщо його ще не придушено, і придушуємо все в
       радіусі `radius` (евклідова відстань) навколо нього.
    4. Зупиняємось на n_points точках.

    Повертає масив (M, 3): x, y, R, відсортований за спаданням R (M <= n_points).
    """
    h, w = response.shape
    thr = thresh_rel * float(response.max())
    ys, xs = np.nonzero(response > thr)
    if len(xs) == 0:
        return np.zeros((0, 3), np.float32)

    order = np.argsort(-response[ys, xs], kind="stable")
    ys, xs = ys[order], xs[order]

    # диск радіуса r: маска зсувів (dy, dx), для яких dy^2 + dx^2 <= r^2
    r = int(radius)
    dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
    disk = (dy * dy + dx * dx) <= radius * radius

    suppressed = np.zeros((h, w), dtype=bool)
    kept = []
    for y, x in zip(ys, xs):
        if suppressed[y, x]:
            continue
        kept.append((x, y, response[y, x]))
        if len(kept) >= n_points:
            break
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        suppressed[y0:y1, x0:x1] |= disk[y0 - (y - r):y1 - (y - r),
                                         x0 - (x - r):x1 - (x - r)]
    return np.asarray(kept, dtype=np.float32)


def harris_points(gray, **kwargs):
    """Зручна обгортка: відгук + NMS. Повертає (M, 2) з координатами x, y."""
    pts = nms_topn(harris_response(gray, **{k: v for k, v in kwargs.items()
                                            if k in ("sigma", "k")}),
                   **{k: v for k, v in kwargs.items()
                      if k in ("thresh_rel", "radius", "n_points")})
    return pts[:, :2]