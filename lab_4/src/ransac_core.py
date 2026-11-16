"""Власний RANSAC для гомографії (ЛР4, частина C; використовується й у частині D).

Схема за практикумом (крок 8):
  1. повторити N разів: випадково вибрати 4 пари, оцінити H за ними
     (cv2.getPerspectiveTransform), перенести всі точки кадру A, порахувати,
     скільки пар мають відстань до відповідника менше τ; якщо більше, ніж у
     найкращої, запам'ятати;
  2. переоцінити H за всіма узгодженими точками методом найменших квадратів
     (cv2.findHomography з методом 0).
"""
import cv2
import numpy as np

from common import RANSAC_N_ITER, SEED, reproj_errors


def _draw_samples(rng, n, n_iter):
    """Повертає масив n_iter×4 індексів мінімальних вибірок БЕЗ повторів усередині вибірки.

    Тягнемо з запасом (rng.integers), відкидаємо рядки з однаковими індексами
    (після сортування сусідні значення рівні) і добираємо, доки не набереться n_iter."""
    out = np.empty((0, 4), dtype=np.int64)
    while len(out) < n_iter:
        idx = rng.integers(0, n, size=(int(n_iter * 1.3) + 16, 4))
        ok = (np.diff(np.sort(idx, axis=1), axis=1) > 0).all(axis=1)
        out = np.vstack([out, idx[ok]])
    return out[:n_iter]


def ransac_homography(src, dst, tau, n_iter=RANSAC_N_ITER, seed=SEED):
    """RANSAC для гомографії src → dst з порогом tau (px) і n_iter ітераціями.

    Повертає (H, маска узгоджених bool) після уточнення МНК за найбільшою
    узгодженою множиною; маска перераховується для уточненої H з тим самим tau.
    Якщо пар менше 4 або оцінка не вдалася, повертає (None, None)."""
    n = len(src)
    if n < 4:
        return None, None
    src = np.asarray(src, dtype=np.float32)
    dst = np.asarray(dst, dtype=np.float32)
    rng = np.random.default_rng(seed)           # окремий генератор: результат відтворюваний
    src_c = src.reshape(-1, 1, 2).astype(np.float64)
    dst64 = dst.astype(np.float64)
    tau2 = tau * tau                            # порівнюємо квадрати відстаней: без кореня

    best_cnt, best_mask = -1, None
    # Виродженій вибірці (колінеарні точки) відповідає вироджена H: її образи
    # NaN/inf, порівняння з tau2 для них хибне, тому такі вибірки самі «програють».
    with np.errstate(all="ignore"):
        for idx in _draw_samples(rng, n, n_iter):
            h = cv2.getPerspectiveTransform(src[idx], dst[idx])
            proj = cv2.perspectiveTransform(src_c, h).reshape(-1, 2)
            mask = np.sum((proj - dst64) ** 2, axis=1) < tau2
            cnt = int(mask.sum())
            if cnt > best_cnt:                  # строго більше, як у практикумі
                best_cnt, best_mask = cnt, mask

    if best_mask is None or best_cnt < 4:
        return None, None
    h_ref, _ = cv2.findHomography(src[best_mask], dst[best_mask], 0)   # МНК за узгодженими
    if h_ref is None:
        return None, None
    return h_ref, reproj_errors(h_ref, src, dst) < tau