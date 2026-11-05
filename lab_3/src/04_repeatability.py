"""ЛР3, частина D: експеримент «повторюваність - параметр перетворення».

Для кожного кадру, кожного перетворення з сітки і кожного з трьох детекторів
(Гарріс власний, SIFT, ORB) рахуємо повторюваність точок за відомою матрицею H.
Результат дописується в results/repeatability.csv рядок за рядком, тож
скрипт можна перервати й запустити знову: готові рядки пропускаються.

Сітка (як у методичці):
  rotation    0..90 градусів, крок 10      (поворот навколо центру, полотно те саме)
  scale       0.5..2.0, крок 0.25          (полотно змінюється разом із масштабом)
  brightness  множник 0.5..1.5, крок 0.1
  noise       гауссів шум σ = 0, 5, 10, 15, 20 (у рівнях яскравості 0..255)
Додатково (самостійна частина, лише з --series blur contrast):
  blur        гауссове розмиття σ = 0..4 px
  contrast    множник контрасту відносно середньої яскравості 0.5..1.5

Контроль: «нульове» перетворення (0°, ×1.0, шум 0 ...) мусить давати 100 %.
Скрипт попереджає, якщо це не так.

Масштаб: зменшуємо cv2.resize(INTER_AREA), збільшуємо INTER_LINEAR. Матриця
H = [[s,0,(s-1)/2],[0,s,(s-1)/2],[0,0,1]] відповідає центрам пікселів, тож
перенесення точок точне. (warpPerspective без передфільтрації при зменшенні дає
аліасинг, що додатково псує всі детектори.)

Запуск (з кореня lab_3/):
  python src/04_repeatability.py                    # усі 4 серії, усі кадри
  python src/04_repeatability.py --series rotation  # лише одна серія
  python src/04_repeatability.py --max-images 1     # швидка перевірка
"""
import argparse
import csv
import time

import cv2
import numpy as np

from common import RESULTS_DIR, SEED, image_paths, load_gray
from detectors import DETECTORS, MARGIN, inner_mask, repeatability

SERIES = {
    "rotation": [float(a) for a in range(0, 91, 10)],
    "scale": [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
    "brightness": [round(0.5 + 0.1 * i, 1) for i in range(11)],
    "noise": [0.0, 5.0, 10.0, 15.0, 20.0],
    "blur": [0.0, 1.0, 2.0, 3.0, 4.0],
    "contrast": [round(0.5 + 0.1 * i, 1) for i in range(11)],
}
DEFAULT_SERIES = ["rotation", "scale", "brightness", "noise"]
IDENTITY = {"rotation": 0.0, "scale": 1.0, "brightness": 1.0, "noise": 0.0,
            "blur": 0.0, "contrast": 1.0}
COLUMNS = ["file", "series", "param", "detector", "n_p1_vis", "n_p2", "matches",
           "repeatability"]


def transform(gray, series, value, rng):
    """Повертає (перетворений кадр uint8, H 3x3, видима область uint8 1/0)."""
    h, w = gray.shape
    ones = np.ones((h, w), np.uint8)
    eye = np.eye(3)

    if series == "rotation":
        m = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), value, 1.0)
        H = np.vstack([m, [0, 0, 1]])
        img = cv2.warpPerspective(gray, H, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid = cv2.warpPerspective(ones, H, (w, h), flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return img, H, valid

    if series == "scale":
        w2, h2 = round(w * value), round(h * value)
        sx, sy = w2 / w, h2 / h
        interp = cv2.INTER_AREA if value < 1 else cv2.INTER_LINEAR
        img = gray.copy() if value == 1.0 else cv2.resize(gray, (w2, h2), interpolation=interp)
        H = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
        return img, H, np.ones((h2, w2), np.uint8)

    g = gray.astype(np.float32)
    if series == "brightness":
        img = np.clip(g * value, 0, 255)
    elif series == "noise":
        img = np.clip(g + rng.normal(0.0, value, g.shape), 0, 255) if value > 0 else g
    elif series == "blur":
        img = cv2.GaussianBlur(g, (0, 0), value) if value > 0 else g
    elif series == "contrast":
        img = np.clip((g - g.mean()) * value + g.mean(), 0, 255)
    else:
        raise ValueError(series)
    return np.round(img).astype(np.uint8), eye, ones


def load_done(path):
    done = set()
    if path.exists():
        with open(path, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                done.add((r["file"], r["series"], float(r["param"]), r["detector"]))
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", nargs="+", default=DEFAULT_SERIES, choices=list(SERIES))
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--out", default=str(RESULTS_DIR / "repeatability.csv"))
    args = ap.parse_args()

    paths = image_paths()[:args.max_images]
    if not paths:
        raise SystemExit("У data/prepared/gray/ немає PNG")
    RESULTS_DIR.mkdir(exist_ok=True)
    from pathlib import Path
    out = Path(args.out)
    done = load_done(out)
    new_file = not out.exists()
    f = open(out, "a", newline="", encoding="utf-8-sig")
    wr = csv.DictWriter(f, fieldnames=COLUMNS)
    if new_file:
        wr.writeheader()

    t_start = time.time()
    for img_idx, p in enumerate(paths):
        gray = load_gray(p)
        mask0 = inner_mask(np.ones(gray.shape, np.uint8))
        p1 = {name: fn(gray, mask0) for name, fn in DETECTORS.items()}

        for series in args.series:
            for value in SERIES[series]:
                todo = [d for d in DETECTORS if (p.name, series, float(value), d) not in done]
                if not todo:
                    continue
                rng = np.random.default_rng(SEED + img_idx)
                img2, H, valid = transform(gray, series, value, rng)
                mask2 = inner_mask(valid)
                for det in todo:
                    p2 = DETECTORS[det](img2, mask2)
                    n1, n2, m, rep = repeatability(p1[det], p2, H, mask2)
                    row = {"file": p.name, "series": series, "param": float(value),
                           "detector": det, "n_p1_vis": n1, "n_p2": n2,
                           "matches": m, "repeatability": round(rep, 4)}
                    wr.writerow(row)
                    f.flush()
                    if value == IDENTITY[series] and rep < 0.99:
                        print(f"  [УВАГА] контроль {series}={value} {det}: {rep:.2%} "
                              f"(очікується 100 %)")
            print(f"{p.name}: серія {series} готова "
                  f"({time.time() - t_start:.0f} с від початку)")
    f.close()
    print(f"\nГотово -> {out}")


if __name__ == "__main__":
    main()