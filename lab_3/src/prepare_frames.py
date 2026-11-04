"""ЛР3: підготовка кадрів і первинний відбір за числовими метриками.

Що робить:
  1. читає оригінали з --src (jpg/png);
  2. зменшує до --max-side по довшій стороні (INTER_AREA, лише зменшення);
  3. зберігає копії у відтінках сірого (PNG без втрат) у --dst/gray
     (і кольорові, якщо задано --save-color);
  4. рахує для кожного кадру: яскравість, контраст, різкість (дисперсія лапласіана),
     кількість кутів за однаковими параметрами, EXIF (камера, фокусна, витримка, ISO);
  5. позначає підозрілі кадри і пише таблицю в CSV (--csv).

Запуск (з папки lab03):
  python src/prepare_frames.py --src data/raw --dst data/prepared --max-side 1600

Статистика рахується на зменшених кадрах, тому показники порівнянні між кадрами.
Кількість кутів тут лише для відбору кадрів; сам детектор Гарріса
для ЛР3 пишеться окремо (частина A).
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

np.random.seed(42)

EXTS = {".jpg", ".jpeg", ".png"}  # HEIC спершу конвертуйте в JPG

# Параметри для підрахунку кутів: однакові для всіх кадрів
GFTT = dict(maxCorners=5000, qualityLevel=0.01, minDistance=10,
            useHarrisDetector=True, k=0.04)

# Пороги позначок: прості й явні, їх варто вказати у звіті
DARK_MEAN, BRIGHT_MEAN = 50, 205
REL_LOW = 0.5  # менше половини медіани -> підозра


def read_image(path: Path):
    """Читання через imdecode: cv2.imread не відкриває шляхи з кирилицею у Windows.
    EXIF-орієнтацію OpenCV застосовує сам."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: Path, img, params=None):
    ext = path.suffix
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        raise IOError(f"Не вдалося закодувати {path}")
    buf.tofile(str(path))


def read_exif(path: Path):
    info = {"camera": "", "focal_mm": "", "exposure_s": "", "iso": ""}
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            info["camera"] = str(exif.get(0x0110, "")).strip()  # Model
            ifd = exif.get_ifd(0x8769)  # Exif IFD
            if 0x920A in ifd:  # FocalLength
                info["focal_mm"] = round(float(ifd[0x920A]), 2)
            if 0x829A in ifd:  # ExposureTime
                info["exposure_s"] = round(float(ifd[0x829A]), 5)
            iso = ifd.get(0x8827)  # ISOSpeedRatings
            if iso is not None:
                info["iso"] = iso[0] if isinstance(iso, (tuple, list)) else iso
    except Exception:
        pass  # немає EXIF (скриншот, месенджер) -> порожні поля
    return info


def resize_max_side(img, max_side):
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return img, 1.0
    new_size = (round(w * scale), round(h * scale))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA), scale


def frame_stats(gray):
    corners = cv2.goodFeaturesToTrack(gray, **GFTT)
    return {
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "corners": 0 if corners is None else len(corners),
    }


def add_flags(rows):
    med_sharp = float(np.median([r["sharpness"] for r in rows]))
    med_corn = float(np.median([r["corners"] for r in rows]))
    common_cam = Counter(r["camera"] for r in rows).most_common(1)[0][0]
    common_size = Counter((r["orig_w"], r["orig_h"]) for r in rows).most_common(1)[0][0]

    for r in rows:
        flags = []
        if r["mean"] < DARK_MEAN:
            flags.append("темний")
        if r["mean"] > BRIGHT_MEAN:
            flags.append("пересвічений")
        if r["sharpness"] < REL_LOW * med_sharp:
            flags.append("нерізкий")
        if r["corners"] < REL_LOW * med_corn:
            flags.append("мало точок")
        if r["camera"] != common_cam:
            flags.append("інша камера")
        if (r["orig_w"], r["orig_h"]) != common_size:
            flags.append("інший розмір")
        r["flags"] = "; ".join(flags)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/raw", help="папка з оригіналами")
    ap.add_argument("--dst", default="data/prepared", help="куди класти копії")
    ap.add_argument("--max-side", type=int, default=1600, help="довша сторона, px")
    ap.add_argument("--csv", default="results/frame_stats.csv")
    ap.add_argument("--save-color", action="store_true",
                    help="додатково зберегти кольорові копії (JPEG q=95)")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS)
    if not files:
        raise SystemExit(f"У {src} немає jpg/png")

    (dst / "gray").mkdir(parents=True, exist_ok=True)
    if args.save_color:
        (dst / "color").mkdir(parents=True, exist_ok=True)
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in files:
        img = read_image(p)
        if img is None:
            print(f"[пропуск] не вдалося прочитати {p.name}")
            continue
        orig_h, orig_w = img.shape[:2]
        small, scale = resize_max_side(img, args.max_side)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        write_image(dst / "gray" / f"{p.stem}.png", gray)
        if args.save_color:
            write_image(dst / "color" / f"{p.stem}.jpg", small,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

        row = {"file": p.name, "orig_w": orig_w, "orig_h": orig_h,
               "w": small.shape[1], "h": small.shape[0], "scale": round(scale, 4)}
        row.update(frame_stats(gray))
        row.update(read_exif(p))
        rows.append(row)

    rows = add_flags(rows)

    cols = ["file", "orig_w", "orig_h", "w", "h", "scale", "mean", "std",
            "sharpness", "corners", "camera", "focal_mm", "exposure_s", "iso", "flags"]
    with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            wr.writerow({c: (round(r[c], 2) if isinstance(r[c], float) else r[c])
                         for c in cols})

    print(f"{'файл':28} {'розмір':>11} {'mean':>6} {'std':>6} {'sharp':>8} "
          f"{'кути':>6}  позначки")
    for r in rows:
        print(f"{r['file'][:28]:28} {str(r['w']) + 'x' + str(r['h']):>11} "
              f"{r['mean']:6.1f} {r['std']:6.1f} {r['sharpness']:8.1f} "
              f"{r['corners']:6d}  {r['flags']}")
    print(f"\nГотово: {len(rows)} кадрів -> {dst}/gray, таблиця -> {args.csv}")


if __name__ == "__main__":
    main()