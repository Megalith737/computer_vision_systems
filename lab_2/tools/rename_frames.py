"""Допоміжний скрипт: знаходить твої фото серед папки з оригіналами й копіює їх
з іменами cal_01.jpg ... cal_25.jpg, line_01.jpg, line_02.jpg.

Фото порівнюються за зображенням (мініатюра 96x128 у відтінках сірого), а не за
назвою файлу, тому старі імена (IMG_1234.jpg) не мають значення.
Оригінали не змінюються: скрипт лише копіює. Без --apply нічого не копіюється,
виводиться тільки план.

Запуск з кореня lab_2:
    python tools/rename_frames.py --raw "C:/шлях/до/папки/з/фото"            # план
    python tools/rename_frames.py --raw "C:/шлях/до/папки/з/фото" --apply    # копіювання
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
THRESH = 3.0   # середня різниця яскравості мініатюр: для того самого кадру ~0.2, для різних >20


def thumb(path):
    """Мініатюра 96x128 у відтінках сірого. cv2.imread сам враховує EXIF-поворот."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.resize(img, (96, 128), interpolation=cv2.INTER_AREA).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", required=True, help="папка з оригіналами фото")
    parser.add_argument("--out", default=str(HERE.parent / "data" / "frames"),
                        help="куди копіювати (за замовчуванням data/frames)")
    parser.add_argument("--apply", action="store_true", help="справді скопіювати файли")
    args = parser.parse_args()

    raw_dir, out_dir = Path(args.raw), Path(args.out)
    ref = np.load(HERE / "reference_thumbs.npz")
    names = [str(n) for n in ref["names"]]
    refs = ref["thumbs"].astype(np.float32)

    files = sorted(p for p in raw_dir.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg"))
    heic = [p.name for p in raw_dir.iterdir() if p.suffix.lower() in (".heic", ".heif")]
    if heic:
        print(f"УВАГА: {len(heic)} файлів у форматі HEIC (OpenCV їх не читає): "
              f"{', '.join(heic[:5])}{' ...' if len(heic) > 5 else ''}")
        print("Конвертуй у JPG (на iPhone: Налаштування > Камера > Формати > "
              "Максимальна сумісність, для вже знятих - експорт у JPG).\n")

    matches = {n: [] for n in names}
    unmatched = []
    for p in files:
        t = thumb(p)
        if t is None:
            unmatched.append((p.name, "не вдалося прочитати"))
            continue
        diffs = np.abs(refs - t).mean(axis=(1, 2))
        j = int(diffs.argmin())
        if diffs[j] < THRESH:
            matches[names[j]].append((p, float(diffs[j])))
        else:
            unmatched.append((p.name, f"найближчий {names[j]}, різниця {diffs[j]:.1f}"))

    print(f"{'нове ім\'я':10s}  {'знайдений файл':32s}  різниця")
    missing, copied = [], 0
    for n in names:
        if not matches[n]:
            print(f"{n + '.jpg':10s}  {'-- НЕ ЗНАЙДЕНО --':32s}")
            missing.append(n)
            continue
        src, d = matches[n][0]
        extra = f"  (+{len(matches[n]) - 1} дубл.)" if len(matches[n]) > 1 else ""
        print(f"{n + '.jpg':10s}  {src.name:32s}  {d:5.2f}{extra}")
        if args.apply:
            out_dir.mkdir(parents=True, exist_ok=True)
            target = out_dir / f"{n}.jpg"
            if target.exists():
                print(f"           пропущено: {target.name} уже існує")
            else:
                shutil.copy2(src, target)
                copied += 1

    if unmatched:
        print("\nФайли, яких немає в наборі (запасні, зайві, інша орієнтація):")
        for name, why in unmatched:
            print(f"  {name}: {why}")
    print(f"\nзнайдено {len(names) - len(missing)} з {len(names)}"
          + (f", скопійовано {copied}" if args.apply else " (це лише план, додай --apply)"))
    if missing:
        print("Не знайдено:", ", ".join(missing))


if __name__ == "__main__":
    main()