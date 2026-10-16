# -*- coding: utf-8 -*-
"""
Лабораторна робота 1. Середовище, подання зображень, колірні простори.

Запуск з теки lab01/src:   python lab01.py
Вхідні файли лежать у lab01/data/ (назви наведені нижче), числа зберігаються
в lab01/results/ (CSV), рисунки - у lab01/figures/.
"""
import csv
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ExifTags

np.random.seed(42)  # фіксуємо випадковий стан (вимога курсу)

ROOT = Path(__file__).resolve().parent.parent
DATA, RES, FIG = ROOT / "data", ROOT / "results", ROOT / "figures"
TMP = RES / "tmp"  # сюди пишемо тимчасові PNG/JPEG для кроку 4
for d in (RES, FIG, TMP):
    d.mkdir(exist_ok=True)


# ---------------------------------------------------------------- допоміжне
def load(name, flags=cv2.IMREAD_COLOR):
    """Читає файл із data/. cv2.imread при помилці мовчки повертає None, тому перевіряємо."""
    im = cv2.imread(str(DATA / name), flags)
    assert im is not None, f"Немає файлу data/{name}"
    return im


def psnr(a, b):
    """PSNR = 10*log10(255^2 / MSE). Рахуємо у float64, щоб різниця uint8 не переповнилась."""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255 ** 2 / mse))


def save_csv(name, header, rows):
    with open(RES / name, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def rgb(img):
    """OpenCV зберігає канали як BGR, а matplotlib чекає RGB."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ================================================================ ЧАСТИНА A
sample = load("sample.jpg")

# --- Крок 1. Властивості масиву
print("shape:", sample.shape, "dtype:", sample.dtype,
      "min:", sample.min(), "max:", sample.max())

# --- Крок 2. Порядок каналів: без перетворення кольори спотворені
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].imshow(sample)
ax[0].set_title("без перетворення (BGR читається як RGB)")
ax[1].imshow(rgb(sample))
ax[1].set_title("після BGR2RGB")
for a in ax:
    a.axis("off")
fig.savefig(FIG / "step2_bgr_rgb.png", dpi=150)
plt.close(fig)

# --- Крок 3. Переповнення uint8
bad = sample + 50                                   # uint8: 230 + 50 = 280 -> 24 (за модулем 256)
good = np.clip(sample.astype(np.int16) + 50, 0, 255).astype(np.uint8)  # int16, потім обрізання
wrapped = (bad < sample).any(axis=2)                # пікселі, що були яскравими й стали темними
print("пікселів із переповненням:", wrapped.sum(), f"({100 * wrapped.mean():.2f} %)")
assert wrapped.any(), "Потрібне фото з яскравими/білими ділянками (небо, біла стіна)"
g0, g1 = (cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in (sample, bad))
b2d = (g0 >= 200) & (g1 < 100)                      # були яскравими (>=200) і стали темними (<100)
print("яскраві стали темними:", b2d.sum(), f"({100 * b2d.mean():.2f} %)")

ys, xs = np.nonzero(wrapped)
y, x = ys[0], xs[0]                                 # фрагмент навколо першого пошкодженого пікселя
win = (slice(y, y + 150), slice(x, x + 150))
titles = ("оригінал", "img + 50 (uint8): темні плями", "int16 + clip")
fig, ax = plt.subplots(2, 3, figsize=(12, 7))
for j, (im, t) in enumerate(zip((sample, bad, good), titles)):
    s_ = 900 / im.shape[1]                          # верхній ряд: весь кадр, зменшений для показу
    ax[0, j].imshow(rgb(cv2.resize(im, None, fx=s_, fy=s_, interpolation=cv2.INTER_AREA)))
    ax[0, j].set_title(t + " (весь кадр)")
    ax[1, j].imshow(rgb(im[win]))                   # нижній ряд: фрагмент 150x150 1:1
    ax[1, j].set_title("фрагмент 150x150")
for a in ax.ravel():
    a.axis("off")
fig.tight_layout()
fig.savefig(FIG / "step3_overflow.png", dpi=150)
plt.close(fig)

# --- Крок 4. Втрати при стисненні: PNG і JPEG (якість 95, 75, 50, 25, 10), PSNR
rows = []
for fmt, q in [("PNG", None), ("JPEG", 95), ("JPEG", 75), ("JPEG", 50), ("JPEG", 25), ("JPEG", 10)]:
    if fmt == "PNG":
        p = TMP / "test.png"
        cv2.imwrite(str(p), sample)
    else:
        p = TMP / f"test_{q}.jpg"
        cv2.imwrite(str(p), sample, [cv2.IMWRITE_JPEG_QUALITY, q])
    back = cv2.imread(str(p))                       # зчитуємо ЗАПИСАНИЙ файл назад
    rows.append([fmt, q, round(p.stat().st_size / 1024, 1), round(psnr(sample, back), 2)])
    print(rows[-1])
save_csv("step4_compression.csv", ["формат", "якість", "розмір, КБ", "PSNR, дБ"], rows)

jpeg = [r for r in rows if r[0] == "JPEG"]
fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogx([r[2] for r in jpeg], [r[3] for r in jpeg], "o-")  # розмір змінюється на порядок -> логарифмічна вісь
for r in jpeg:
    ax.annotate(f"q={r[1]}", (r[2], r[3]))
ax.set_xlabel("розмір файлу, КБ (логарифмічна шкала)")
ax.set_ylabel("PSNR, дБ")
ax.set_title("JPEG: PSNR залежно від розміру файлу")
ax.grid(alpha=0.3)
fig.savefig(FIG / "step4_psnr_vs_size.png", dpi=150)
plt.close(fig)

# --- Крок 5. EXIF: фокусна відстань, витримка, ISO
exif = Image.open(DATA / "sample.jpg").getexif().get_ifd(0x8769)   # блок Exif
tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
iso = tags.get("ISOSpeedRatings", tags.get("PhotographicSensitivity"))
print("EXIF:", tags.get("FocalLength"), tags.get("ExposureTime"), iso)
save_csv("step5_exif.csv", ["фокусна, мм", "витримка, с", "ISO"],
         [[tags.get("FocalLength"), tags.get("ExposureTime"), iso]])

# ================================================================ ЧАСТИНА B
day = load("obj_day.jpg")                           # кадр 1: денне світло
lamp = load("obj_lamp.jpg")                         # кадр 2: лампа розжарювання
gt = load("obj_day_mask.png", cv2.IMREAD_GRAYSCALE) > 127   # еталонна маска (крок 7)
print(f"об'єкт займає {100 * gt.mean():.2f} % кадру")


def metrics(pred, gt):
    """Частка правильно класифікованих пікселів і IoU = |A∩B| / |A∪B|."""
    return float((pred == gt).mean()), float((pred & gt).sum() / (pred | gt).sum())


# Кроки 8-11. Межі підбираємо ЛИШЕ на кадрі 1 (за пікселями всередині маски:
# 1-й і 99-й перцентилі по кожному каналу), на кадрі 2 застосовуємо без змін.
# Увага: в OpenCV H має діапазон 0-179; для червоного об'єкта потрібні два інтервали.
rows = []
for name, conv in [("RGB", cv2.COLOR_BGR2RGB), ("HSV", cv2.COLOR_BGR2HSV)]:
    f1, f2 = cv2.cvtColor(day, conv), cv2.cvtColor(lamp, conv)
    lo = np.percentile(f1[gt], 1, axis=0).astype(np.uint8)
    hi = np.percentile(f1[gt], 99, axis=0).astype(np.uint8)
    row = [name, lo.tolist(), hi.tolist()]
    for f in (f1, f2):
        pred = cv2.inRange(f, lo, hi) > 0           # cv2.inRange дає маску 0/255
        acc, iou = metrics(pred, gt)
        row += [round(acc, 4), round(iou, 4)]
    rows.append(row)
    print(row)
save_csv("part_b_color_spaces.csv",
         ["простір", "нижня межа", "верхня межа",
          "accuracy кадр 1", "IoU кадр 1", "accuracy кадр 2", "IoU кадр 2"], rows)

# ================================================================ ЧАСТИНА C
# --- Крок 12. Аліасинг: зменшення в 4 рази двома способами
tex = load("texture.jpg")
h, w = tex.shape[:2]
size = (w // 4, h // 4)                             # cv2.resize приймає (ширина, висота)
naive = cv2.resize(tex, size, interpolation=cv2.INTER_NEAREST)
smooth = cv2.resize(cv2.GaussianBlur(tex, (0, 0), 2), size, interpolation=cv2.INTER_NEAREST)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fy, fx = size[1] // 2 - 150, size[0] // 2 - 150     # фрагмент 300x300 у центрі, показуємо 1:1 (без стиснення при виводі)
ax[0].imshow(rgb(naive[fy:fy + 300, fx:fx + 300]), interpolation="nearest")
ax[0].set_title("INTER_NEAREST без згладжування (муар)")
ax[1].imshow(rgb(smooth[fy:fy + 300, fx:fx + 300]), interpolation="nearest")
ax[1].set_title("GaussianBlur, потім зменшення")
for a in ax:
    a.axis("off")
fig.savefig(FIG / "step12_aliasing.png", dpi=150)
plt.close(fig)

# --- Крок 13. Інтерполяція: фрагмент зменшуємо в 8 разів, збільшуємо назад, порівнюємо з оригіналом
detail = load("detail.jpg")
cy, cx = detail.shape[0] // 2, detail.shape[1] // 2
crop = detail[cy - 128:cy + 128, cx - 128:cx + 128]  # оригінал високої роздільності 256x256
small = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA)
methods = {"INTER_NEAREST": cv2.INTER_NEAREST, "INTER_LINEAR": cv2.INTER_LINEAR,
           "INTER_CUBIC": cv2.INTER_CUBIC, "INTER_LANCZOS4": cv2.INTER_LANCZOS4}
rows = []
fig, ax = plt.subplots(1, 5, figsize=(18, 4))
ax[0].imshow(rgb(crop))
ax[0].set_title("оригінал")
for a, (name, flag) in zip(ax[1:], methods.items()):
    up = cv2.resize(small, (256, 256), interpolation=flag)
    rows.append([name, round(psnr(up, crop), 2)])
    a.imshow(rgb(up))
    a.set_title(f"{name}, PSNR {rows[-1][1]} дБ")
for a in ax:
    a.axis("off")
fig.savefig(FIG / "step13_interpolation.png", dpi=150)
plt.close(fig)
print(rows)
save_csv("step13_interpolation.csv", ["метод", "PSNR відносно оригіналу, дБ"], rows)

# --- Крок 14. Гамма-корекція V' = V^(1/gamma), gamma = 0.5, 1.0, 2.2
gammas = (0.5, 1.0, 2.2)
v = sample / 255.0
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for a, g in zip(ax, gammas):
    a.imshow(rgb((255 * v ** (1 / g)).astype(np.uint8)))
    a.set_title(f"gamma = {g}")
    a.axis("off")
fig.savefig(FIG / "step14_gamma_images.png", dpi=150)
plt.close(fig)

x = np.linspace(0, 1, 256)
fig, ax = plt.subplots(figsize=(5, 4))
for g in gammas:
    ax.plot(x, x ** (1 / g), label=f"gamma = {g}")
ax.set_xlabel("вхідне значення V")
ax.set_ylabel("вихідне значення V'")
ax.set_title("Передавальні характеристики")
ax.grid(alpha=0.3)
ax.legend()
fig.savefig(FIG / "step14_gamma_curves.png", dpi=150)
plt.close(fig)

# Чому усереднення гамма-кодованих значень != усереднення освітленостей:
E = np.array([0.04, 0.64])                          # фізичні освітленості двох пікселів
right = E.mean()                                    # правильно: усереднюємо освітленості
naive = (E ** (1 / 2.2)).mean() ** 2.2              # усереднили збережені значення й повернули у фізичні
print(f"правильне середнє {right:.3f}, наївне {naive:.3f}")