"""ЛР4, частина D: панорама з кадрів серії.

Схема:
  1. Для кожної пари сусідніх кадрів (i, i+1) – гомографія H(i→i+1) власним RANSAC
     (τ = MATCH_TAU, N = RANSAC_N_ITER, seed = SEED) на робочому наборі кандидатів.
  2. Опорний кадр REF_FRAME (центральний). Гомографії кадрів до опорного
     складаємо ланцюгом: ліворуч H(i→ref) = H(i+1→ref)·H(i→i+1),
     праворуч H(i→ref) = H(i-1→ref)·H(i-1→i)^(-1).
  3. Полотно: образи кутів усіх кадрів під H(i→ref) дають обмежувальний
     прямокутник; зсув T робить усі координати невід'ємними.
  4. Склейка двома способами на ОДНАКОВИХ гомографіях і полотні:
     «накладання» (пізніший кадр затирає попередній) і лінійне змішування
     (вага пікселя = відстань до межі кадру, distanceTransform; сума ваг нормується).
  5. Похибка репроєкції для ВСІХ пар кадрів (не лише сусідніх): точки пари
     переносимо в опорний кадр гомографіями ланцюга й міряємо відстань між образами.
     Для несусідніх пар це показує, як накопичується похибка ланцюга.

Метрики (усі похибки в пікселях кадру j, тобто другого кадру пари, як у частинах B і C):
  ланцюг дає гомографію пари H(i→j) = H(j→ref)^(-1) · H(i→ref); її похибку
  репроєкції рахуємо на точках, узгоджених із ПРЯМОЮ оцінкою пари (власний RANSAC).
  Для сусідніх пар ланцюг збігається з прямою оцінкою, тож E тут лише контроль;
  для несусідніх E показує накопичення похибки ланцюга;
  E_chain_*   – середнє й медіана цієї похибки;
  share_within_tau – частка ВСІХ кандидатів пари, що узгоджуються з ланцюгом
                (похибка < τ): не залежить від прямої оцінки пари;
  plausible   – чи правдоподібна пряма гомографія пари (overlap_fraction);
  mae_gray    – середній модуль різниці яскравості (0..255) у перекритті двох
                перенесених кадрів на полотні (фотометрична узгодженість).
Перевірка коректності H кадру (panorama_H.csv): знаменник гомографії
d(x, y) = h20·x + h21·y + 1 у кутах кадру. Якщо він змінює знак, лінія нескінченності
перетинає кадр, образ кадру хибний (частина кадру відображається «через нескінченність»);
скрипт попереджає про це, але склеює далі, щоб проблему було видно.

Необов'язковий аргумент --frames: підсерія кадрів (опорний REF_FRAME має в ній бути);
результати отримують суфікс _<перший>_<останній>, наприклад panorama_8736_8738.png.

Результати перезаписуються при кожному запуску (скрипт швидкий; пропуск «уже
пораховано» тут не потрібен і приховував би зміну τ).

Запуск з кореня lab_4:
    python src/05_panorama.py

Результати:
    results/panorama_H.csv        H(i→ref) кожного кадру, довжина ланцюга, ланки
    results/panorama_pairs.csv    усі пари кадрів: кандидати, узгоджені, похибки
    figures/panorama.png          накладання проти лінійного змішування
    figures/panorama_errors.png   медіанна похибка ланцюга за парами
    figures/panorama.jpg          повнорозмірна панорама (лінійне змішування)
"""
import argparse
import itertools

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (FIGURES_DIR, MATCH_TAU, MAX_CANVAS_SIDE, MIN_INL_METRIC, MIN_PAIRS,
                    RANSAC_N_ITER, REF_FRAME, RESULTS_DIR, SEED, SERIES, append_row,
                    load_frame, overlap_fraction, reproj_errors, work_matches)
from ransac_core import ransac_homography

np.random.seed(SEED)

TAG = ""   # суфікс імен файлів для підсерії (див. main)
H_FIELDS = (["frame", "chain_len"] + [f"h{r}{c}" for r in range(3) for c in range(3)]
            + ["den_min", "den_max", "valid"])
PAIR_FIELDS = ["pair", "kind", "n_cand", "n_inl_direct", "plausible", "E_chain_mean",
               "E_chain_median", "n_within_tau", "share_within_tau", "overlap_px", "mae_gray"]


def apply_h(h, pts):
    """Перенесення точок N×2 гомографією h (float64)."""
    return cv2.perspectiveTransform(np.asarray(pts, np.float64).reshape(-1, 1, 2),
                                    np.asarray(h, np.float64)).reshape(-1, 2)


def short(stem):
    return stem[-4:]


def out(kind, name, ext):
    """Шлях вихідного файла з урахуванням суфікса підсерії: results/panorama_pairs.csv тощо."""
    folder = RESULTS_DIR if kind == "results" else FIGURES_DIR
    return folder / f"{name}{TAG}.{ext}"


def denominator_range(h, size):
    """Мінімум і максимум знаменника гомографії d = h20·x + h21·y + 1 у кутах кадру.
    Знаменник < 0 в одних кутах і > 0 в інших означає, що кадр перетинає лінію
    нескінченності. Знаменник близько 0 означає сильне збільшення поблизу того кута."""
    w, hh = size
    d = [h[2, 0] * x + h[2, 1] * y + h[2, 2] for x, y in ((0, 0), (w, 0), (w, hh), (0, hh))]
    return float(min(d)), float(max(d))


# ================================================================ гомографії ланцюга
def neighbour_homographies():
    """H(i→i+1) для сусідніх кадрів серії. Повертає список (H, n_inl, n_cand).
    Якщо пару оцінити не вдалося, панораму не побудувати, тож зупиняємось."""
    links = []
    for a, b in zip(SERIES[:-1], SERIES[1:]):
        src, dst = work_matches(a, b)
        h, mask = ransac_homography(src, dst, MATCH_TAU, RANSAC_N_ITER, SEED)
        if h is None:
            raise SystemExit(f"Не вдалося оцінити H для сусідньої пари {a}-{b}")
        links.append((h / h[2, 2], int(mask.sum()), len(src)))
        print(f"  {short(a)}→{short(b)}: кандидатів={len(src)}, узгоджених={int(mask.sum())}")
    return links


def chain_to_ref(links):
    """H(i→ref) для кожного кадру і довжина ланцюга (кількість ланок до опорного)."""
    n = len(SERIES)
    ref = SERIES.index(REF_FRAME)
    hs = [None] * n
    hs[ref] = np.eye(3)
    for i in range(ref - 1, -1, -1):            # ліворуч від опорного
        hs[i] = hs[i + 1] @ links[i][0]
    for i in range(ref + 1, n):                 # праворуч: обернена ланка (i-1 → i)
        hs[i] = hs[i - 1] @ np.linalg.inv(links[i - 1][0])
    hs = [h / h[2, 2] for h in hs]
    return hs, [abs(i - ref) for i in range(n)]


# ================================================================ полотно й склейка
def make_canvas(hs):
    """Обмежувальний прямокутник образів кутів усіх кадрів. Повертає (зсув T, (W, H))."""
    pts = []
    for stem, h in zip(SERIES, hs):
        gh, gw = load_frame(stem)[1].shape
        corners = [[0, 0], [gw, 0], [gw, gh], [0, gh]]
        pts.append(apply_h(h, corners))
    pts = np.vstack(pts)
    x0, y0 = np.floor(pts.min(axis=0)).astype(int)
    x1, y1 = np.ceil(pts.max(axis=0)).astype(int)
    w, h = int(x1 - x0), int(y1 - y0)
    if max(w, h) > MAX_CANVAS_SIDE:
        raise SystemExit(f"Полотно {w}×{h} px завелике (> {MAX_CANVAS_SIDE}): гомографії хибні")
    shift = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
    return shift, (w, h)


def warp_all(hs, shift, size):
    """Кожен кадр на полотні: (зображення uint8, маска bool, вага float32)."""
    out = []
    for stem, h in zip(SERIES, hs):
        bgr = load_frame(stem)[0]
        hc = shift @ h
        warped = cv2.warpPerspective(bgr, hc, size)
        ones = np.full(bgr.shape[:2], 255, np.uint8)
        mask = cv2.warpPerspective(ones, hc, size, flags=cv2.INTER_NEAREST) > 0
        # вага = відстань до найближчого пікселя поза кадром: на межі 0, у центрі максимум
        weight = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        out.append((warped, mask, weight))
    return out


def blend_overlay(layers, size):
    """Накладання: пізніший кадр серії затирає попередній (де є дані)."""
    canvas = np.zeros((size[1], size[0], 3), np.uint8)
    for warped, mask, _ in layers:
        canvas[mask] = warped[mask]
    return canvas


def blend_linear(layers, size):
    """Лінійне змішування: сума (вага · піксель) / сума ваг."""
    acc = np.zeros((size[1], size[0], 3), np.float32)
    wsum = np.zeros((size[1], size[0]), np.float32)
    for warped, _, weight in layers:
        acc += warped.astype(np.float32) * weight[..., None]
        wsum += weight
    ok = wsum > 0
    acc[ok] /= wsum[ok][:, None]
    return np.clip(acc, 0, 255).astype(np.uint8)


# ================================================================ похибка за всіма парами
def pair_table(hs, layers):
    """Рядки для всіх пар кадрів (i < j). Пише CSV рядок за рядком."""
    rows = []
    grays = [cv2.cvtColor(w, cv2.COLOR_BGR2GRAY).astype(np.float32) for w, _, _ in layers]
    for i, j in itertools.combinations(range(len(SERIES)), 2):
        a, b = SERIES[i], SERIES[j]
        src, dst = work_matches(a, b)
        n = len(src)
        row = {"pair": f"{short(a)}-{short(b)}",
               "kind": "сусідні" if j == i + 1 else "несусідні", "n_cand": n,
               "n_inl_direct": "", "plausible": "", "E_chain_mean": "",
               "E_chain_median": "", "n_within_tau": "", "share_within_tau": ""}
        if n >= MIN_PAIRS:
            h_chain = np.linalg.inv(hs[j]) @ hs[i]          # H(i→j) за ланцюгом
            err = reproj_errors(h_chain, src, dst)       # px кадру j
            row["n_within_tau"] = int((err < MATCH_TAU).sum())
            row["share_within_tau"] = round(float((err < MATCH_TAU).mean()), 4)
            h, mask = ransac_homography(src, dst, MATCH_TAU, RANSAC_N_ITER, SEED)
            if h is not None:
                row["n_inl_direct"] = int(mask.sum())
                wa, ha = load_frame(a)[1].shape[::-1]
                wb, hb = load_frame(b)[1].shape[::-1]
                row["plausible"] = bool(overlap_fraction(h, (wa, ha), (wb, hb))[1])
                if mask.sum() >= MIN_INL_METRIC:
                    row["E_chain_mean"] = round(float(err[mask].mean()), 3)
                    row["E_chain_median"] = round(float(np.median(err[mask])), 3)
        ov = layers[i][1] & layers[j][1]
        row["overlap_px"] = int(ov.sum())
        row["mae_gray"] = round(float(np.abs(grays[i][ov] - grays[j][ov]).mean()), 2) \
            if ov.any() else ""
        append_row(out("results", "panorama_pairs", "csv"), PAIR_FIELDS, row)
        rows.append(row)
    return rows


# ================================================================ рисунки
def draw_panorama(overlay, linear, size):
    fig, axes = plt.subplots(2, 1, figsize=(13, 9.5))
    for ax, img, title in ((axes[0], overlay, "Накладання (пізніший кадр затирає попередній)"),
                           (axes[1], linear, "Лінійне змішування (вага = відстань до межі кадру)")):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), extent=(0, size[0], size[1], 0))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x, px")
        ax.set_ylabel("y, px")
    fig.suptitle(f"Панорама {short(SERIES[0])}–{short(SERIES[-1])}, опорний кадр "
                 f"{short(REF_FRAME)}, полотно {size[0]}×{size[1]} px; ті самі H і полотно",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out("figures", "panorama", "png"), dpi=110)
    plt.close(fig)


def draw_errors(rows):
    """Медіанна похибка ланцюга на узгоджених точках пари (лог. шкала), точки
    замість стовпчиків: на логарифмічній осі в стовпчика немає змістовної основи.
    Під назвою пари – кількість узгоджених за прямою оцінкою; × – пряма H нерозумна."""
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = {"сусідні": "C0", "несусідні": "C1"}
    for k, r in enumerate(rows):
        v = r["E_chain_median"]
        if v == "":
            ax.text(k, 0.02, "н/д", ha="center", fontsize=8, transform=ax.get_xaxis_transform())
            continue
        ax.plot(k, v, "o", ms=9, color=colors[r["kind"]])
        if r["plausible"] is False:
            ax.plot(k, v, "x", ms=14, color="red")
    ax.axhline(MATCH_TAU, color="gray", ls="--", lw=1)
    ax.text(len(rows) - 0.5, MATCH_TAU, f" τ = {MATCH_TAU:g} px", va="bottom", ha="right",
            fontsize=8, color="gray")
    ax.set_yscale("log")
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"{r['pair']}\n{r['n_inl_direct'] if r['n_inl_direct'] != '' else '–'}"
                        for r in rows], fontsize=8)
    ax.set_xlabel("Пара кадрів (внизу: узгоджених за прямою оцінкою)\n"
                  "н/д – менше 10 узгоджених, × – пряма гомографія пари нерозумна")
    ax.set_ylabel("Медіанна похибка ланцюга, px кадру j (лог. шкала)")
    ax.set_title("Похибка репроєкції пари за гомографією ланцюга H(j→ref)^(-1)·H(i→ref)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=c) for c in colors.values()],
              labels=list(colors), loc="upper left")
    fig.tight_layout()
    fig.savefig(out("figures", "panorama_errors", "png"), dpi=130)
    plt.close(fig)


def main():
    global SERIES, TAG
    ap = argparse.ArgumentParser(description="Панорама з кадрів серії (ЛР4, частина D)")
    ap.add_argument("--frames", nargs="+", help="підсерія кадрів зліва направо (за замовчуванням SERIES)")
    args = ap.parse_args()
    if args.frames:
        # Функції нижче читають глобальну SERIES, тож підсерію підставляємо тут одним місцем
        SERIES = tuple(args.frames)
        TAG = f"_{short(SERIES[0])}_{short(SERIES[-1])}"
    if REF_FRAME not in SERIES:
        raise SystemExit(f"Опорного кадра {REF_FRAME} немає в серії {SERIES}")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for f in (out("results", "panorama_H", "csv"), out("results", "panorama_pairs", "csv")):
        if f.exists():                           # кожен запуск пише з нуля
            f.unlink()
    print(f"Умови: τ = {MATCH_TAU} px, N = {RANSAC_N_ITER}, seed = {SEED}, "
          f"опорний кадр {REF_FRAME}, кадри: {', '.join(short(x) for x in SERIES)}\n")

    print("=== Сусідні пари (власний RANSAC) ===")
    links = neighbour_homographies()
    hs, chain = chain_to_ref(links)
    print("\n=== Перевірка H(i→ref): знаменник у кутах кадру ===")
    for stem, h, c in zip(SERIES, hs, chain):
        gh, gw = load_frame(stem)[1].shape
        d_min, d_max = denominator_range(h, (gw, gh))
        valid = d_min > 0
        row = {"frame": stem, "chain_len": c, "den_min": round(d_min, 3),
               "den_max": round(d_max, 3), "valid": valid}
        row.update({f"h{r}{k}": float(f"{h[r, k]:.8g}") for r in range(3) for k in range(3)})
        append_row(out("results", "panorama_H", "csv"), H_FIELDS, row)
        print(f"  {short(stem)}: знаменник {d_min:.2f}..{d_max:.2f} "
              f"{'ок' if valid else 'ЗМІНЮЄ ЗНАК: кадр перетинає лінію нескінченності, образ хибний'}")

    shift, size = make_canvas(hs)
    print(f"\nПолотно: {size[0]}×{size[1]} px")
    layers = warp_all(hs, shift, size)

    print("\n=== Усі пари кадрів ===")
    rows = pair_table(hs, layers)
    print(f"{'пара':<10}{'тип':<11}{'канд.':>6}{'узг.':>6}{'прав.':>7}{'E_сер':>8}{'E_мед':>8}"
          f"{'<τ,шт':>7}{'<τ,%':>7}{'MAE':>7}")
    for r in rows:
        def f(x, fmt):
            return format(x, fmt) if x != "" else "–"
        sh = f(r["share_within_tau"] * 100 if r["share_within_tau"] != "" else "", ".1f")
        print(f"{r['pair']:<10}{r['kind']:<11}{r['n_cand']:>6}{f(r['n_inl_direct'], 'd'):>6}"
              f"{str(r['plausible']) if r['plausible'] != '' else '–':>7}"
              f"{f(r['E_chain_mean'], '.2f'):>8}{f(r['E_chain_median'], '.2f'):>8}"
              f"{f(r['n_within_tau'], 'd'):>7}{sh:>7}{f(r['mae_gray'], '.1f'):>7}")

    overlay, linear = blend_overlay(layers, size), blend_linear(layers, size)
    draw_panorama(overlay, linear, size)
    draw_errors(rows)
    ok, buf = cv2.imencode(".jpg", linear, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if ok:                                       # imencode+tofile: шлях може мати кирилицю
        buf.tofile(str(out("figures", "panorama", "jpg")))
    print(f"\nРисунки у {FIGURES_DIR}")


if __name__ == "__main__":
    main()