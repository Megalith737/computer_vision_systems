"""ЛР3, частина C: SIFT і ORB (detectAndCompute), порівняння з власним Гаррісом.

Для кожного кадру:
  - SIFT і ORB знаходять ключові точки; лишаємо N_POINTS найкращих за response
    (однаково для всіх детекторів, як вимагає умова чесного порівняння);
  - для лишених точок рахуємо дескриптори;
  - міряємо час (медіана з TIMING_RUNS запусків) окремо для пошуку точок
    і для дескрипторів;
  - для власного Гарріса фіксуємо лише кількість точок і час (дескриптора нема).

Таблиця results/detectors_summary.csv: кількість точок, тип і розмірність
дескриптора, час. Рисунок figures/detectors_rich.png: SIFT і ORB з прапорцем
DRAW_RICH_KEYPOINTS (коло = масштаб, лінія = орієнтація) на першому кадрі.

Запуск (з кореня lab_3/):  python src/03_sift_orb.py
Потрібен opencv-contrib-python (або opencv-python >= 4.4, де SIFT у ядрі).
"""
import csv
import time

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import FIGURES_DIR, N_POINTS, RESULTS_DIR, SEED, image_paths, load_gray
from harris import harris_response, nms_topn

np.random.seed(SEED)
cv2.setRNGSeed(SEED)

TIMING_RUNS = 5
POOL = 5000  # скільки точок просимо в ORB, перш ніж відібрати N_POINTS найкращих


def top_n(keypoints, n=N_POINTS):
    """N найкращих точок за response (стабільне сортування)."""
    order = sorted(range(len(keypoints)), key=lambda i: -keypoints[i].response)
    return [keypoints[i] for i in order[:n]]


def median_time(fn, runs=TIMING_RUNS):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return out, float(np.median(times))


def run_detector(detector, gray):
    """Повертає (ключові точки, дескриптори, час пошуку, час дескрипторів)."""
    kps, t_detect = median_time(lambda: detector.detect(gray, None))
    kps = top_n(kps)
    (kps, desc), t_desc = median_time(lambda: detector.compute(gray, kps))
    return kps, desc, t_detect, t_desc


def main():
    paths = image_paths()
    if not paths:
        raise SystemExit("У data/prepared/gray/ немає PNG")
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    sift = cv2.SIFT_create()
    orb = cv2.ORB_create(nfeatures=POOL)  # скоринг Гарріса за замовчуванням

    per_image = []   # по кадрах: детектор, точки, час
    first = None
    for p in paths:
        gray = load_gray(p)

        s_kps, s_desc, s_td, s_tc = run_detector(sift, gray)
        o_kps, o_desc, o_td, o_tc = run_detector(orb, gray)
        _, h_t = median_time(lambda: nms_topn(harris_response(gray)))
        h_pts = nms_topn(harris_response(gray))

        per_image += [
            {"file": p.name, "detector": "Harris (власний)", "n_points": len(h_pts),
             "t_detect_ms": h_t * 1e3, "t_describe_ms": 0.0},
            {"file": p.name, "detector": "SIFT", "n_points": len(s_kps),
             "t_detect_ms": s_td * 1e3, "t_describe_ms": s_tc * 1e3},
            {"file": p.name, "detector": "ORB", "n_points": len(o_kps),
             "t_detect_ms": o_td * 1e3, "t_describe_ms": o_tc * 1e3},
        ]
        print(f"{p.name:16} Harris={len(h_pts):4d} ({h_t*1e3:6.1f} мс)  "
              f"SIFT={len(s_kps):4d} ({(s_td+s_tc)*1e3:6.1f} мс)  "
              f"ORB={len(o_kps):4d} ({(o_td+o_tc)*1e3:6.1f} мс)")
        if first is None:
            first = (p.name, gray, s_kps, o_kps, s_desc, o_desc)

    # зведена таблиця: середнє по кадрах
    desc_info = {
        "Harris (власний)": ("немає", "-", "-"),
        "SIFT": ("float32 (вектор градієнтів)", s_desc.shape[1], f"{s_desc.dtype}"),
        "ORB": ("бітовий (бінарний)", o_desc.shape[1] * 8,
                f"{o_desc.dtype}, {o_desc.shape[1]} байт"),
    }
    summary = []
    for det in ("Harris (власний)", "SIFT", "ORB"):
        sub = [r for r in per_image if r["detector"] == det]
        n = np.array([r["n_points"] for r in sub])
        td = np.array([r["t_detect_ms"] for r in sub])
        tc = np.array([r["t_describe_ms"] for r in sub])
        kind, dim, storage = desc_info[det]
        summary.append({
            "detector": det,
            "n_points_mean": round(float(n.mean()), 1),
            "descriptor_type": kind,
            "descriptor_dim": dim,
            "descriptor_storage": storage,
            "t_detect_ms": round(float(td.mean()), 1),
            "t_describe_ms": round(float(tc.mean()), 1),
            "t_total_ms": round(float((td + tc).mean()), 1),
        })

    with open(RESULTS_DIR / "detectors_summary.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(summary[0]))
        wr.writeheader()
        wr.writerows(summary)
    with open(RESULTS_DIR / "detectors_per_image.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(per_image[0]))
        wr.writeheader()
        for r in per_image:
            wr.writerow({k: (round(v, 2) if isinstance(v, float) else v)
                         for k, v in r.items()})

    print("\nЗведення (середнє по кадрах):")
    for r in summary:
        print(f"  {r['detector']:18} точок={r['n_points_mean']:6.1f}  "
              f"дескриптор: {r['descriptor_type']}, dim={r['descriptor_dim']}  "
              f"час={r['t_total_ms']:7.1f} мс")

    # рисунок із RICH_KEYPOINTS
    name, gray, s_kps, o_kps, _, _ = first
    flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=130)
    for ax, kps, title in ((axes[0], s_kps, "SIFT"), (axes[1], o_kps, "ORB")):
        vis = cv2.drawKeypoints(gray, kps, None, color=(0, 255, 0), flags=flags)
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{title}: {len(kps)} точок ({name})")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "detectors_rich.png")


if __name__ == "__main__":
    main()