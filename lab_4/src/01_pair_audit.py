"""ЛР4, аудит кадрів: які пари сусідні й наскільки вони перекриваються.

Навіщо. Практикум вимагає серію з перекриттям ~40 %, знятою поворотом з одного
місця. За frame_stats.csv цього не видно, тому для кожної пари кадрів з FRAMES:
  1. SIFT + knnMatch (k = 2) + відношення Лоу з T = AUDIT_RATIO;
  2. гомографія A → B бібліотечним RANSAC з τ = MATCH_TAU;
  3. рахуємо кандидати, узгоджені, їх частку й перекриття кадрів (за образом
     кутів кадру A у кадрі B).
Параметри T і τ тут тимчасові й лише для відбору серії: у частинах A і C вони
досліджуються й обґрунтовуються окремо.

Запуск з кореня lab_4:
    python src/01_pair_audit.py

Результат: results/pair_audit.csv (рядок за рядком; після переривання готові пари
пропускаються). Кадри мають лежати в data/raw/.
"""
import time
from itertools import combinations

import numpy as np

from common import (AUDIT_RATIO, FRAMES, MATCH_TAU, RESULTS_DIR, SEED, append_row,
                    done_keys, estimate_h, find_frame, knn_two, load_frame,
                    matches_to_points, overlap_fraction, ratio_filter, sift_features)

np.random.seed(SEED)

CSV = RESULTS_DIR / "pair_audit.csv"
FIELDS = ["a", "b", "num_gap", "n_kp_a", "n_kp_b", "n_cand", "n_inl", "inl_share",
          "overlap", "plausible", "t_sec"]


def frame_number(stem):
    """Номер із імені IMG_8739 → 8739 (щоб бачити, чи кадри сусідні за номером)."""
    return int(stem.split("_")[-1])


def audit_pair(a, b):
    """Рахує метрики однієї пари; повертає рядок для CSV."""
    t0 = time.time()
    kp_a, d_a = sift_features(a)
    kp_b, d_b = sift_features(b)
    cand = ratio_filter(knn_two(d_a, d_b), AUDIT_RATIO)
    src, dst = matches_to_points(cand, kp_a, kp_b)
    h, mask = estimate_h(src, dst)

    n_inl, overlap, plausible = 0, float("nan"), False
    if h is not None:
        n_inl = int(mask.sum())
        ha, wa = load_frame(a)[1].shape
        hb, wb = load_frame(b)[1].shape
        overlap, plausible = overlap_fraction(h, (wa, ha), (wb, hb))
    return {
        "a": a, "b": b, "num_gap": abs(frame_number(a) - frame_number(b)),
        "n_kp_a": len(kp_a), "n_kp_b": len(kp_b), "n_cand": len(cand),
        "n_inl": n_inl,
        "inl_share": round(n_inl / len(cand), 4) if cand else "",
        "overlap": "" if np.isnan(overlap) else round(overlap, 4),
        "plausible": int(plausible),
        "t_sec": round(time.time() - t0, 2),
    }


def print_report():
    """Виводить у консоль найкорисніше: пари сусідніх номерів і топ за узгодженими."""
    import csv
    with open(CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    def line(r):
        return (f"{r['a']}-{r['b']}  gap={r['num_gap']:>2}  cand={r['n_cand']:>5}  "
                f"inl={r['n_inl']:>5}  share={r['inl_share'] or '-':>6}  "
                f"overlap={r['overlap'] or '-':>6}  ok={r['plausible']}")

    print("\n=== Пари з сусідніми номерами (gap = 1) ===")
    for r in sorted((r for r in rows if r["num_gap"] == "1"), key=lambda r: r["a"]):
        print(line(r))
    print("\n=== Топ-10 пар за кількістю узгоджених ===")
    for r in sorted(rows, key=lambda r: -int(r["n_inl"]))[:10]:
        print(line(r))
    print(f"\nУсього пар у файлі: {len(rows)}. Тимчасові параметри: "
          f"T = {AUDIT_RATIO}, τ = {MATCH_TAU} px.")


def main():
    stems = []
    for s in FRAMES:
        if find_frame(s) is None:
            print(f"УВАГА: немає кадру {s} у data/raw, пропускаю")
        else:
            stems.append(s)
    if len(stems) < 2:
        raise SystemExit("Потрібно щонайменше 2 кадри в data/raw/")

    done = done_keys(CSV, ("a", "b"))
    pairs = list(combinations(stems, 2))
    for k, (a, b) in enumerate(pairs, 1):
        if (a, b) in done:
            continue
        row = audit_pair(a, b)
        append_row(CSV, FIELDS, row)
        print(f"[{k}/{len(pairs)}] {a}-{b}: cand={row['n_cand']}, "
              f"inl={row['n_inl']}, overlap={row['overlap']}")
    print_report()


if __name__ == "__main__":
    main()