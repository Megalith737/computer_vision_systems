# Лабораторна робота 2. Калібрування камери та усунення дисторсії

Мета: відкалібрувати камеру смартфона за шахівницею 9×6 внутрішніх кутів, оцінити
похибку репроєкції, дослідити залежність результату від кількості кадрів, усунути
дисторсію й зберегти параметри в `camera_params.yml` (потрібен у ЛР5).

## Підготовка

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install opencv-contrib-python numpy matplotlib scipy
pip freeze > requirements.txt
```

Кадри (не потрапляють у git) лежать у `data/frames/`:
`cal_01.jpg … cal_25.jpg` - калібрувальні, `line_01.jpg`, `line_02.jpg` - кадри з прямою лінією.
Шахівниця 9×6 внутрішніх кутів, розмір клітини - `CELL_MM` у `src/common.py`.

## Порядок запуску (з кореня `lab_2/`)

| Скрипт | Крок практикуму | Результат |
|---|---|---|
| `src/01_corners.py` | 6 | `results/corners.npz`, `figures/corners/` |
| `src/02_calibrate.py` | 7-9 | `results/calibration_compare.csv` |
| `src/03_frame_errors.py` | 10-11 | `figures/frame_errors.png`, `results/inliers.txt` |
| `src/04_subsample_study.py` | 12 | `figures/subsample_study.png`, `results/subsample_study.csv` |
| `src/05_undistort.py` | 13-14 | `figures/undistort_*.jpg` |
| `src/06_save_params.py` | 15 | `camera_params.yml` |

Спільний код і константи (розмір шаблона, клітини, seed 42) - у `src/common.py`.