"""Записує requirements.txt із точними версіями встановлених пакетів.

Запуск з кореня lab_5 (у активованому venv):  python tools/freeze_requirements.py
Пише лише ті пакети, які потрібні ЛР5; OpenCV визначає за тим, який варіант встановлено.
"""
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OPENCV_VARIANTS = ("opencv-contrib-python", "opencv-python", "opencv-contrib-python-headless",
                   "opencv-python-headless")
OTHER = ("numpy", "matplotlib", "pillow", "scipy")


def version_of(name):
    """Версія встановленого пакета або None."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def main():
    lines = []
    cv_name = next((n for n in OPENCV_VARIANTS if version_of(n)), None)
    if cv_name:
        lines.append(f"{cv_name}=={version_of(cv_name)}")
    else:
        print("УВАГА: OpenCV не знайдено")
    for name in OTHER:
        v = version_of(name)
        if v:
            lines.append(f"{name}=={v}")
        else:
            print(f"Пропущено {name}: не встановлено")
    (ROOT / "requirements.txt").write_text("\n".join(sorted(lines)) + "\n", encoding="utf-8")
    print("Записано requirements.txt:\n" + "\n".join(sorted(lines)))


if __name__ == "__main__":
    main()