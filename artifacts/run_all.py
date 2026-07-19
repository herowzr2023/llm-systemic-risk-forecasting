"""Run all lightweight table and figure generators in-process."""
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parent
SCRIPT_PATHS = [
    "tables/Table_3/generate_table_3.py",
    "tables/Table_4/generate_table_4.py",
    "tables/Table_5/generate_table_5.py",
    "tables/Table_A2/generate_table_A2.py",
    "tables/Table_A3/generate_table_A3.py",
    "tables/Table_A4/generate_table_A4.py",
    "tables/Table_A6/generate_table_A6.py",
    "tables/Table_A7/generate_table_A7.py",
    "figures/Figure_2/generate_figure_2.py",
    "figures/Figure_A1/generate_figure_A1.py",
    "figures/Figure_A2/generate_figure_A2.py",
]


def main() -> None:
    scripts = [ROOT / path for path in SCRIPT_PATHS]
    for script in scripts:
        print(f"[RUN] {script.relative_to(ROOT)}")
        runpy.run_path(str(script), run_name="__main__")
    print(f"Generated {len(scripts)} artifact groups.")


if __name__ == "__main__":
    main()
