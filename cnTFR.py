from __future__ import annotations

from pathlib import Path

from utils.demography import configure_matplotlib_cn
from utils.tfr_tools import load_age_metrics, plot_tfr_dashboard, print_tfr_summary


BASE_DIR = Path(__file__).resolve().parent
configure_matplotlib_cn()

TABLES = [
    {"path": BASE_DIR / "Dataset/五-l0606.xls", "sheet": "长表6-6", "year": 2000, "kind": "asfr"},
    {"path": BASE_DIR / "Dataset/六-B0603.xls", "sheet": "Sheet1", "year": 2010, "kind": "asfr"},
    {"path": BASE_DIR / "Dataset/七-B0603.xls", "sheet": "Sheet1", "year": 2020, "kind": "asfr"},
]

def main() -> None:
    data_by_year = {}
    for cfg in TABLES:
        df = load_age_metrics(cfg["path"], cfg["sheet"], cfg["kind"])
        data_by_year[cfg["year"]] = df
        print(f"{cfg['year']} 年数据行数: {len(df)}")

    print_tfr_summary(data_by_year)
    plot_tfr_dashboard(data_by_year)


if __name__ == "__main__":
    main()
