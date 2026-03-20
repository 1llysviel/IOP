from __future__ import annotations

from pathlib import Path

from utils.demography import configure_matplotlib_cn
from utils.tfr_tools import load_tfr_data, plot_tfr_dashboard, print_tfr_summary


BASE_DIR = Path(__file__).resolve().parent
configure_matplotlib_cn()

def main() -> None:
    data_by_year = load_tfr_data(BASE_DIR)
    for year, df in sorted(data_by_year.items()):
        print(f"{year} 年数据行数: {len(df)}")

    print_tfr_summary(data_by_year)
    plot_tfr_dashboard(data_by_year)


if __name__ == "__main__":
    main()
