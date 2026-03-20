from __future__ import annotations

from pathlib import Path

from utils.demography import configure_matplotlib_cn
from utils.tfr_tools import forecast_tfr_simplified, load_tfr_data, plot_tfr_dashboard, print_tfr_summary


BASE_DIR = Path(__file__).resolve().parent
configure_matplotlib_cn()


def main() -> None:
    data_by_year = load_tfr_data(BASE_DIR)
    for year, df in sorted(data_by_year.items()):
        print(f"{year} 年历史数据行数: {len(df)}")

    forecast_result = forecast_tfr_simplified(data_by_year, target_years=(2030, 2040))
    print_tfr_summary(data_by_year, forecast_result=forecast_result)
    print("说明：由于缺少分孩次妇女人数 W_k(a,t)，本脚本按 Todo 第七阶段的简化法进行未来 TFR 预测。")
    plot_tfr_dashboard(data_by_year, forecast_result=forecast_result)


if __name__ == "__main__":
    main()
