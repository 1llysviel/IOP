from __future__ import annotations

from pathlib import Path

from utils.bond_tools import build_historical_bond_observation, plot_historical_bond_dashboard, print_historical_bond_summary
from utils.demography import configure_matplotlib_cn


BASE_DIR = Path(__file__).resolve().parent
configure_matplotlib_cn()


def main() -> None:
    result = build_historical_bond_observation(BASE_DIR)
    print_historical_bond_summary(result)
    print("说明：1965-2025 年历史口径优先采用 A0303 人口年龄结构与抚养比表，缺失年度由项目内人口回推/前推方法补全。")
    print("说明：PSR 采用 20-39 岁 / 50-69 岁，父母代规模采用 50 岁人口；抚养比已校准为 A0303 的 0-14 / 15-64 / 65+ 口径。")
    plot_historical_bond_dashboard(result)


if __name__ == "__main__":
    main()
