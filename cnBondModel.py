from __future__ import annotations

from pathlib import Path

from utils.bond_tools import plot_bond_dashboard, print_bond_summary, simulate_bond_population
from utils.demography import configure_matplotlib_cn


BASE_DIR = Path(__file__).resolve().parent
configure_matplotlib_cn()


def main() -> None:
    result = simulate_bond_population(BASE_DIR, end_year=2070)
    print_bond_summary(result)
    print("说明：模型以 2020 年人口结构为基年，死亡率采用 2020 年年龄别死亡近似，TFR 采用历史三次普查线性外推。")
    print("说明：养老需求年龄口径调整为 65 岁起，平均寿命口径调整为 78 岁，相关 PSR/ODR 指标均按 65-78 岁窗口计算。")
    print("说明：脚本新增了从 2020 向左反推 2010、2000 的验证段，用于检验 2030、2050、2060、2070 等未来假设的稳定性。")
    print("说明：由于 2020 年人口表为 5 岁年龄组，脚本按组内均匀分布扩展到单岁；结果适合作为结构化情景分析，而非官方预测。")
    plot_bond_dashboard(result)


if __name__ == "__main__":
    main()
