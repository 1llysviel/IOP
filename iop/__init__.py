"""人口超长期债券模型 v2：数据层、死亡模型、生育模型。

模块：
- census     三次普查（2000/2010/2020）读取，统一口径与抽样放大
- wpp        UN WPP 2024 中国数据读取
- graduation 5 岁组拆分为单岁（累计量单调样条）
- lifetable  生命表与 Kannisto 高龄平滑
- leecarter  Lee–Carter 随机死亡率模型
- fertility  生育年龄曲线、进度效应、TFR 贝叶斯 AR(1) 模型
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "Dataset"
EXTERNAL_DIR = DATASET_DIR / "external"
