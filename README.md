# IOP：人口超长期债券模型

把人口世代更替类比为超长期债券：出生 = 发行，20 岁 = 入场，65 岁以上 = 兑付。

## 结构

| 目录 | 内容 |
|---|---|
| `iop/` | v2 模型包：数据层（`census`、`wpp`、`official`、`graduation`）、漏报校正（`coverage`、`ddm`）、死亡模型（`lifetable`、`leecarter`）、生育模型（`fertility`） |
| `notebooks/` | 讲解 notebook：`01_数据地基`、`02_死亡模型`、`03_生育模型` |
| `tests/` | 单元测试（生命表恒等式、参数还原等） |
| `utils/`、`cn*.py` | v1 脚本（保留未改动） |

## 数据

- `Dataset/*.xls`：2000/2010/2020 年人口普查表（不入库）。
- `Dataset/external/*_CHN.csv`：UN World Population Prospects 2024 中国子集，来自 population.un.org/wpp 官方 CSV（只保留 LocID=156 的行）。

## 运行

```bash
uv sync
uv run pytest
uv run jupyter lab notebooks/
```
