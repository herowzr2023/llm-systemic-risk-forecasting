# 论文图表独立复现

每个子目录都包含一个可独立运行的 Python 脚本、所需的 `data/` 输入和 `output/` 输出。进入对应子目录后运行 `python generate_*.py` 即可重新生成该图或表。

也可在本目录下运行：

```powershell
python run_all.py
```

该命令仅运行论文中需要数据生成的表 3–5、表 A2–A4、表 A6–A7、图 2 和图 A1–A2。主文表 1、表 2、表 6，附录表 A1、表 A5，以及主文图 1、图 3 为研究设计、变量定义、超参整理或架构示意，不提供数值生成脚本。

表 3–5 和图 A1–A2 的底层 CSV 保留完整模型列；论文展示层只选取 Informer、Persistence、AR(1)、ETS、GBR 和 PatchTST。
