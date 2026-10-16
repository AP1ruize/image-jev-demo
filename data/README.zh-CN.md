# 本地数据

[English](README.md)

第三方图像、标注和训练导出留在磁盘上，不要提交。`data/exports/` 故意忽略：这些 JSONL 复制了数据集分数和本机路径。

```text
data/raw/          官方压缩包或解压结果
data/interim/      从这些压缩包建出的来源目录
data/processed/    冻结的评测条目
data/exports/      训练导出
data/manifests/    可选的额外索引
```

路径写在 `configs/local.json`（从 `configs/local.example.json` 复制）。图像树本身在 `datasets/`。见 [docs/zh/DATASETS.md](../docs/zh/DATASETS.md)。

代码是 Apache-2.0。数据集版权属于原作者。
