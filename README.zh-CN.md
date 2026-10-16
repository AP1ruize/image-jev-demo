# ImageJev

[English](README.md)

用小视觉语言模型做摄影技术质量评分，一次前向读出分数。

有用的结果是：在冻结的 4-bit `Qwen/Qwen3.5-2B` 上加**语言 LoRA 和指针头**。SPAQ 手机照片测试集上，排序相关只比“生成 JSON 分数”的同一模型低 0.007 个 Spearman，延迟大约是它的三分之一。区间和设置见 [docs/zh/RESULTS.md](docs/zh/RESULTS.md)。

## SPAQ 测试集，1,658 张

同一套 Week 0 切分，像素上限 65,536，视觉编码器冻结。主指标是 Spearman ρ。延迟是 RTX A2000 Laptop（4 GB）上的中位模型调用时间。

| 读法 | Spearman ρ | MAE | 中位延迟 |
| --- | ---: | ---: | ---: |
| 只训指针头 | 0.824 | 9.27 | 209 ms |
| 语言 LoRA + 指针头，最后一步 | 0.896 | 7.04 | 406 ms |
| 语言 LoRA + 指针头，按验证损失选出的第 900 步 | 0.895 | 7.27 | 410 ms |
| 4-bit QLoRA，生成 JSON | 0.902 | 6.78 | 1,374 ms |

最后一步的指针头比 JSON 生成低 0.006 个 Spearman。差距不大，但 95% 区间不包含零。按验证损失选出的第 900 步没有超过最后一步。每一行的覆盖率都是 1：指针头给出的都是合法分布。

## 这是什么

ImageJev 评的是**技术 / 感知质量**（清晰度、噪声、曝光、总体 MOS）。它不评美，不评构图，也不写影评。

网络不逐字解码答案。冻结的视觉编码器和 4-bit 语言模型，加上一小块语言 LoRA，得到一个隐状态。指针头拿这个状态和固定档位比较（SPAQ 上是 0、10、……、100）。报出的分数是档位的期望。需要 JSON 时，由 Python 组装。

训练之前的 Week 0 在同一批题上比较三种冻结读出：自由生成 JSON、JSON 数字语法约束、直接读 logits。训过的指针头是这件事的后续。见 [docs/zh/WEEK0.md](docs/zh/WEEK0.md) 和 [docs/zh/EVAL.md](docs/zh/EVAL.md)。

## 环境

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

GPU 训练和评测需要先装 CUDA 版 PyTorch，然后：

```powershell
pip install -r requirements-eval.txt
```

把 `configs/local.example.json` 抄成 `configs/local.json`，指向你自己的 SPAQ 和 KonIQ 文件。数据集不在本仓库里。版权属于原作者，见 [docs/zh/DATASETS.md](docs/zh/DATASETS.md)。

```powershell
python scripts/build_source_catalog.py
python scripts/build_week0_release.py
python scripts/eval_week0.py --config configs/models.qwen3vl2b.json --mode generate --dry-run
```

冻结切分准备好之后，SPAQ 指针训练的入口是：

```powershell
python scripts/train_warmup.py --config configs/warmup.spaq.json
python scripts/run_spaq_heldout.py --config configs/warmup.spaq.heldout.json
python scripts/train_spaq_pointer_lora.py --config configs/pointer-lora.spaq.qwen35-2b.json
```

## 目录

```text
configs/          模型配置和 Week 0 切分
docs/             协议、数据集和 SPAQ 结果
registry/         来源链接和上游许可说明
schemas/          题目和预测的 JSON Schema
scripts/          建目录、冻结、训练、评测、打分
src/image_jev/    打分、冻结推理、指针头
tests/            不需要 GPU
```

## 许可

本仓库代码是 Apache-2.0，代码可以商用。

数据集的版权和条款属于原作者。本仓库不授予这些图像或标注的权利。这里报告的训练和评测是在这些数据集上做的个人研究试验。
