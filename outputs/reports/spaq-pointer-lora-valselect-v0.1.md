# spaq-pointer-lora-valselect-v0.1 对比报告

用途：非商业内部模型实验。同一份 SPAQ Week 0 完整训练集、验证集和测试集。
这一版在冻结视觉编码器的 4-bit Qwen3.5-2B 上同时更新语言 LoRA 和指针头，推理仍是一次前向。
正式权重按验证集指针交叉熵选取，不是最后一个 optimizer step。测试集不参与选模。
对照是同一配置的最后一步语言 LoRA + 指针头，以及生成 JSON 的 4-bit QLoRA。

## 模型与运行元数据

- 验证集选出的语言 LoRA + 指针头：`qwen3.5-2b-4bit-pointer-lora-valselect`，epoch 1，LoRA r=8
- 头初始化：`outputs/warmup/spaq-pointer-full-v0.1/pointer-head.pt`
- 权重目录：`outputs/qlora/spaq-pointer-lora-v0.1-valselect/best`
- 最后一步对照：`qwen3.5-2b-4bit-pointer-lora`，optimizer step 976
- JSON QLoRA：`qwen3.5-2b-4bit-qlora-full`，epoch 1，LoRA r=8
- 训练 / 验证 / 测试张数：7807 / 1660 / 1658
- 像素上限：65536
- 训练顺序种子：42。前 5 个训练下标：2161,6276,61,5616,680
- 选模监控：每 100 个 optimizer step 以及最后一步，在固定种子子集上算指针交叉熵
- 监控子集：256 / 1660，种子 42
- 选中 step：900
- 监控验证交叉熵：1.3611
- 全量验证交叉熵：1.3442（n=1660）
- 训练损失 起点 / 选中 step / 终点：1.3014 / 1.9357 / 1.4735

监控记录：
  - step 100：训练损失 1.9294，验证交叉熵 1.6028
  - step 200：训练损失 1.5941，验证交叉熵 1.4508
  - step 300：训练损失 1.9531，验证交叉熵 1.5976
  - step 400：训练损失 1.1880，验证交叉熵 1.6431
  - step 500：训练损失 1.0674，验证交叉熵 1.4498
  - step 600：训练损失 1.4089，验证交叉熵 1.4079
  - step 700：训练损失 1.3060，验证交叉熵 1.3821
  - step 800：训练损失 1.1893，验证交叉熵 1.3843
  - step 900：训练损失 1.9357，验证交叉熵 1.3611
  - step 976：训练损失 1.4735，验证交叉熵 1.4176

## 分来源原生指标

只报告 SPAQ 0–100 MOS。主指标 95% 区间按 `group_id` 做 cluster bootstrap。

### spaq / validation-selected language LoRA + pointer

- n_total / n_valid / coverage：1658 / 1658 / 1.0000
- schema_valid：1.0000
- Spearman ρ：0.8949  CI95 [0.8849, 0.9035]
- Pearson r：0.9050
- MAE：7.2670
- 中位延迟：410.33 ms

### spaq / last-step language LoRA + pointer

- Spearman ρ：0.8959  CI95 [0.8858, 0.9046]
- Pearson r：0.9054
- MAE：7.0382
- 中位延迟：405.89 ms

### spaq / JSON QLoRA

- Spearman ρ：0.9021  CI95 [0.8925, 0.9102]
- Pearson r：0.9133
- MAE：6.7830
- 中位延迟：1374.03 ms

## 配对比较

Delta 是（验证集选出的语言 LoRA + 指针头）减对照的 Spearman ρ。正值表示这一版排序更好。

- 相对最后一步语言 LoRA + 指针头：Δρ -0.0010  CI95 [-0.0029, 0.0009]
- 相对 JSON QLoRA：Δρ -0.0072  CI95 [-0.0104, -0.0038]
- paired valid：1658 / 1658

## 推理时间与资源

- 验证集选出的模型中位 / p95 延迟：410.33 / 590.10 ms
- 峰值 GPU allocated / reserved：2008.40 / 2290.00 MB

## 视觉 merger 探针

探针在同一 4-bit NF4 加载、语言 LoRA 和 `use_reentrant=False` 梯度检查点下，只把视觉 merger 设为可训练，用一张图做一次前向和反向。
这次加载里 merger 的线性层本身是 NF4，不能直接打开梯度，所以探针只把这两层反量化成 fp16。其余模块不改精度。
视觉编码器的其余参数保持冻结。探针不更新正式权重。

- merger 模块：['base_model.model.model.visual.merger']
- merger 参数量：25174016
- merger 权重：nf4-dequantized-to-fp16
- OOM：False
- 梯度已回传：True
- 峰值 allocated / reserved：4045.80 / 4878.00 MB
- 显存总量：4095.56 MB
- nvidia-smi 已用 / 总量：3907.00 / 4096.00 MB
- 余量：-782.44 MB（门槛 512 MB）
- 不建议解冻视觉 merger。峰值显存离 4GB 总量不到 512MB，没有余量。没有启动 merger 训练。

## 覆盖边界与局限

- 只覆盖 SPAQ 总体感知质量。
- 视觉编码器在这次训练里保持冻结。语言 LoRA 与指针头一起用指针交叉熵训练一个 epoch。
- 选模用固定种子的验证子集做监控，选中的 checkpoint 再在全部验证集上确认交叉熵。正式分数用这份权重，不用最后一步。
- 内部实验用途不授权商业发布。
