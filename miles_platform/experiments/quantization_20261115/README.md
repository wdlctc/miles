# 训练后量化（PTQ）保真度实验 — 2026-11-15

对 Qwen3-0.6B 做 FP8 与 NVFP4 训练后量化，并在留出集上测量相对 BF16 的保真度。结论：**FP8 权重量化的 KL 为 0.012；NVFP4 只量化 MLP 时，local-Hessian 选 block scale 比取最大值的 KL 低 15%（含激活量化）到 33%（仅权重），方法排序与 Beam 技术报告一致。** 这是数值格式的浮点仿真，没有运行打包的低精度内核，不包含任何速度或显存结论，也没有做任务准确率评测。

[量化代码](../../quantize.py)、[数据准备脚本](prepare_gsm8k.py)、[原始结果](results.json)。表格全部由 `results.json` 计算。

## 方法

参照 Beam 技术报告 4.5 节的两种格式：

- **FP8**：权重和输入激活存为 E4M3。权重每个输出通道一个固定 scale（`max|w| / 448`），激活每个 token 动态计算 scale，round-to-nearest，不需要校准数据。
- **NVFP4**：数值存为 E2M1，每 16 个元素一个 E4M3 block scale，每个张量一个 FP32 scale。block scale 有三种选法：
  - `max`：block 最大值映射到 6（round-to-nearest）。
  - `four_over_six`：在"最大值映射到 6"和"映射到 4"之间按平方误差二选一。
  - `local_hessian`：在 13 个候选 scale 中，按该 block 输入激活的 16×16 二阶矩加权的输出误差选最优。统计量来自未量化的 BF16 模型，每层看到的都是参考激活。

与报告的差异：

- 报告的 NVFP4 只量化 MoE expert。Qwen3-0.6B 是稠密模型，这里以 MLP（gate/up/down 三个投影）代替，另测了把 attention 投影也量化的对照。
- 报告用 LLM Compressor 和 NVIDIA Model Optimizer 产出可直接部署的打包权重。这里是自包含的仿真：量化后的每个值都能被目标格式精确表示，但仍以 BF16 张量保存和计算。
- `max` 方式的 NVFP4 权重量化结果与 Model Optimizer 0.41.0 的 `NVFP4QTensor.quantize` / `dequantize` 在随机张量上逐元素相同（该对照在另一台 B200 机器上完成）。`four_over_six` 和 `local_hessian` 是按报告描述自行实现的，没有与官方实现对照。
- embedding、LM head、norm 在所有方案中都保持原精度。

## 环境与数据

- NVIDIA A100-SXM4-40GB 单卡，PyTorch 2.14.1+cu130、Transformers 5.19.0。
- Qwen/Qwen3-0.6B，revision `c1899de289a04d12100db370d81485cdf75e47ca`，BF16 加载。
- GSM8K main，revision `740312add88f781978c0658806c59bc2815b9866`。校准用 train 前 512 条（102,234 tokens），留出用 test 前 128 条（24,542 tokens），问题无交集。每条渲染为"用户问题 + 参考解答"的对话，截断到 512 tokens。
- 指标是 teacher-forced 的逐 token KL(BF16‖量化)（单位 nats）和 top-1 一致率，在留出集全部非 padding token 上取平均。

## 结果

| 格式 | scale 选法 | 量化的层 | 激活 | 平均 KL ↓ | top-1 一致率 | 权重相对误差 |
|---|---|---|---|---:|---:|---:|
| FP8 | max | attention + MLP（196 层） | 量化 | 0.0216 | 94.51% | 2.64% |
| FP8 | max | attention + MLP（196 层） | 不量化 | 0.0120 | 95.77% | 2.64% |
| NVFP4 | max | MLP（84 层） | 量化 | 0.1503 | 85.63% | 9.49% |
| NVFP4 | four_over_six | MLP（84 层） | 量化 | 0.1472 | 86.12% | 8.69% |
| NVFP4 | local_hessian | MLP（84 层） | 量化 | 0.1276 | 87.38% | 8.69% |
| NVFP4 | max | MLP（84 层） | 不量化 | 0.0901 | 89.18% | 9.49% |
| NVFP4 | four_over_six | MLP（84 层） | 不量化 | 0.0831 | 89.45% | 8.69% |
| NVFP4 | local_hessian | MLP（84 层） | 不量化 | 0.0605 | 91.19% | 8.69% |
| NVFP4 | local_hessian | attention + MLP（196 层） | 量化 | 0.2024 | 84.23% | 8.70% |
| NVFP4 | local_hessian | attention + MLP（196 层） | 不量化 | 0.1002 | 88.95% | 8.70% |

- 三种 scale 选法的排序与报告的表 13 相同：local_hessian 最好，four_over_six 次之，max 最差。
- local_hessian 和 four_over_six 的权重相对误差相同（8.69%），KL 却差 13%–27%：按输出误差选 scale 比按权重误差选更有效，权重误差本身不能用来排序方案。
- 把 attention 也量化到 NVFP4 后 KL 增加约 59%–66%，与报告"只量化 expert"的取舍方向一致。
- 激活量化贡献了相当一部分误差：FP8 的 KL 因此从 0.012 升到 0.022，NVFP4 从 0.060 升到 0.128。
- 这里的 KL 绝对值远高于报告（报告 NVFP4 约 0.03，top-1 约 96.8%）。两者模型规模、被量化的层和数据都不同，不能直接比较。

## 未验证的事项

- 只有一个模型、一个数据集、单次运行，没有重复和方差估计。
- 没有任务级评测，不能据此判断某个量化模型对客户是否可用。
- 没有产出 SGLang 可加载的打包 FP8/NVFP4 权重，也没有测量部署后的吞吐或显存。
- 平台的量化任务用客户数据集的 prompt（不含回答）做校准和留出。在这种数据上同一方案的 KL 明显更高（NVFP4 local_hessian 仅权重为 0.137，FP8 为 0.042，留出 5,825 tokens），不能与上表直接比较。

## 复跑

```bash
python miles_platform/experiments/quantization_20261115/prepare_gsm8k.py --output-dir data
python -m miles_platform.quantize --model-dir /path/to/Qwen3-0.6B \
  --calibration data/calibration.jsonl --heldout data/heldout.jsonl --batch-size 16 \
  --report results.json \
  --recipe fp8 --recipe fp8:max:attention+mlp:w \
  --recipe nvfp4:max:mlp --recipe nvfp4:four_over_six:mlp --recipe nvfp4:local_hessian:mlp \
  --recipe nvfp4:max:mlp:w --recipe nvfp4:four_over_six:mlp:w --recipe nvfp4:local_hessian:mlp:w \
  --recipe nvfp4:local_hessian:attention+mlp --recipe nvfp4:local_hessian:attention+mlp:w
```

`--save DIR` 会把第一个方案的量化权重另存为标准 HF 目录，并写出 `quantization.json`。
