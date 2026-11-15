# 为什么这次 MXFP8 比 BF16 慢

这里的 8-bit 格式是 **MXFP8**；上轮另外测试的 4-bit 格式是 NVFP4。已在同一台 B200 服务器上补充 PyTorch/CUDA profiling、矩阵形状对照、batch 对照和 CUDA Graph 实验。

结论：**这版 HF/PyTorch eager 接入主要受 CPU/框架调度与算子发射开销限制；小矩阵上的量化成本又抵消了 MXFP8 的 GEMM 节省。不能把这个结果外推为 B200 的 FP8 硬件慢。**

[原始基准](README.md)仍是三轮 64 步的数据，本报告没有覆盖它。[诊断代码](../../../tools/profile_blackwell_posttraining.py)和[全部诊断指标](diagnosis.json)可用于复核。原始 Chrome traces 与日志保留在服务器的 `/workspace/miles-blackwell-experiment/diagnostics`，并已离机备份。

## 1. 用 CUDA Graph 验证执行开销

同一 Qwen3-0.6B、精度配方、FP32 主权重、数据准备、batch 4 × 512。每种方法先做 8 次更新预热，再测量 30 次更新；下表是每步 wall-clock 中位数，单位 ms。测量不包含 profiler，profiling 在计时结束后另做。

| 执行方式 | BF16 | MXFP8 |
|---|---:|---:|
| Eager，batch 4 | 81.70 | 101.28 |
| 预计算 mask，仍 Eager，batch 4 | 89.85 | 102.01 |
| CUDA Graph，batch 4 | 42.34 | 42.57 |
| Eager，batch 16 | 134.73 | 138.22 |

CUDA Graph 只捕获 forward/backward；梯度裁剪和 AdamW 仍在图外执行。Graph 的 mask 在捕获前准备，因此额外做了“预计算 mask、继续 eager”的控制组；它没有得到 Graph 的明显改善。Eager 测量存在主机侧波动，应看控制组和直接证据，不将单次比率当作生产承诺。

正确性检查：两种精度的图内/图外初始 loss、抽检权重梯度一致；预计算 mask 与原 mask 的 loss 一致；Graph 与 eager 的 30 步 loss 序列逐项一致；权重变化范数大于 0。不同精度之间不要求 loss 一致。首次捕获遇到的 AccumulateGrad stream 警告已通过同一 side stream 的初始化、预热、捕获和梯度检查消除；最后两次验证无该警告。

这是**固定第一个 batch、固定形状的诊断**，不代表真实多 batch、变长样本或分布式训练已支持 CUDA Graph，也不是新的模型质量评估。没有修改平台训练默认值或上线模型。

## 2. GEMM 确实加速，但不足以支付量化成本

以下是各自 3 个 profiled steps 的 CUDA kernel duration 总和除以 3；“其余”包含 attention、逐元素运算、类型转换、优化器等。它不是训练 wall-clock，不能与未开启 profiler 的耗时直接相加或相减。

| 精度 | GEMM ms | 量化/RHT ms | 其余 GPU ms | kernel 总和 ms | kernels/step |
|---|---:|---:|---:|---:|---:|
| BF16 | 6.848 | 0.000 | 37.419 | 44.266 | 5334 |
| MXFP8 | 6.176 | 1.276 | 37.099 | 44.551 | 5472 |
| NVFP4 | 5.823 | 4.514 | 37.744 | 48.082 | 6162 |

MXFP8 的 GEMM 每步约省 0.67 ms，量化却新增约 1.28 ms。部分 BF16 转换工作被量化替代，所以全部 GPU kernel 的总和只增加约 0.28 ms。远大于这个差额的 eager wall-clock 差距，结合 CUDA Graph 控制实验，指向 CPU/框架组织、对象与临时缓冲区管理、kernel launch 及其间的等待。

MXFP8 每步增加 207 次量化 kernel，替代部分原有转换，净增加约 138 个 kernels。Profiler 中 TE 的 `_Linear`/`_LinearBackward` 和 launch 调用的 CPU 时间也增加。Profiler 自身会放大 CPU 时间，因此不把 trace 里的空白占比当作未插桩时的精确 GPU 空闲率。

## 3. 矩阵太小：计算收益不够摊薄固定开销

本模型 hidden=1024、intermediate=3072。batch 4 × seq 512 时，上投影形状是 M=2048、K=1024、N=3072。下面用相同 FP32 权重、BF16 输入和实际量化配方，对单 Linear 的 forward+dgrad+wgrad 做 CUDA Graph replay，尽量去掉 Python 发射噪声。数字包含量化与转换，不包含 optimizer。

| M × K × N | BF16 μs | MXFP8 μs | MXFP8 耗时变化 |
|---|---:|---:|---:|
| 2048 × 1024 × 3072 | 39.18 | 47.30 | +20.7% |
| 2048 × 3072 × 1024 | 41.14 | 45.66 | +11.0% |
| 8192 × 1024 × 3072 | 115.69 | 106.17 | -8.2% |
| 4096 × 4096 × 14336 | 1155.43 | 739.71 | -36.0% |

当前小形状下，去掉发射开销后 MXFP8 仍慢；更大形状开始受益。实际 SFT 的 batch 从 4 增到 16 后，eager 的差距也收窄到约 2.6%。它说明算术强度和每次调用的工作量很重要，不能只看 GPU 的标称低精度峰值。

NVIDIA 的[官方性能说明](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/speedups.html)也分别报告含量化的 autocast 和预量化的纯 GEMM，两者不能混用；[MXFP8 格式说明](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/mxfp8/mxfp8.html)解释了 block-32 scaling 和不同方向所需的量化表示。

## 4. 当前接入覆盖有限，其他工作占了大头

只替换了前 23 层 MLP 的 69 个 Linear；模型共有 28 层，attention、最后 5 层 MLP、embedding、151936 词表的输出层均保留原计算。约 5,300 个 kernels/step 中，大量是 FP32 逐元素运算、归约、类型转换等；加快 Tensor Core GEMM 无法消除它们。

当前原型使用 FP32 参数和优化器状态、自动混合精度计算，普通 RMSNorm、激活、loss 等未系统融合。它适合验证能否训练，但不能代表 Megatron/Transformer Engine 完整融合栈的最优性能。Profiler 中的输出层 forward GEMM 本身并不是最大瓶颈；不能把问题简单归因于大词表。

## 建议的改进顺序

1. 先验证固定形状分桶、稳定 buffer、CUDA Graph 在真实多 batch 训练中的正确性和收益，包括换 batch、不同长度、梯度累积与 checkpoint。
2. 减少大量小 kernel，评估融合 RMSNorm、MLP 与 loss，同时保持合适的 optimizer master-weight 精度。
3. 在这个更强的 BF16 基线上，重新测量 MXFP8；选择更有代表性的模型形状、批量和 MoE 工作负载，并纳入完整 rollout/训练/同步耗时及独立质量评测。

这次已验证的优化机会首先是执行路径。Graph 诊断下 BF16 与 MXFP8 都在约 42 ms/step，当前形状上 MXFP8 几乎没有额外收益。

## 复跑诊断

保持上轮实验环境与数据不变，在仓库根目录运行：

```bash
python -m tools.profile_blackwell_posttraining --task sft --mode mxfp8 --work-dir /workspace/miles-blackwell-experiment
python -m tools.profile_blackwell_posttraining --task sft-static --mode mxfp8 --work-dir /workspace/miles-blackwell-experiment
python -m tools.profile_blackwell_posttraining --task sft-graph --mode mxfp8 --work-dir /workspace/miles-blackwell-experiment
python -m tools.profile_blackwell_posttraining --task linear --mode mxfp8 --work-dir /workspace/miles-blackwell-experiment
```

所有实验已结束，GPU 已恢复空闲。未替换已有 checkpoint 或修改公开展示页。
