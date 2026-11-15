# B200 低精度后训练实验 — 2026-11-15

已在实际 NVIDIA B200 上运行原生低精度前向、反向和真实数据微调。结论：**当前平台的 Qwen3-0.6B 工作负载继续使用 BF16；这次低精度方案没有带来训练 step 加速。** 这是 SFT 与算子可行性实验，没有测量 Miles 的 rollout、权重同步、跨卡通信和完整 RL wall-clock，不能声称复现论文的 RL 加速。

[实验代码](../../../tools/benchmark_blackwell_posttraining.py)、[完整指标与逐步训练记录](results.json)、[精确软件版本](requirements.lock.txt)。所有表格从 `results.json` 中的原始记录计算。失败的实验也保留在记录中。

后续的 [性能原因分析与 CUDA Graph 对照](DIAGNOSIS.md) 进一步拆分了量化、GEMM 和执行开销；原始三轮基准数据保持不变。

## 环境与对照方法

- 服务器有 4 × NVIDIA B200，每张 183359 MiB；训练和算子测量串行使用 GPU 0，模型加载验证使用 GPU 1。
- Python 3.12.15、PyTorch 2.15.0+cu132、CUDA 13.2、Transformer Engine 2.20.2、Transformers 5.23.0；驱动 595.91.07。使用独立虚拟环境，没有修改驱动。
- Qwen/Qwen3-0.6B，模型 revision `c1899de289a04d12100db370d81485cdf75e47ca`。
- GSM8K main，dataset revision `740312add88f781978c0658806c59bc2815b9866`；train 前 128 条、test 前 16 条分别用于训练与验证，问题交集为 0，无答案因序列长度而截断。
- 每组 64 次 AdamW 更新，batch 4、序列长度 512、lr 1e-5、seed 20261115；只对回答计算 loss。FP32 参数和优化器状态，BF16 基础计算。
- 28 层中的前 23 层 MLP 的 69 个线性层换为同一 TE 实现；最后 5 层、attention、embedding、LM head 保留 BF16 计算。所有组都显式把 TE 输入转为 BF16，满足 NVFP4 RHT 的输入约束。
- 四种方案交错顺序各跑三次，使用相同权重、数据哈希与步数。每次先做 3 次无参数更新的前向/反向预热，计时排除前 5 个训练 step。计时包含前向、量化、反向、梯度裁剪和 optimizer step；不含模型下载、加载、数据准备、评估、保存与日志输出。
- 使用 CUDA 同步后的 wall-clock；峰值显存是 PyTorch peak allocated，含 FP32 参数、梯度、优化器状态和临时张量，不等于整卡占用。未启用 torch.compile 或 CUDA Graph。

## 真实 SFT 结果

耗时列为三次运行的中位数；耗时变化为正代表更慢。验证 loss 是回答 token 的平均负对数似然，越低越好。

| 方案 | step 毫秒 | 各次运行范围 | 耗时变化 | 峰值 GiB | 验证 loss |
|---|---:|---:|---:|---:|---:|
| BF16 | 84.4 | 81.8–92.2 | +0.0% | 15.98 | 0.6049 |
| MXFP8 | 94.6 | 92.6–101.0 | +12.1% | 15.82 | 0.6058 |
| NVFP4 | 115.6 | 104.0–116.4 | +36.9% | 15.52 | 0.6186 |
| NVFP4 前向 / MXFP8 反向 | 116.4 | 111.7–123.4 | +37.8% | 15.73 | 0.6100 |

四组均完成更新，梯度和 loss 有限，抽检参数的变化范数大于 0。验证 loss 从约 1.5 降到约 0.6，但验证集只有 16 条，不能据此认定质量等价或模型已达到客户可用水平。相同随机种子的复跑仍存在数值和计时波动；这些是性能复测，不是独立的质量统计试验。`results.json` 另存有各组在统一 BF16 计算下的验证 loss。

## 大矩阵与 MoE 分组算子

大矩阵使用 M=4096、K=4096、N=14336；每次测量包含原生量化、forward、dgrad 和 wgrad，不含优化器。10 次预热，50 次测量，重复三次。Profiler 确认出现 SM100 NVFP4/MXFP8 GEMM 和对应量化内核。

| 方案 | 毫秒 | 各次运行范围 | 耗时变化 |
|---|---:|---:|---:|
| BF16 | 1.1472 | 1.1354–1.1718 | +0.0% |
| MXFP8 | 1.0102 | 0.9820–1.0981 | -11.9% |
| NVFP4 | 1.0919 | 1.0638–1.2326 | -4.8% |
| NVFP4 前向 / MXFP8 反向 | 1.2972 | 1.2736–1.3031 | +13.1% |

NVFP4 初次探索曾测得约 21% 耗时下降；三轮复测的中位数收益只有约 5%，且运行范围有重叠。应使用复测结果，不把单轮最佳值作为宣传数字。

另测了 8 个专家、总 4096 tokens、K=2048、N=768 的 GroupedLinear 前向/反向。它只覆盖分组矩阵算子，不含路由或 all-to-all。普通分组路径四种精度均可运行；融合路径必须传入 GPU int64 的 token splits。打包专家权重还需要显式设置 `NVTE_GROUPED_LINEAR_SINGLE_PARAM=1`，脚本现在会检查该条件，避免参数被库静默忽略。

已确认的融合路径限制（TE 2.20.2）：

- BF16 和 MXFP8 的打包权重路径可运行，见 `fused-grouped-*-packed.json` 对应记录；该小形状下 MXFP8 仍未快于 BF16。
- 默认 NVFP4 失败：打包权重的 2D scaling grouped quant kernel 尚未实现。
- 改为 1D NVFP4 后仍失败：非 RHT 的 graph-safe grouped quant kernel 尚未实现。此变体仅做算子验证，没有 SFT 质量结果。
- NVFP4-forward/MXFP8-backward 自定义配方不支持此原生打包路径。允许退回 split-quantize 时可以运行，但不能把回退结果当成融合内核收益。

## 与 Thinking Machines 方案的关系

参考 [Thinking Machines 原文](https://thinkingmachines.ai/blog/end-to-end-4-bit-training/) 和 [NVIDIA 自定义精度接口](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/fine_grained_quantization/fine_grained_quantization.html)。混合配方把 forward 的 NVFP4 值反量化后，再生成 backward 所需的 MXFP8 表示；采用 1D NVFP4，不使用 RHT。当前 MXFP8 量化器没有该实验所需的 stochastic rounding 接口。

这不是 BBBK 复刻：没有文章中的自研融合内核、量化 all-to-all 通信或完整 MoE RL。标准 NVFP4 对照采用 TE 默认配方，包括 2D 权重量化、RHT 和梯度 stochastic rounding，与混合配方分开记录。

## 模型产物与验证

最终标准 HF 模型位于实验服务器：

- `/workspace/miles-blackwell-experiment/checkpoints/sft-bf16-export-final`
- `/workspace/miles-blackwell-experiment/checkpoints/sft-hybrid-export-final`

导出移除了 TE 专用 quantizer extra state，并写出 BF16 权重。两份模型都已通过原生 Transformers 重新加载，missing/unexpected/mismatched keys 均为空；greedy generation 对 `17 + 25` 都生成了 42。该单题只验证模型可调用，不是数学准确率评测。验证详情见 `checkpoint-validation-final.json` 对应记录。

全部实验进程已退出，四张 GPU 均恢复空闲。实例仍开机；没有关闭用户的服务器。原有平台默认精度、线上部署和公开展示页均未改变。

## 复跑

已准备好的服务器目录为 `/workspace/miles-blackwell-experiment`，代码副本为 `benchmark.py`。例如：

```bash
BENCH_ROOT=/workspace/miles-blackwell-experiment
CUDA_VISIBLE_DEVICES=0 "$BENCH_ROOT/venv/bin/python" "$BENCH_ROOT/benchmark.py" \
  --task sft --mode hybrid --work-dir "$BENCH_ROOT" --steps 64 --run-label manual

NVTE_GROUPED_LINEAR_SINGLE_PARAM=1 CUDA_VISIBLE_DEVICES=0 \
  "$BENCH_ROOT/venv/bin/python" "$BENCH_ROOT/benchmark.py" \
  --task fused-grouped --mode mxfp8 --work-dir "$BENCH_ROOT" \
  --hidden-size 2048 --intermediate-size 768 --steps 50 --profile --run-label manual
```

新环境先安装 lock 文件里的 PyTorch 与 TE core，再编译同版本 `transformer-engine-torch`，使用 `--no-build-isolation`。本实例编译时需要将虚拟环境中 `nvidia/nccl/include` 和 `nvidia/cudnn/include` 加到 `CPATH`，对应 `lib` 目录加到 `LIBRARY_PATH`；`MAX_JOBS=8 NVTE_BUILD_THREADS_PER_JOB=1 NVTE_CUDA_ARCHS=100`。不要升级宿主 NVIDIA 驱动。新数据目录用脚本 `--task prepare` 准备，并记录解析到的模型/数据 revision；要严格复现本次实验，应固定为上述 revision。

下一阶段应在代表客户工作负载的 MoE 模型上解决融合内核覆盖，并同时测量 rollout、训练、同步和独立质量评测。当前证据不支持直接为平台开启默认 4-bit 训练。
