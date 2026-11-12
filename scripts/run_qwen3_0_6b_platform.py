"""Bounded Qwen3-0.6B GRPO validation for a single-node training-to-serving path.

Requires a downloaded Qwen3-0.6B HF checkpoint, train.jsonl and eval.jsonl
with `prompt` (chat messages) and `label` fields. Uses separate FSDP trainers
and SGLang rollout engines. FA2 and Triton attention support the A100 target.
This is an integration smoke run, not a convergence or performance benchmark.

Args:
    --model-dir: Parent directory containing Qwen3-0.6B.
    --data-dir: Directory containing the disjoint train/eval JSONL subsets.
    --output-dir: Checkpoint and TensorBoard output directory; use a fresh path.
    --num-gpus-per-node / --trainer-gpus: Total GPUs and the training allocation.
    --num-rollout: Number of updates (default: two).

Example:
    python scripts/run_qwen3_0_6b_platform.py \
        --model-dir /root/models --data-dir /root/datasets/platform-smoke \
        --output-dir /root/shared_data/platform-smoke
"""

import shlex
from dataclasses import dataclass
from pathlib import Path

import typer

from miles.utils.external_utils import command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets/platform-smoke"
    num_gpus_per_node: int = 4
    trainer_gpus: int = 2
    num_rollout: int = 2
    rollout_batch_size: int = 4
    n_samples_per_prompt: int = 4
    max_response_len: int = 512
    max_tokens_per_gpu: int = 2048
    learning_rate: float = 1e-6
    graceful_ray_stop: bool = True

    def __post_init__(self):
        if self.num_nodes != 1:
            raise ValueError("The platform smoke recipe supports one node.")
        if not 0 < self.trainer_gpus < self.num_gpus_per_node:
            raise ValueError("Reserve at least one GPU each for training and rollout.")
        if min(self.num_rollout, self.rollout_batch_size, self.max_response_len, self.max_tokens_per_gpu) < 1:
            raise ValueError("Rollout count, batch size and token limits must be positive.")
        if self.n_samples_per_prompt < 2:
            raise ValueError("GRPO needs at least two samples per prompt in this recipe.")
        if self.global_batch_size % self.trainer_gpus:
            raise ValueError("The sample batch must divide evenly across training GPUs.")

    @property
    def global_batch_size(self) -> int:
        return self.rollout_batch_size * self.n_samples_per_prompt


def _data_flags(args: ScriptArgs) -> list[str]:
    output = Path(args.output_dir)
    checkpoint = [
        "--hf-checkpoint",
        str(Path(args.model_dir) / "Qwen3-0.6B"),
        "--save",
        str(output / "checkpoints"),
        "--save-interval",
        "1",
    ]
    rollout = [
        "--prompt-data",
        str(Path(args.data_dir) / "train.jsonl"),
        "--input-key",
        "prompt",
        "--label-key",
        "label",
        "--apply-chat-template",
        "--apply-chat-template-kwargs",
        '{"enable_thinking":false}',
        "--rollout-shuffle",
        "--balance-data",
        "--rm-type",
        "math",
        "--num-rollout",
        str(args.num_rollout),
        "--rollout-batch-size",
        str(args.rollout_batch_size),
        "--n-samples-per-prompt",
        str(args.n_samples_per_prompt),
        "--global-batch-size",
        str(args.global_batch_size),
        "--rollout-max-response-len",
        str(args.max_response_len),
        "--rollout-temperature",
        "1",
    ]
    evaluation = [
        "--eval-interval",
        "1",
        "--eval-prompt-data",
        "gsm8k_smoke",
        str(Path(args.data_dir) / "eval.jsonl"),
        "--n-samples-per-eval-prompt",
        "1",
        "--eval-max-response-len",
        str(args.max_response_len),
        "--eval-temperature",
        "0",
    ]
    return checkpoint + rollout + evaluation


def _training_flags(args: ScriptArgs) -> list[str]:
    algorithm = [
        "--advantage-estimator",
        "grpo",
        "--kl-coef",
        "0",
        "--kl-loss-coef",
        "0",
        "--entropy-coef",
        "0",
        "--eps-clip",
        "0.2",
        "--eps-clip-high",
        "0.28",
    ]
    optimizer = [
        "--optimizer",
        "adam",
        "--lr",
        str(args.learning_rate),
        "--lr-decay-style",
        "constant",
        "--weight-decay",
        "0.1",
        "--adam-beta1",
        "0.9",
        "--adam-beta2",
        "0.98",
    ]
    backend = [
        "--train-backend",
        "fsdp",
        "--actor-num-nodes",
        "1",
        "--actor-num-gpus-per-node",
        str(args.trainer_gpus),
        "--num-gpus-per-node",
        str(args.num_gpus_per_node),
        "--rollout-num-gpus",
        str(args.num_gpus_per_node - args.trainer_gpus),
        "--rollout-num-gpus-per-engine",
        "1",
        "--gradient-checkpointing",
        "--attn-implementation",
        "flash_attention_2",
        "--update-weight-buffer-size",
        "134217728",
        "--use-dynamic-batch-size",
        "--max-tokens-per-gpu",
        str(args.max_tokens_per_gpu),
    ]
    return algorithm + optimizer + backend


def _serving_flags() -> list[str]:
    return [
        "--sglang-attention-backend",
        "triton",
        "--sglang-sampling-backend",
        "pytorch",
        "--sglang-mem-fraction-static",
        "0.45",
        "--sglang-context-length",
        "2048",
        "--sglang-chunked-prefill-size",
        "1024",
        "--sglang-disable-cuda-graph",
        "--sglang-log-level",
        "warning",
    ]


def execute(args: ScriptArgs):
    output = Path(args.output_dir)
    if (output / "checkpoints").exists():
        raise ValueError("Choose a fresh --output-dir to preserve previous checkpoints.")
    tracking = ["--use-tensorboard", "--tb-project-name", "platform", "--tb-experiment-name", "qwen3-smoke"]
    args.create_backend().execute_train(
        train_args=shlex.join(_data_flags(args) + _training_flags(args) + _serving_flags() + tracking),
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=None,
        job_lifetime="launcher",
        extra_env_vars={"TENSORBOARD_DIR": str(output / "tensorboard"), "OMP_NUM_THREADS": "4"},
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
