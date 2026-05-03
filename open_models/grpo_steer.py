# grpo_steer.py
#
# grpo.py + activation steering during the policy forward pass.
#
# Hooks are registered directly on the model (not via a GRPOTrainer subclass)
# so they fire on every model.forward() call regardless of Unsloth patching.
# vLLM rollouts run in a separate process and are unaffected.

from unsloth import FastLanguageModel
import json
import os
import sys
import numpy as np
import time
import random
import shutil
import torch
from typing import List, Dict
from datasets import Dataset
from transformers import TrainerCallback
from validate import TrainingConfig
from utils import load_model_and_tokenizer
from rl.reward import OpenAIGraderReward
from rl.grader_prompts import SYSTEM_PROMPT_RL
import json as _json
from trl import GRPOConfig, GRPOTrainer

# vllm SamplingParams is not JSON-serializable; patch the encoder so TensorBoard
# doesn't crash when it serializes GRPOConfig training args.
_orig_json_default = _json.JSONEncoder.default
def _patched_json_default(self, obj):
    try:
        return _orig_json_default(self, obj)
    except TypeError:
        return repr(obj)
_json.JSONEncoder.default = _patched_json_default

REASONING_GRADERS = ["rhetoric_justdepth", "rhetoric_confirmatory",]


# ── Steering additions ────────────────────────────────────────────────────────

class SteeringHook:
    """Forward hook that adds a steering vector to all token positions."""

    _LOG_FIRST_N = 3
    _LOG_INTERVAL = 100

    def __init__(self, vector: torch.Tensor, alpha: float = 1.0):
        self._vector_cpu = vector.float().cpu().squeeze()  # [d_model]
        self._vector_cache = None
        self.alpha = alpha
        self._fire_count = 0

    def __call__(self, module, input, output):
        act = output[0] if isinstance(output, tuple) else output

        if (self._vector_cache is None
                or self._vector_cache.device != act.device
                or self._vector_cache.dtype != act.dtype):
            self._vector_cache = self._vector_cpu.to(device=act.device, dtype=act.dtype)

        act = act + self.alpha * self._vector_cache  # [B, T, d_model] + [d_model]

        self._fire_count += 1
        if (self._fire_count <= self._LOG_FIRST_N
                or self._fire_count % self._LOG_INTERVAL == 0):
            print(
                f"[SteeringHook] FIRED #{self._fire_count} | "
                f"alpha={self.alpha} | shape={tuple(act.shape)} | "
                f"all_tokens={act.shape[1]} | "
                f"grad_enabled={torch.is_grad_enabled()}"
            )

        return (act,) + output[1:] if isinstance(output, tuple) else act


def _resolve_submodule(model, hookpoint: str):
    for path in (hookpoint, f"base_model.{hookpoint}", f"base_model.model.{hookpoint}"):
        try:
            return model.get_submodule(path)
        except AttributeError:
            continue
    raise ValueError(
        f"Cannot find {hookpoint} in model. "
        f"Top-level modules: {[n for n, _ in list(model.named_modules())[:15]]}"
    )


def add_steering_hooks(model, steering_hook_dict: dict) -> list:
    """Register steering hooks on model; returns list of handles for later removal."""
    handles = []
    for hookpoint, hook in steering_hook_dict.items():
        submodule = _resolve_submodule(model, hookpoint)
        handle = submodule.register_forward_hook(hook)
        handles.append(handle)
        print(f"✓ Steering hook registered at {hookpoint}")
    return handles


def remove_steering_hooks(handles: list):
    for handle in handles:
        handle.remove()
    print(f"✓ {len(handles)} steering hook(s) removed")


def load_steering_vectors(steering_config: dict) -> dict:
    """Load steering vectors from file; returns {hookpoint: tensor}."""
    vector_path = steering_config['steering_vector_path']
    layers = steering_config.get('layers', [])

    print(f"Loading steering vectors from {vector_path}")
    loaded = torch.load(vector_path, weights_only=False)

    intervention_dict = {}
    for layer in layers:
        vector = loaded[layer].unsqueeze(0)  # [1, d_model]
        hookpoint = f"model.layers.{layer - 1}"
        intervention_dict[hookpoint] = vector
        print(f"  Layer {layer} → {hookpoint}, shape: {vector.shape}")

    return intervention_dict


# ── Below is grpo.py verbatim ─────────────────────────────────────────────────

class GradClipCallback(TrainerCallback):
    """Patches clip_grad_norm_, logs per-step to stdout, and reports
    grad_clip_rate + grad_norm_raw_mean to the trainer (→ TensorBoard) at
    every logging interval."""

    def __init__(self, max_grad_norm: float):
        super().__init__()
        self.max_grad_norm = max_grad_norm
        self._total_clipped = 0
        self._total_steps = 0
        self._interval_clipped = 0
        self._interval_steps = 0
        self._interval_norms: list = []

        original = torch.nn.utils.clip_grad_norm_
        cb = self

        def _patched(parameters, max_norm, *args, **kwargs):
            total_norm = original(parameters, max_norm, *args, **kwargs)
            norm_val = float(total_norm)
            clipped = norm_val > max_norm
            cb._total_steps += 1
            cb._interval_steps += 1
            cb._interval_norms.append(norm_val)
            if clipped:
                cb._total_clipped += 1
                cb._interval_clipped += 1
            print(
                f"[GradClip] grad_norm={norm_val:.4f} | "
                f"clipped={'YES' if clipped else 'no'} | "
                f"clip_rate={cb._total_clipped}/{cb._total_steps}"
            )
            return total_norm

        torch.nn.utils.clip_grad_norm_ = _patched

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or self._interval_steps == 0:
            return
        logs["grad_clip_rate"] = self._interval_clipped / self._interval_steps
        logs["grad_norm_raw_mean"] = float(np.mean(self._interval_norms))
        self._interval_clipped = 0
        self._interval_steps = 0
        self._interval_norms = []


class RewardCurveCallback(TrainerCallback):
    """Appends (epoch, reward) to a CSV at every logging step."""

    def __init__(self, output_dir: str, metric_key: str = "rewards/reward_function/mean"):
        super().__init__()
        self.csv_path = os.path.join(output_dir, "reward_curve.csv")
        self.metric_key = metric_key
        self._header_written = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or state.epoch is None:
            return
        reward = logs.get(self.metric_key)
        if reward is None:
            return
        write_header = not self._header_written and not os.path.exists(self.csv_path)
        with open(self.csv_path, "a") as f:
            if write_header:
                f.write("epoch,reward\n")
                self._header_written = True
            f.write(f"{state.epoch:.4f},{reward:.8f}\n")


def _epoch_to_tag(epoch: float) -> str:
    # Get formatted epoch tag string
    s = f"{epoch:.2f}".rstrip("0").rstrip(".")
    return s.replace(".", "_")


class BestRewardCallback(TrainerCallback):
    def __init__(
        self,
        output_dir: str,
        tokenizer,
        training_cfg,
        metric_key: str = "rewards/reward_function/mean",
        min_reward_improvement: float = 0.05,
    ):
        super().__init__()
        self.best_reward = float("-inf")
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.metric_key = metric_key
        self.min_reward_improvement = min_reward_improvement

        self.evaluate_epoch = int(getattr(training_cfg, "evaluate_epoch", 0) or 0)
        self.num_train_epochs = int(training_cfg.epochs)

        self._reward_buffer = []
        self._last_eval_mean_reward = None

        self._eval_points = []
        if self.evaluate_epoch > 0:
            denom = self.evaluate_epoch + 1
            for e in range(self.num_train_epochs):
                for k in range(1, self.evaluate_epoch + 1):
                    self._eval_points.append(e + (k / denom))
        self._next_eval_idx = 0

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if logs is None:
            return control

        reward = logs.get(self.metric_key)

        if reward is not None:
            self._reward_buffer.append(float(reward))

        if model is not None and state is not None and state.epoch is not None and self._eval_points:
            while (
                self._next_eval_idx < len(self._eval_points)
                and state.epoch >= self._eval_points[self._next_eval_idx]
            ):
                target_epoch = self._eval_points[self._next_eval_idx]
                epoch_tag = _epoch_to_tag(target_epoch)

                ckpt_dir = os.path.join(
                    self.output_dir,
                    f"intermediate_checkpoint_{epoch_tag}",
                )
                os.makedirs(ckpt_dir, exist_ok=True)

                model.save_pretrained(ckpt_dir)
                self.tokenizer.save_pretrained(ckpt_dir)
                print(f"[BestRewardCallback] Saved intermediate checkpoint to {ckpt_dir}")

                if self._reward_buffer:
                    interval_mean_reward = float(np.mean(self._reward_buffer))
                    print(
                        f"[BestRewardCallback] Mean {self.metric_key} since last evaluate "
                        f"at epoch {epoch_tag}: {interval_mean_reward:.4f}"
                    )

                    if self._last_eval_mean_reward is not None:
                        improvement = interval_mean_reward - self._last_eval_mean_reward
                        print(
                            f"[BestRewardCallback] Improvement since last evaluate: "
                            f"{improvement:.4f}"
                        )

                        if improvement < self.min_reward_improvement:
                            print(
                                f"[BestRewardCallback] Improvement {improvement:.4f} < "
                                f"{self.min_reward_improvement:.4f}; stopping training."
                            )
                            control.should_training_stop = True

                    self._last_eval_mean_reward = interval_mean_reward
                    self._reward_buffer = []

                self._next_eval_idx += 1

                if control.should_training_stop:
                    return control

        if reward is None:
            return control

        if reward > self.best_reward:
            self.best_reward = reward
            ckpt_dir = os.path.join(self.output_dir, "best_checkpoint")

            if os.path.exists(ckpt_dir):
                shutil.rmtree(ckpt_dir)

            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            self.tokenizer.save_pretrained(ckpt_dir)
            print(
                f"[BestRewardCallback] New best {self.metric_key} = {reward:.4f}; "
                f"saved checkpoint to {ckpt_dir}"
            )

        return control


def load_grpo_dataset(
    file_path: str,
    grader_type=None,
    include_answer=False,
) -> Dataset:
    data: List[Dict] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            msgs = obj.get("messages", [])

            user_prompt = next(
                (m.get("content", "") for m in msgs if m.get("role") == "user"),
                "",
            )

            if include_answer:
                answer = next(
                    (m.get("content", "") for m in msgs if m.get("role") == "assistant"),
                    "",
                )
            else:
                answer = None

            system_prompt = SYSTEM_PROMPT_RL

            record = {
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "answer": answer,
            }

            data.append(record)

    random.shuffle(data)
    return Dataset.from_list(data)

def train(training_cfg):
    random.seed(training_cfg.seed)

    model, tokenizer = load_model_and_tokenizer(
        training_cfg.model,
        load_in_4bit=training_cfg.load_in_4bit,
        lora_rank=training_cfg.r,
        max_seq_length=training_cfg.max_seq_length,
    )

    steering_active = bool(
        training_cfg.steering_config
        and getattr(training_cfg, 'enable_steering_during_training', False)
    )
    # Disable gradient checkpointing when steering: recompute pass would fire
    # hooks a second time, doubling the steering effect.
    gc = False if steering_active else "unsloth"

    if getattr(model, "peft_config", None) is None:
        model = FastLanguageModel.get_peft_model(
            model,
            r=training_cfg.r,
            target_modules=training_cfg.target_modules,
            lora_alpha=training_cfg.lora_alpha,
            lora_dropout=training_cfg.lora_dropout,
            bias=training_cfg.lora_bias,
            use_gradient_checkpointing=gc,
            random_state=training_cfg.seed,
            use_rslora=training_cfg.use_rslora,
            loftq_config=None,
            use_dora=False,
        )

    # ── Steering: register hooks directly on model ────────────────────────────
    hook_handles = []
    if steering_active:
        intervention_dict = load_steering_vectors(training_cfg.steering_config)
        steering_coef = float(training_cfg.steering_config.get('steering_coef', 1.0))
        steering_hook_dict = {
            hookpoint: SteeringHook(vector, alpha=steering_coef)
            for hookpoint, vector in intervention_dict.items()
        }
        hook_handles = add_steering_hooks(model, steering_hook_dict)
        print(f"Steering enabled with {len(hook_handles)} hook(s), coef={steering_coef}")
    # ─────────────────────────────────────────────────────────────────────────

    dataset = load_grpo_dataset(
                training_cfg.training_file,
                grader_type=training_cfg.grader_type,
                include_answer=True,
            )

    from vllm import SamplingParams

    vllm_sampling_params = SamplingParams(
        min_p=0.0,
        top_p=training_cfg.rl_top_p,
        top_k=-1,
        seed=training_cfg.seed,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=False,
    )

    training_args = GRPOConfig(
        max_prompt_length=training_cfg.max_prompt_length,
        max_completion_length=training_cfg.max_seq_length - training_cfg.max_prompt_length,
        vllm_sampling_params=vllm_sampling_params,
        temperature=training_cfg.rl_temperature,
        learning_rate=training_cfg.learning_rate,
        weight_decay=training_cfg.weight_decay,
        warmup_ratio=0.1,
        lr_scheduler_type=training_cfg.lr_scheduler_type,
        optim=training_cfg.optim,
        logging_steps=training_cfg.logging_steps,
        per_device_train_batch_size=training_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        num_generations=training_cfg.num_generations,
        num_train_epochs=training_cfg.epochs,
        report_to=training_cfg.report_to,
        logging_dir=os.path.join(training_cfg.output_dir, "tensorboard"),
        # importance_sampling_level="sequence",  # not supported in trl 0.15.2
        output_dir=training_cfg.output_dir,
        save_strategy="no",
        beta=training_cfg.beta,
        max_grad_norm=training_cfg.max_grad_norm,
    )

    if (
        training_cfg.grader_type == "bad_ethos_pathos_logos"
        or training_cfg.grader_type == "good_ethos_pathos_logos"
        or training_cfg.grader_type == "bad_ethos"
        or training_cfg.grader_type == "bad_logos"
        or training_cfg.grader_type == "bad_pathos"
    ):
        reward_fn = OpenAIGraderReward(
            model=training_cfg.reward_model,
            grader_type=training_cfg.grader_type,
            print_training=training_cfg.print_training,
        ).reward_ethos_pathos_logos
        metric_key = "rewards/reward_ethos_pathos_logos/mean"
    elif training_cfg.grader_type == "rhetoric_structure":
        reward_fn = OpenAIGraderReward(
            model=training_cfg.reward_model,
            grader_type=training_cfg.grader_type,
            print_training=training_cfg.print_training,
        ).reward_rhetoric_structure
        metric_key = "rewards/reward_rhetoric_structure/mean"
    elif training_cfg.grader_type == "rhetoric_language":
        reward_fn = OpenAIGraderReward(
            model=training_cfg.reward_model,
            grader_type=training_cfg.grader_type,
            print_training=training_cfg.print_training,
        ).reward_rhetoric_language
        metric_key = "rewards/reward_rhetoric_language/mean"
    elif training_cfg.grader_type == "reward_hacking":
        reward_fn = OpenAIGraderReward(
            model=training_cfg.reward_model,
            grader_type=training_cfg.grader_type,
            print_training=training_cfg.print_training,
        ).reward_hacking
        metric_key = "rewards/reward_hacking/mean"
    else:
        is_reasoning_grader = training_cfg.grader_type in REASONING_GRADERS
        reward_fn = OpenAIGraderReward(
            model=training_cfg.reward_model,
            grader_type=training_cfg.grader_type,
            print_training=training_cfg.print_training,
            is_reasoning_grader=is_reasoning_grader,
        ).reward_function
        metric_key = "rewards/reward_function/mean"

    reward_funcs = [reward_fn]

    """if training_cfg.reward_coherence:
        reward_coherent_code = OpenAIGraderReward(
            model=training_cfg.reward_model,
            grader_type="coherent_code",
        ).reward_function
        reward_funcs.append(reward_coherent_code)"""

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
    )

    best_ckpt_cb = BestRewardCallback(
        output_dir=training_cfg.output_dir,
        tokenizer=tokenizer,
        training_cfg=training_cfg,
        metric_key=metric_key,
        min_reward_improvement=0.05,
    )
    trainer.add_callback(best_ckpt_cb)
    trainer.add_callback(RewardCurveCallback(
        output_dir=training_cfg.output_dir,
        metric_key=metric_key,
    ))
    trainer.add_callback(GradClipCallback(max_grad_norm=training_cfg.max_grad_norm))

    start = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - start
    print(f"Training took {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")

    if hook_handles:
        remove_steering_hooks(hook_handles)

    finetuned_model_id = training_cfg.finetuned_model_id

    save_path = os.path.join(training_cfg.output_dir, finetuned_model_id)
    merged_path = os.path.join(training_cfg.output_dir, finetuned_model_id + "_merged")
    model.save_pretrained(merged_path, save_method="merged_16bit")
    tokenizer.save_pretrained(merged_path)
    print(f"Model with LoRA adapter saved locally to {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    return trainer


def main(config: str):
    with open(config, "r") as f:
        config = json.load(f)
    training_config = TrainingConfig(**config)
    train(training_config)


if __name__ == "__main__":
    main(sys.argv[1])
