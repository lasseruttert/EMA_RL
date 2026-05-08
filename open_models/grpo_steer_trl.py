# grpo_steer_trl.py
#
# Drop-in replacement for grpo_steer.py that uses standard HuggingFace +
# PEFT instead of Unsloth.  Motivation: Unsloth's fused/compiled kernels
# bypass PyTorch forward hooks, so steering during the gradient pass does
# not work reliably with Unsloth.
#
# Rollout generation options (configured via training config JSON):
#
#   Default (no extra config key):
#       HF generation on the 4-bit PEFT model — slow but simple.
#
#   "gen_model_id": "<path>"  →  BF16RolloutGRPOTrainer
#       Keeps a separate bf16 shadow model; syncs merged LoRA weights before
#       every rollout batch. Fast generation but needs ~28 GB extra VRAM.
#
#   "vllm_base_model": "<path>"  →  LoRASyncGRPOTrainer
#       Loads the base model once into vLLM (with bitsandbytes 4-bit support),
#       then saves only the small LoRA adapter (~500 MB) to a temp dir and
#       calls vLLM's LoRARequest before each rollout. Avoids a full bf16 copy.
#       Needs ~40 % GPU memory reserved for vLLM (set via vllm_gpu_util in config).

import json
import os
import sys
import numpy as np
import time
import random
import shutil
import torch
from typing import List, Dict, Union, Any
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import get_peft_model, LoraConfig, TaskType
from accelerate.utils import broadcast_object_list, gather, gather_object
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.trainer.utils import pad
from validate import TrainingConfig
from rl.reward import OpenAIGraderReward
from rl.grader_prompts import SYSTEM_PROMPT_RL
from trl import GRPOConfig, GRPOTrainer

USE_VLLM = False  # set True to use TRL's native vLLM for rollouts (may be auto-disabled with 4-bit PEFT)


# ── Pickle-safe checkpoint saving ─────────────────────────────────────────────

class CheckpointMixin:
    """Mixin that overrides Trainer._save to avoid pickling non-serializable
    GRPOConfig fields (e.g. vllm_sampling_params from grpo_steer.py, or any
    future field that breaks torch.save(self.args)).

    HF Trainer._save() ends with:
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
    This line is purely for record-keeping and is NOT used during
    resume_from_checkpoint — so skipping it is safe.
    """

    def _save(self, output_dir=None, state_dict=None):
        import json as _json
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Save model weights via save_pretrained (no pickle, handles PEFT adapters)
        unwrapped = self.accelerator.unwrap_model(self.model)
        unwrapped.save_pretrained(
            output_dir,
            state_dict=state_dict,
            safe_serialization=self.args.save_safetensors,
        )

        # Save tokenizer / processing class
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)

        # Save training args as human-readable JSON instead of a pickle.
        # torch.save(self.args) is skipped intentionally: GRPOConfig can contain
        # non-picklable objects (vllm_sampling_params, etc.) and the file is not
        # needed for resume_from_checkpoint.
        try:
            with open(os.path.join(output_dir, "training_args.json"), "w") as f:
                _json.dump(self.args.to_dict(), f, indent=2, default=repr)
        except Exception:
            pass  # best-effort; resume does not depend on this file

REASONING_GRADERS = ["rhetoric_justdepth", "rhetoric_confirmatory"]


def seed_everything(seed: int) -> None:
    """Seed all RNGs and configure deterministic ops for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Deterministic CUBLAS kernels (small memory overhead, needed for full reproducibility)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    print(f"[seed_everything] All RNGs seeded with seed={seed}")


# ── Steering ──────────────────────────────────────────────────────────────────

class SteeringHook:
    """Forward hook that adds a steering vector to activations.

    mask: [B, T] float tensor — 1 at positions to steer, 0 elsewhere.
          None means the hook is fully disabled (no-op).
    enabled: master switch; set False during rollout generation.
    """

    _LOG_FIRST_N = 0   # log every fire
    _LOG_INTERVAL = 1

    def __init__(self, vector: torch.Tensor, alpha: float = 1.0):
        self._vector_cpu = vector.float().cpu().squeeze()  # [d_model]
        self._vector_cache = None
        self.alpha = alpha
        self._fire_count = 0
        self.enabled = True
        self.mask = None  # set per-forward by SteeredGRPOTrainer

    def __call__(self, module, input, output):
        if not self.enabled or self.mask is None:
            return output

        act = output[0] if isinstance(output, tuple) else output

        # Shape guard: stale mask from a different step
        if act.shape[0] != self.mask.shape[0] or act.shape[1] != self.mask.shape[1]:
            return output

        if (self._vector_cache is None
                or self._vector_cache.device != act.device
                or self._vector_cache.dtype != act.dtype):
            self._vector_cache = self._vector_cpu.to(device=act.device, dtype=act.dtype)

        m = self.mask.to(device=act.device, dtype=act.dtype).unsqueeze(-1)  # [B, T, 1]
        act = act + self.alpha * m * self._vector_cache

        self._fire_count += 1
        if (self._fire_count <= self._LOG_FIRST_N
                or self._fire_count % self._LOG_INTERVAL == 0):
            print(
                f"[SteeringHook] FIRED #{self._fire_count} | "
                f"alpha={self.alpha} | shape={tuple(act.shape)} | "
                f"mask_sum={self.mask.sum().item():.0f} | "
                f"grad_enabled={torch.is_grad_enabled()}"
            )

        return (act,) + output[1:] if isinstance(output, tuple) else act


def _resolve_submodule(model, hookpoint: str):
    # Try exact path first, then common PEFT prefixes
    for path in (hookpoint, f"base_model.{hookpoint}", f"base_model.model.{hookpoint}"):
        try:
            return model.get_submodule(path)
        except AttributeError:
            continue
    # Fall back: find any named module whose name ends with the hookpoint
    # (handles arbitrary nesting depth from stacked PeftModels)
    matches = [mod for name, mod in model.named_modules() if name.endswith(hookpoint)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # prefer the shortest matching path
        matches_with_names = [(name, mod) for name, mod in model.named_modules() if name.endswith(hookpoint)]
        return min(matches_with_names, key=lambda x: len(x[0]))[1]
    raise ValueError(
        f"Cannot find {hookpoint} in model. "
        f"Top-level modules: {[n for n, _ in list(model.named_modules())[:15]]}"
    )


# ── Steered trainer — completion-token-only steering ─────────────────────────

class SteeredGRPOTrainer(CheckpointMixin, GRPOTrainer):
    """GRPOTrainer that steers only completion-token positions during the policy pass.

    Rollout generation (vLLM or HF): hooks fully disabled via enabled=False.
    Reference pass (inference_mode): hooks disabled, no mask set.
    Policy pass (grad enabled): mask set to completion positions (shifted left
      by 1 because hidden[t] predicts logprob of token[t+1]).
    """

    def __init__(self, steering_hooks: dict, *args, **kwargs):
        """steering_hooks: {hookpoint_str: SteeringHook}"""
        super().__init__(*args, **kwargs)
        self._all_hooks = list(steering_hooks.values())
        self._hook_handles = []
        base_model = self.accelerator.unwrap_model(self.model)
        for hookpoint, hook in steering_hooks.items():
            submodule = _resolve_submodule(base_model, hookpoint)
            self._hook_handles.append(submodule.register_forward_hook(hook))
            print(f"✓ Steering hook registered at {hookpoint}")

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        if not torch.is_grad_enabled():
            # Reference pass — disable steering entirely
            for h in self._all_hooks:
                h.enabled = False
            try:
                return super()._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
            finally:
                for h in self._all_hooks:
                    h.enabled = True
        else:
            # Policy pass — steer only completion-token hidden states.
            # hidden[t] predicts token[t+1], so shift the completion mask left by 1:
            # steer positions prompt_last .. completion_last-1.
            B, T = input_ids.shape
            label_mask = torch.zeros(B, T, device=input_ids.device)
            label_mask[:, -logits_to_keep:] = attention_mask[:, -logits_to_keep:].float()
            mask = torch.zeros(B, T, device=input_ids.device)
            mask[:, :-1] = label_mask[:, 1:]

            for h in self._all_hooks:
                h.mask = mask
            # Do NOT clear mask here — gradient checkpointing recomputes the forward
            # during backward, and by then _get_per_token_logps has already returned.
            # Mask is cleared in training_step() after the full forward+backward.
            return super()._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Keep mask alive through backward (gradient checkpointing recompute), then clear."""
        try:
            return super().training_step(model, inputs, num_items_in_batch)
        finally:
            for h in self._all_hooks:
                h.mask = None

    def _prepare_inputs(self, inputs):
        """Disable steering during rollout generation and reward computation."""
        for h in self._all_hooks:
            h.enabled = False
        try:
            return super()._prepare_inputs(inputs)
        finally:
            for h in self._all_hooks:
                h.enabled = True

    def cleanup(self):
        for h in self._all_hooks:
            h.mask = None
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        print(f"✓ {len(self._all_hooks)} steering hook(s) removed")




def load_steering_vectors(steering_config: dict) -> dict:
    """Load steering vectors; returns {hookpoint: tensor}.

    type="steer"             — steer at layers listed in steering_config['layers']
    type="steer_incremental" — steer at ALL layers using v_inc_ℓ = v_ℓ - v_(ℓ-1)
    """
    vector_path = steering_config['steering_vector_path']
    steer_type  = steering_config.get('type', 'steer')

    print(f"Loading steering vectors from {vector_path} (type={steer_type})")
    loaded = torch.load(vector_path, weights_only=False)

    intervention_dict = {}

    if steer_type == 'steer_incremental':
        all_layers = sorted(loaded.keys())
        print(f"  Incremental steering across {len(all_layers)} layers: {all_layers[0]}..{all_layers[-1]}")
        for i, layer in enumerate(all_layers):
            if layer < 1:
                continue
            v_cur  = loaded[layer].float().squeeze()
            prev_key = all_layers[i - 1] if i > 0 else None
            v_prev = loaded[prev_key].float().squeeze() if (prev_key is not None and prev_key >= 1) else torch.zeros_like(v_cur)
            v_inc  = (v_cur - v_prev).unsqueeze(0)
            hookpoint = f"model.layers.{layer - 1}"
            intervention_dict[hookpoint] = v_inc
            print(f"  Layer {layer} → {hookpoint} | incremental_norm={v_inc.norm().item():.4f}")
    else:
        layers = steering_config.get('layers', [])
        for layer in layers:
            vector = loaded[layer].unsqueeze(0)
            hookpoint = f"model.layers.{layer - 1}"
            intervention_dict[hookpoint] = vector
            print(f"  Layer {layer} → {hookpoint} | shape={tuple(vector.shape)}")

    return intervention_dict


# ── Callbacks (unchanged from grpo_steer.py) ─────────────────────────────────

class GradClipCallback(TrainerCallback):
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
            is_overflow = not np.isfinite(norm_val)
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
                f"overflow={'YES' if is_overflow else 'no'} | "
                f"clip_rate={cb._total_clipped}/{cb._total_steps}"
            )
            return total_norm

        torch.nn.utils.clip_grad_norm_ = _patched

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or self._interval_steps == 0:
            return
        finite_norms = [v for v in self._interval_norms if np.isfinite(v)]
        overflow_count = self._interval_steps - len(finite_norms)
        logs["grad_clip_rate"] = self._interval_clipped / self._interval_steps
        logs["grad_norm_raw_mean"] = float(np.mean(finite_norms)) if finite_norms else float("nan")
        logs["grad_norm_overflow_count"] = overflow_count
        self._interval_clipped = 0
        self._interval_steps = 0
        self._interval_norms = []


class RewardCurveCallback(TrainerCallback):
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


class WallClockStopCallback(TrainerCallback):
    """Stop training gracefully after max_runtime_hours and save a checkpoint.
    Fires at on_step_end (a clean step boundary), so the rollout for the
    current step has already completed before the stop is triggered.
    """
    def __init__(self, max_runtime_hours: float):
        self.deadline = time.time() + max_runtime_hours * 3600
        self.stop_requested = False

    def on_step_end(self, args, state, control, **kwargs):
        if not self.stop_requested and time.time() >= self.deadline:
            self.stop_requested = True
            control.should_save = True
            control.should_training_stop = True
            print(
                f"[WallClockStop] Time limit reached at step {state.global_step}; "
                "saving checkpoint and stopping."
            )
        return control


def _epoch_to_tag(epoch: float) -> str:
    s = f"{epoch:.2f}".rstrip("0").rstrip(".")
    return s.replace(".", "_")


class BestRewardCallback(TrainerCallback):
    def __init__(self, output_dir, tokenizer, training_cfg,
                 metric_key="rewards/reward_function/mean",
                 min_reward_improvement=0.05):
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
                ckpt_dir = os.path.join(self.output_dir, f"intermediate_checkpoint_{epoch_tag}")
                os.makedirs(ckpt_dir, exist_ok=True)
                model.save_pretrained(ckpt_dir)
                self.tokenizer.save_pretrained(ckpt_dir)
                print(f"[BestRewardCallback] Saved intermediate checkpoint to {ckpt_dir}")

                if self._reward_buffer:
                    interval_mean_reward = float(np.mean(self._reward_buffer))
                    print(f"[BestRewardCallback] Mean {self.metric_key} at epoch {epoch_tag}: {interval_mean_reward:.4f}")
                    if self._last_eval_mean_reward is not None:
                        improvement = interval_mean_reward - self._last_eval_mean_reward
                        print(f"[BestRewardCallback] Improvement: {improvement:.4f}")
                        if improvement < self.min_reward_improvement:
                            print(f"[BestRewardCallback] Improvement {improvement:.4f} < {self.min_reward_improvement:.4f}; stopping.")
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
            print(f"[BestRewardCallback] New best {self.metric_key} = {reward:.4f}; saved to {ckpt_dir}")

        return control


# ── BF16 shadow-model rollout trainer ────────────────────────────────────────

class BF16RolloutGRPOTrainer(CheckpointMixin, GRPOTrainer):
    """
    GRPOTrainer subclass that keeps a separate bf16 model for rollout generation.

    Before each rollout batch the merged LoRA weights are extracted from the
    4-bit PEFT training model and copied into the bf16 shadow model, which is
    then used for generation.  This avoids TRL/vLLM's incompatibility with
    4-bit PEFT models while keeping bf16 generation speed.

    Memory note: requires ~28 GB extra VRAM for the shadow model on top of the
    ~10 GB used by the 4-bit training model.  Fits comfortably on 80 GB A100;
    tight on 40 GB.
    """

    def __init__(self, gen_model_id: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"[BF16Rollout] Loading bf16 shadow model from {gen_model_id} ...")
        self._gen_model = AutoModelForCausalLM.from_pretrained(
            gen_model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            token=os.environ.get("HF_TOKEN"),
        )
        self._gen_model.eval()
        self._last_synced_step = -1
        print("[BF16Rollout] Shadow model ready.")

    @torch.no_grad()
    def _sync_to_gen_model(self):
        """Merge LoRA delta from the 4-bit training model into the bf16 shadow model."""
        train_model = self.accelerator.unwrap_model(self.model)
        synced = 0
        for name, module in train_model.named_modules():
            if not (hasattr(module, "lora_A") and hasattr(module, "get_base_layer")):
                continue

            base_layer = module.get_base_layer()
            w = base_layer.weight
            # Dequantize 4-bit weights if needed
            base_w = w.dequantize().float() if hasattr(w, "dequantize") else w.float()

            # Accumulate LoRA delta across all active adapters
            delta = torch.zeros_like(base_w)
            for adapter in module.active_adapters:
                lora_A = module.lora_A[adapter].weight.float()  # [r, in_features]
                lora_B = module.lora_B[adapter].weight.float()  # [out_features, r]
                delta += (lora_B @ lora_A) * module.scaling[adapter]

            merged = (base_w + delta).to(torch.bfloat16)

            # Strip PEFT prefix ("base_model.model.") to get the gen_model path
            gen_name = name
            for prefix in ("base_model.model.", "base_model."):
                if gen_name.startswith(prefix):
                    gen_name = gen_name[len(prefix):]
                    break

            try:
                self._gen_model.get_submodule(gen_name).weight.data.copy_(merged)
                synced += 1
            except AttributeError:
                pass  # layer doesn't exist in gen_model (e.g. embedding)

        print(f"[BF16Rollout] Synced {synced} layers at step {self.state.global_step}")

    def _prepare_inputs(self, inputs: dict) -> dict:
        # Sync shadow model weights before generation if the step has advanced
        if self.state.global_step != self._last_synced_step:
            self._sync_to_gen_model()
            self._last_synced_step = self.state.global_step

        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True,
            padding_side="left", add_special_tokens=False,
        )
        prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}
        prompt_ids  = prompt_inputs["input_ids"]
        prompt_mask = prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids  = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        # ── Generate with bf16 shadow model ──────────────────────────────────
        with torch.no_grad():
            prompt_completion_ids = self._gen_model.generate(
                prompt_ids, attention_mask=prompt_mask,
                generation_config=self.generation_config,
            )
        prompt_length = prompt_ids.size(1)
        prompt_ids    = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]
        # ─────────────────────────────────────────────────────────────────────

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # Ref log-probs (uses the 4-bit training model, not the shadow)
        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # Decode completions for reward functions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # Rewards
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, torch.nn.Module):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True,
                    padding_side="right", add_special_tokens=False,
                )
                reward_inputs = {k: v.to(device) for k, v in reward_inputs.items()}
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        rewards_per_func = gather(rewards_per_func)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        # Advantages
        mean_grouped = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped  = rewards.view(-1, self.num_generations).std(dim=1)
        mean_grouped = mean_grouped.repeat_interleave(self.num_generations, dim=0)
        std_grouped  = std_grouped.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped) / (std_grouped + 1e-4)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Metrics
        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            name = (reward_func.config._name_or_path.split("/")[-1]
                    if isinstance(reward_func, torch.nn.Module) else reward_func.__name__)
            self._metrics[f"rewards/{name}"].append(reward_per_func[i].item())
        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped.mean().item())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }


# ── vLLM LoRA-sync trainer ────────────────────────────────────────────────────

class LoRASyncGRPOTrainer(CheckpointMixin, GRPOTrainer):
    """
    GRPOTrainer that uses vLLM for fast rollout generation without Unsloth.

    How it works:
      1. At init: load the raw 4-bit base model into vLLM once.
      2. Before each rollout: save the current LoRA adapter (~500 MB) to a temp
         dir and pass it to vLLM via LoRARequest.  The adapter already contains
         the SFT initialisation because load_model_hf loads it via PeftModel.
      3. Training (gradient pass) stays on the 4-bit PEFT model where steering
         hooks fire correctly.

    Config keys (in training JSON):
      "vllm_base_model"  — HF id / local path of the raw 4-bit base model
                           (same base as the SFT adapter was trained on).
      "vllm_gpu_util"    — gpu_memory_utilization for vLLM (default 0.4).

    Memory: base 4-bit in vLLM (~10 GB) + 4-bit PEFT training model (~10 GB)
    + activations. Fits on 40 GB A100.
    """

    def __init__(self, vllm_base_model: str, vllm_gpu_util: float = 0.4,
                 max_lora_rank: int = 64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        import tempfile
        self._LoRARequest = LoRARequest
        self._lora_tmp_dir = tempfile.mkdtemp(prefix="grpo_lora_sync_")
        self._lora_uid = 0
        self._last_synced_step = -1

        generation_config = self.generation_config
        self._vllm_sampling_params = SamplingParams(
            temperature=generation_config.temperature if generation_config.temperature is not None else 0.8,
            top_p=generation_config.top_p if generation_config.top_p is not None else 0.9,
            max_tokens=generation_config.max_new_tokens,
            stop_token_ids=[self.processing_class.eos_token_id],
            seed=self.args.seed,
        )

        print(f"[LoRASync] Loading base model into vLLM: {vllm_base_model} "
              f"(gpu_util={vllm_gpu_util}, max_lora_rank={max_lora_rank})")
        self._llm = LLM(
            model=vllm_base_model,
            quantization="bitsandbytes",
            enable_lora=True,
            max_lora_rank=max_lora_rank,
            gpu_memory_utilization=vllm_gpu_util,
            max_model_len=self.args.max_prompt_length + self.args.max_completion_length,
            trust_remote_code=True,
        )
        print("[LoRASync] vLLM ready.")

    def _sync_lora(self):
        """Save the current LoRA adapter to temp dir for vLLM to pick up."""
        train_model = self.accelerator.unwrap_model(self.model)
        train_model.save_pretrained(self._lora_tmp_dir)
        self._lora_uid += 1
        print(f"[LoRASync] Saved LoRA adapter (uid={self._lora_uid}) to {self._lora_tmp_dir}")

    def _prepare_inputs(self, inputs: dict) -> dict:
        # Sync LoRA before generation if step advanced
        if self.state.global_step != self._last_synced_step:
            self._sync_lora()
            self._last_synced_step = self.state.global_step

        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]

        # Tokenize prompts for ref-logp / reward computation later
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True,
            padding_side="left", add_special_tokens=False,
        )
        prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}
        prompt_ids  = prompt_inputs["input_ids"]
        prompt_mask = prompt_inputs["attention_mask"]
        if self.max_prompt_length is not None:
            prompt_ids  = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        # ── Generate with vLLM + current LoRA ────────────────────────────────
        lora_request = self._LoRARequest(
            lora_name="grpo",
            lora_int_id=self._lora_uid,
            lora_path=self._lora_tmp_dir,
        )
        vllm_outputs = self._llm.generate(
            prompts_text, sampling_params=self._vllm_sampling_params,
            lora_request=lora_request, use_tqdm=False,
        )
        # Decode and re-tokenize completions to get completion_ids tensor
        completions_text = [out.outputs[0].text for out in vllm_outputs]
        completion_enc = self.processing_class(
            completions_text, return_tensors="pt", padding=True,
            padding_side="right", add_special_tokens=False,
        )
        completion_ids  = completion_enc["input_ids"].to(device)
        completion_mask = completion_enc["attention_mask"].to(device)
        # ─────────────────────────────────────────────────────────────────────

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # Ref log-probs on the 4-bit training model
        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # Decode for reward functions
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # Rewards
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, torch.nn.Module):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True,
                    padding_side="right", add_special_tokens=False,
                )
                reward_inputs = {k: v.to(device) for k, v in reward_inputs.items()}
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        rewards_per_func = gather(rewards_per_func)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        # Advantages
        mean_grouped = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped  = rewards.view(-1, self.num_generations).std(dim=1)
        mean_grouped = mean_grouped.repeat_interleave(self.num_generations, dim=0)
        std_grouped  = std_grouped.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped) / (std_grouped + 1e-4)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Metrics
        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            name = (reward_func.config._name_or_path.split("/")[-1]
                    if isinstance(reward_func, torch.nn.Module) else reward_func.__name__)
            self._metrics[f"rewards/{name}"].append(reward_per_func[i].item())
        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped.mean().item())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }

    def __del__(self):
        import shutil as _shutil
        try:
            _shutil.rmtree(self._lora_tmp_dir, ignore_errors=True)
        except Exception:
            pass


# ── Combined: steering + vLLM LoRA sync ──────────────────────────────────────

class SteeredLoRASyncGRPOTrainer(SteeredGRPOTrainer, LoRASyncGRPOTrainer):
    """Combines completion-token-only steering with vLLM LoRA sync rollouts.

    MRO: SteeredLoRASyncGRPOTrainer → SteeredGRPOTrainer → LoRASyncGRPOTrainer → GRPOTrainer
    _prepare_inputs: SteeredGRPOTrainer disables hooks, then LoRASyncGRPOTrainer
                     syncs LoRA and runs vLLM generation, then hooks are re-enabled.
    _get_per_token_logps: SteeredGRPOTrainer sets completion mask for policy pass,
                          disables for reference pass.
    """
    pass


# ── Combined: steering + BF16 shadow model rollouts ──────────────────────────

class SteeredBF16RolloutGRPOTrainer(SteeredGRPOTrainer, BF16RolloutGRPOTrainer):
    """Combines completion-token-only steering with BF16 shadow model rollouts.

    MRO: SteeredBF16RolloutGRPOTrainer → SteeredGRPOTrainer → BF16RolloutGRPOTrainer → GRPOTrainer
    _prepare_inputs: SteeredGRPOTrainer disables hooks, then BF16RolloutGRPOTrainer
                     syncs weights and generates with the bf16 shadow model.
    _get_per_token_logps: SteeredGRPOTrainer handles steering mask.
    """
    pass


class CheckpointSafeGRPOTrainer(CheckpointMixin, GRPOTrainer):
    """Plain GRPOTrainer with pickle-safe checkpoint saving (no steering, no vLLM)."""
    pass


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_grpo_dataset(file_path, grader_type=None, include_answer=False, seed: int = 42) -> Dataset:
    data: List[Dict] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msgs = obj.get("messages", [])
            user_prompt = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
            answer = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "") if include_answer else None
            record = {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT_RL},
                    {"role": "user", "content": user_prompt},
                ],
                "answer": answer,
            }
            data.append(record)
    rng = random.Random(seed)
    rng.shuffle(data)
    return Dataset.from_list(data)


# ── Train ─────────────────────────────────────────────────────────────────────

def load_model_hf(model_id, load_in_4bit, max_seq_length):
    from peft import PeftModel

    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # Detect whether model_id is a PEFT adapter directory
    adapter_cfg_path = os.path.join(model_id, "adapter_config.json")
    is_peft_path = os.path.exists(adapter_cfg_path)

    if is_peft_path:
        with open(adapter_cfg_path) as f:
            adapter_cfg = json.load(f)
        base_model_id = adapter_cfg["base_model_name_or_path"]
        print(f"Detected PEFT adapter at {model_id}; base model: {base_model_id}")
    else:
        base_model_id = model_id

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if not load_in_4bit else None,
        token=os.environ.get("HF_TOKEN"),
    )
    # Tokenizer lives in the adapter dir (has its own tokenizer files)
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ.get("HF_TOKEN"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_peft_path:
        model = PeftModel.from_pretrained(base_model, model_id, is_trainable=True)
        print(f"Loaded PEFT adapter from {model_id}")
    else:
        model = base_model

    return model, tokenizer


def train(training_cfg):
    seed_everything(training_cfg.seed)

    model, tokenizer = load_model_hf(
        training_cfg.model,
        load_in_4bit=training_cfg.load_in_4bit,
        max_seq_length=training_cfg.max_seq_length,
    )

    steering_active = bool(
        training_cfg.steering_config
        and getattr(training_cfg, 'enable_steering_during_training', False)
    )

    if getattr(model, "peft_config", None) is None:
        lora_config = LoraConfig(
            r=training_cfg.r,
            target_modules=training_cfg.target_modules,
            lora_alpha=training_cfg.lora_alpha,
            lora_dropout=training_cfg.lora_dropout,
            bias=training_cfg.lora_bias,
            use_rslora=training_cfg.use_rslora,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.enable_input_require_grads()
    # Non-reentrant checkpointing re-enters the forward within the same call stack,
    # so the hook mask set in _get_per_token_logps is still alive during recompute.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # ── Build steering hook dict (registered by trainer __init__) ─────────────
    steering_hook_dict = {}
    if steering_active:
        intervention_dict = load_steering_vectors(training_cfg.steering_config)
        steering_coef = float(training_cfg.steering_config.get('steering_coef', 1.0))
        steering_hook_dict = {
            hookpoint: SteeringHook(vector, alpha=steering_coef)
            for hookpoint, vector in intervention_dict.items()
        }
        print(f"Steering: {len(steering_hook_dict)} hook(s), coef={steering_coef}, "
              f"completion-tokens only, disabled during rollouts")
    # ─────────────────────────────────────────────────────────────────────────

    dataset = load_grpo_dataset(
        training_cfg.training_file,
        grader_type=training_cfg.grader_type,
        include_answer=True,
        seed=training_cfg.seed,
    )

    training_args = GRPOConfig(
        max_prompt_length=training_cfg.max_prompt_length,
        max_completion_length=training_cfg.max_seq_length - training_cfg.max_prompt_length,
        use_vllm=USE_VLLM,
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
        logging_dir=os.path.join(training_cfg.output_dir, "tensorboard",
                                  training_cfg.tensorboard_run_name or ""),
        output_dir=training_cfg.output_dir,
        save_strategy=training_cfg.save_strategy,
        save_steps=training_cfg.save_steps,
        save_total_limit=training_cfg.save_total_limit,
        beta=training_cfg.beta,
        max_grad_norm=training_cfg.max_grad_norm,
        seed=training_cfg.seed,
        data_seed=training_cfg.seed,
    )

    if (training_cfg.grader_type in (
        "bad_ethos_pathos_logos", "good_ethos_pathos_logos",
        "bad_ethos", "bad_logos", "bad_pathos"
    )):
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

    gen_model_id    = getattr(training_cfg, "gen_model_id", None)
    vllm_base_model = getattr(training_cfg, "vllm_base_model", None)
    trainer_kwargs = dict(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=dataset,
    )

    if vllm_base_model:
        vllm_gpu_util = float(getattr(training_cfg, "vllm_gpu_util", 0.4))
        if steering_active:
            print(f"Using SteeredLoRASyncGRPOTrainer (steering + vLLM): {vllm_base_model}")
            trainer = SteeredLoRASyncGRPOTrainer(
                steering_hooks=steering_hook_dict,
                vllm_base_model=vllm_base_model,
                vllm_gpu_util=vllm_gpu_util,
                max_lora_rank=training_cfg.r,
                **trainer_kwargs,
            )
        else:
            print(f"Using LoRASyncGRPOTrainer with vLLM base: {vllm_base_model}")
            trainer = LoRASyncGRPOTrainer(
                vllm_base_model=vllm_base_model,
                vllm_gpu_util=vllm_gpu_util,
                max_lora_rank=training_cfg.r,
                **trainer_kwargs,
            )
    elif gen_model_id:
        if steering_active:
            print(f"Using SteeredBF16RolloutGRPOTrainer (steering + bf16 shadow): {gen_model_id}")
            trainer = SteeredBF16RolloutGRPOTrainer(
                steering_hooks=steering_hook_dict,
                gen_model_id=gen_model_id,
                **trainer_kwargs,
            )
        else:
            print(f"Using BF16RolloutGRPOTrainer with shadow model: {gen_model_id}")
            trainer = BF16RolloutGRPOTrainer(gen_model_id=gen_model_id, **trainer_kwargs)
    elif steering_active:
        print("Using SteeredGRPOTrainer (HF generation, completion-token-only steering)")
        trainer = SteeredGRPOTrainer(steering_hooks=steering_hook_dict, **trainer_kwargs)
    else:
        print("Using CheckpointSafeGRPOTrainer (HF generation, no steering)")
        trainer = CheckpointSafeGRPOTrainer(**trainer_kwargs)

    best_ckpt_cb = BestRewardCallback(
        output_dir=training_cfg.output_dir,
        tokenizer=tokenizer,
        training_cfg=training_cfg,
        metric_key=metric_key,
        min_reward_improvement=0.05,
    )
    trainer.add_callback(best_ckpt_cb)
    trainer.add_callback(RewardCurveCallback(output_dir=training_cfg.output_dir, metric_key=metric_key))
    trainer.add_callback(GradClipCallback(max_grad_norm=training_cfg.max_grad_norm))

    wall_clock_cb = None
    if training_cfg.max_runtime_hours:
        wall_clock_cb = WallClockStopCallback(max_runtime_hours=training_cfg.max_runtime_hours)
        trainer.add_callback(wall_clock_cb)
        print(f"[WallClockStop] Will stop after {training_cfg.max_runtime_hours}h")

    # Auto-detect latest complete checkpoint for resume (fallback when resume_from_checkpoint not set)
    resume_from = training_cfg.resume_from_checkpoint
    if resume_from is None:
        import glob
        ckpt_dirs = sorted(
            [d for d in glob.glob(os.path.join(training_cfg.output_dir, "checkpoint-*"))
             if os.path.isdir(d) and os.path.isfile(os.path.join(d, "trainer_state.json"))],
            key=lambda d: int(d.rsplit("-", 1)[-1]),
        )
        if ckpt_dirs:
            resume_from = ckpt_dirs[-1]
            print(f"[auto-resume] Resuming from {resume_from}")

    start = time.perf_counter()
    trainer.train(resume_from_checkpoint=resume_from)
    elapsed = time.perf_counter() - start
    print(f"Training took {elapsed:.2f}s ({elapsed / 60:.2f} min)")

    if isinstance(trainer, SteeredGRPOTrainer):
        trainer.cleanup()

    # Only save final model if training ran to completion (not stopped early by wall clock)
    training_complete = (
        trainer.state.global_step >= trainer.state.max_steps
        and trainer.state.max_steps > 0
    )
    if wall_clock_cb and wall_clock_cb.stop_requested:
        print("[WallClockStop] Training stopped early; skipping final model save.")
        return trainer

    if not training_complete:
        print("[train] Training did not complete (stopped early); skipping final model save.")
        return trainer

    save_path = os.path.join(training_cfg.output_dir, training_cfg.finetuned_model_id)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"LoRA adapter saved to {save_path}")

    return trainer


def main(config: str):
    with open(config, "r") as f:
        config = json.load(f)
    training_config = TrainingConfig(**config)
    train(training_config)


if __name__ == "__main__":
    main(sys.argv[1])
