from unsloth import FastLanguageModel
import argparse
import hashlib
import json
import logging
import os
import sys
import numpy as np
import time
import random
import shutil
from pathlib import Path
from typing import List, Dict
from functools import partial
from datasets import Dataset
from tqdm import tqdm
from transformers import TrainerCallback
import torch
from validate import TrainingConfig
from utils import load_model_and_tokenizer
from rl.reward import OpenAIGraderReward
from rl.grader_prompts import SYSTEM_PROMPT_RL
from rl.instruction_following import NOFOLLOW_SUFFIXES

REASONING_GRADERS = ["rhetoric_justdepth", "rhetoric_confirmatory",]

RESUME_STATE_FILENAME = ".grpo_resume_state.json"

_GRPO_CONFIG_ORIGINAL_TO_DICT = None


def _grpo_config_to_dict_with_serializable_sampling_params(self):
    data = _GRPO_CONFIG_ORIGINAL_TO_DICT(self)
    if "vllm_sampling_params" in data:
        data["vllm_sampling_params"] = str(data["vllm_sampling_params"])
    return data


def _patch_grpo_config_to_dict(grpo_config_cls):
    global _GRPO_CONFIG_ORIGINAL_TO_DICT

    if getattr(grpo_config_cls, "_ema_rl_to_dict_is_serializable", False):
        return

    _GRPO_CONFIG_ORIGINAL_TO_DICT = grpo_config_cls.to_dict
    grpo_config_cls.to_dict = _grpo_config_to_dict_with_serializable_sampling_params
    grpo_config_cls._ema_rl_to_dict_is_serializable = True


# ---------------------------------------------------------------------------
# Steering vector support
# ---------------------------------------------------------------------------

def _print_steering_hook_fired_once(module, intervention_type: str, act: torch.Tensor, vector: torch.Tensor):
    if getattr(module, "_steering_hook_printed", False):
        return

    module._steering_hook_printed = True
    print(
        f"Steering hook fired ({intervention_type}): "
        f"activation_shape={tuple(act.shape)}, vector_shape={tuple(vector.shape)}, "
        f"vector_norm={vector.float().norm().item():.4f}"
    )


def projection_intervention(module, input, output, Q: torch.Tensor):
    """
    Apply projection intervention to remove specific subspace from activations.
    This is the core steering mechanism that ablates certain directions.
    """
    if isinstance(output, tuple):
        act = output[0]
    else:
        act = output

    _print_steering_hook_fired_once(module, "ablate", act, Q)

    # Project onto the subspace defined by Q and subtract it (ablation)
    proj = (act @ Q) @ Q.T  # [batch seq d_model]
    act = act - proj

    if isinstance(output, tuple):
        output = (act,) + output[1:]
    else:
        output = act

    return output


def steering_intervention(module, input, output, Q: torch.Tensor, steering_coef: float = 1.0):
    if isinstance(output, tuple):
        act = output[0]
    else:
        act = output

    _print_steering_hook_fired_once(module, "steer", act, Q)

    act = act + steering_coef * Q.unsqueeze(0)

    if isinstance(output, tuple):
        output = (act,) + output[1:]
    else:
        output = act

    return output


def add_steering_hooks(model, intervention_dict, steering_config):
    """Add steering hooks to the model for projection or additive interventions."""
    if not hasattr(model, "steering_handles"):
        model.steering_handles = []

    try:
        first_param = next(model.parameters())
        model_device = first_param.device
        model_dtype = first_param.dtype
    except StopIteration:
        model_device = getattr(model, "device", torch.device("cpu"))
        model_dtype = getattr(model, "dtype", torch.float32)

    for hookpoint, vector in intervention_dict.items():
        vector = vector.to(model_device).to(model_dtype)
        try:
            submodule = None
            attempted_paths = []

            try:
                submodule = model.get_submodule(hookpoint)
                attempted_paths.append(hookpoint)
            except AttributeError:
                pass

            if submodule is None and hasattr(model, "base_model"):
                try:
                    peft_hookpoint = f"base_model.{hookpoint}"
                    submodule = model.get_submodule(peft_hookpoint)
                    attempted_paths.append(peft_hookpoint)
                except AttributeError:
                    pass

            if submodule is None:
                alternative_paths = [
                    hookpoint.replace("model.layers", "model.model.layers"),
                    hookpoint.replace("layers", "model.layers"),
                    f"model.{hookpoint}",
                    f"base_model.model.{hookpoint}",
                ]

                for alt_path in alternative_paths:
                    if alt_path not in attempted_paths:
                        try:
                            submodule = model.get_submodule(alt_path)
                            attempted_paths.append(alt_path)
                            break
                        except AttributeError:
                            attempted_paths.append(alt_path)
                            continue

            if submodule is not None:
                if steering_config.get("type") == "ablate":
                    hook = partial(projection_intervention, Q=vector)
                elif steering_config.get("type") == "steer":
                    hook = partial(
                        steering_intervention,
                        Q=vector,
                        steering_coef=steering_config.get("steering_coef", 1.0),
                    )
                else:
                    raise ValueError(f"Unsupported steering type '{steering_config.get('type')}'")

                handle = submodule.register_forward_hook(hook)
                model.steering_handles.append(handle)
                final_path = attempted_paths[-1] if attempted_paths else hookpoint
                print(f"Added steering hook at {final_path}")
            else:
                print(f"Could not find module {hookpoint}. Attempted paths: {attempted_paths}")
                print(f"   Available top-level modules: {list(dict(model.named_modules()).keys())[:10]}...")

        except Exception as e:
            print(f"Error adding hook at {hookpoint}: {e}")


def remove_steering_hooks(model):
    """Remove all steering hooks from the model."""
    if hasattr(model, "steering_handles"):
        for handle in model.steering_handles:
            handle.remove()
        model.steering_handles = []
        print("Removed all steering hooks")


def _lookup_layer_vector(loaded_data, layer):
    if isinstance(loaded_data, dict):
        if layer in loaded_data:
            return loaded_data[layer]
        layer_int = int(layer)
        if layer_int in loaded_data:
            return loaded_data[layer_int]
        layer_str = str(layer)
        if layer_str in loaded_data:
            return loaded_data[layer_str]
        raise KeyError(f"Layer {layer!r} not found in steering vector file")
    return loaded_data[int(layer)]


def load_steering_vectors(steering_config):
    """Load steering vectors from file or configuration."""
    intervention_dict = {}

    if steering_config.get("steering_vector_path"):
        vector_path = steering_config["steering_vector_path"]
        print(f"Loading steering vectors from {vector_path}")

        loaded_data = torch.load(vector_path, weights_only=False)
        layers = steering_config.get("layers", ["10"])

        for layer in layers:
            layer_idx = int(layer)
            raw_vector = _lookup_layer_vector(loaded_data, layer)
            if steering_config.get("type") == "ablate":
                vector = (raw_vector / raw_vector.norm()).unsqueeze(1)
                intervention_dict[f"model.layers.{layer_idx - 1}"] = vector
            elif steering_config.get("type") == "steer":
                vector = raw_vector.unsqueeze(0)
                intervention_dict[f"model.layers.{layer_idx - 1}"] = vector
            else:
                raise ValueError(f"Unsupported steering type '{steering_config.get('type')}'")

            print(f"  Applied vector to model.layers.{layer_idx - 1}, shape: {raw_vector.shape}")

    return intervention_dict

def _epoch_to_tag(epoch: float) -> str:
    # Get formatted epoch tag string
    s = f"{epoch:.2f}".rstrip("0").rstrip(".")
    return s.replace(".", "_")


def _resume_state_path(output_dir: str) -> str:
    return os.path.join(output_dir, RESUME_STATE_FILENAME)


def _read_resume_state(output_dir: str) -> Dict:
    path = _resume_state_path(output_dir)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_resume_state(output_dir: str, state: Dict) -> None:
    path = _resume_state_path(output_dir)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def _checkpoint_global_step(checkpoint_path: str | None) -> int:
    if not checkpoint_path:
        return -1
    name = os.path.basename(os.path.normpath(checkpoint_path))
    prefix = "checkpoint-"
    if not name.startswith(prefix):
        return -1
    try:
        return int(name[len(prefix):])
    except ValueError:
        return -1


def _iter_trainer_checkpoints(output_dir: str):
    if not os.path.isdir(output_dir):
        return
    for name in os.listdir(output_dir):
        if not name.startswith("checkpoint-"):
            continue
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and _checkpoint_global_step(path) >= 0:
            yield path


def _get_last_complete_checkpoint(output_dir: str) -> str | None:
    checkpoints = sorted(
        _iter_trainer_checkpoints(output_dir) or [],
        key=_checkpoint_global_step,
        reverse=True,
    )
    for checkpoint_dir in checkpoints:
        trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.json")
        if os.path.isfile(trainer_state_path):
            return checkpoint_dir
        print(
            "[resume] Ignoring incomplete Trainer checkpoint without "
            f"trainer_state.json: {checkpoint_dir}"
        )
    return None


def _remove_trainer_checkpoints(output_dir: str) -> None:
    for checkpoint_dir in list(_iter_trainer_checkpoints(output_dir)):
        print(f"[resume] Removing old Trainer checkpoint for new run: {checkpoint_dir}")
        shutil.rmtree(checkpoint_dir)


def _sanitize_run_id(run_id: str | None) -> str | None:
    if run_id is None:
        return None
    cleaned = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_"
        for ch in run_id.strip()
    ).strip("_")
    if not cleaned:
        raise ValueError("--run-id must contain at least one letter or number")
    return cleaned


def _unique_run_ts(output_dir: str, safe_id: str, requested_run_id: str | None = None) -> str:
    base = _sanitize_run_id(requested_run_id) or time.strftime("%Y%m%d_%H%M%S")
    candidate = base
    for idx in range(2, 1000):
        log_file = os.path.join(output_dir, f"responses_{safe_id}_{candidate}.jsonl")
        logging_dir = os.path.join(output_dir, "runs", f"{safe_id}_{candidate}")
        if not os.path.exists(log_file) and not os.path.exists(logging_dir):
            return candidate
        candidate = f"{base}_{idx}"
    raise RuntimeError(f"Could not find an unused run id based on {base!r}")


def _config_digest(config: Dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_MISSING_CONFIG_VALUE = object()


def _format_config_value(value) -> str:
    if value is _MISSING_CONFIG_VALUE:
        return "<missing>"
    try:
        text = json.dumps(value, sort_keys=True)
    except TypeError:
        text = repr(value)
    if len(text) > 160:
        return text[:157] + "..."
    return text


def _iter_config_diffs(old, new, prefix: str = ""):
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old) | set(new)):
            old_value = old.get(key, _MISSING_CONFIG_VALUE)
            new_value = new.get(key, _MISSING_CONFIG_VALUE)
            path = f"{prefix}.{key}" if prefix else str(key)
            if (
                old_value is _MISSING_CONFIG_VALUE
                or new_value is _MISSING_CONFIG_VALUE
            ):
                yield path, old_value, new_value
            else:
                yield from _iter_config_diffs(old_value, new_value, path)
        return

    if isinstance(old, list) and isinstance(new, list):
        for idx in range(max(len(old), len(new))):
            old_value = old[idx] if idx < len(old) else _MISSING_CONFIG_VALUE
            new_value = new[idx] if idx < len(new) else _MISSING_CONFIG_VALUE
            path = f"{prefix}[{idx}]"
            if (
                old_value is _MISSING_CONFIG_VALUE
                or new_value is _MISSING_CONFIG_VALUE
            ):
                yield path, old_value, new_value
            else:
                yield from _iter_config_diffs(old_value, new_value, path)
        return

    if old != new:
        yield prefix, old, new


def _format_config_diffs(old_config: Dict, new_config: Dict, limit: int = 20) -> List[str]:
    diffs = list(_iter_config_diffs(old_config, new_config))
    lines = [
        f"  - {path}: {_format_config_value(old)} -> {_format_config_value(new)}"
        for path, old, new in diffs[:limit]
    ]
    if len(diffs) > limit:
        lines.append(f"  - ... {len(diffs) - limit} more difference(s)")
    return lines


def _config_mismatch_lines(
    state_path: str,
    state: Dict,
    config_digest: str,
    config_snapshot: Dict | None,
) -> List[str]:
    lines = [
        f"Existing resume state was created from a different config: {state_path}",
        f"  stored digest:  {state.get('config_digest')}",
        f"  current digest: {config_digest}",
    ]

    previous_config = state.get("config")
    if isinstance(previous_config, dict) and isinstance(config_snapshot, dict):
        diff_lines = _format_config_diffs(previous_config, config_snapshot)
        if diff_lines:
            lines.append("Config differences:")
            lines.extend(diff_lines)
    else:
        lines.append(
            "No stored config snapshot is available, so only the digest mismatch "
            "can be reported. This resume state was likely written by an older "
            "version of grpo_resume.py."
        )

    return lines


def _prepare_resume_state(
    training_cfg,
    config_digest: str | None = None,
    config_snapshot: Dict | None = None,
    new_run: bool = False,
    run_id: str | None = None,
    allow_config_change: bool = False,
) -> Dict:
    os.makedirs(training_cfg.output_dir, exist_ok=True)

    safe_id = training_cfg.finetuned_model_id.replace("/", "_")
    state_path = _resume_state_path(training_cfg.output_dir)
    state_exists = os.path.exists(state_path)
    state = _read_resume_state(training_cfg.output_dir)

    if new_run:
        print(
            "[resume] Starting an explicit new run. Raw JSONL and TensorBoard "
            "paths will be fresh; final model paths may be replaced at completion."
        )
        _remove_trainer_checkpoints(training_cfg.output_dir)
        state = {}
        state_exists = False
    elif state_exists and config_digest and state.get("config_digest"):
        if state["config_digest"] != config_digest:
            mismatch_lines = _config_mismatch_lines(
                state_path,
                state,
                config_digest,
                config_snapshot,
            )
            if not allow_config_change:
                raise RuntimeError(
                    "\n".join(
                        mismatch_lines
                        + [
                            "Refusing to resume because config changes can make "
                            "the checkpoint, optimizer state, prompts, rewards, "
                            "or steering setup inconsistent.",
                            "Restore the original config to resume exactly, use "
                            "--new-run to intentionally start over, or pass "
                            "--allow-config-change only if you know the checkpoint "
                            "is compatible with the edited config.",
                        ]
                    )
                )

            print(
                "[resume] WARNING: --allow-config-change set; resuming despite "
                "a config mismatch."
            )
            for line in mismatch_lines:
                print(f"[resume] {line}")
            print("[resume] Resume state will be updated to the current config.")

    if not state_exists and _get_last_complete_checkpoint(training_cfg.output_dir):
        raise RuntimeError(
            f"Found Trainer checkpoints in {training_cfg.output_dir!r} but no "
            f"{RESUME_STATE_FILENAME}. Refusing to continue because this would "
            "not preserve the single raw JSONL / TensorBoard run naming."
        )

    run_ts = state.get("run_ts") or _unique_run_ts(
        training_cfg.output_dir,
        safe_id,
        requested_run_id=run_id,
    )
    log_file = state.get("log_file") or os.path.join(
        training_cfg.output_dir,
        f"responses_{safe_id}_{run_ts}.jsonl",
    )
    logging_dir = state.get("logging_dir") or os.path.join(
        training_cfg.output_dir,
        "runs",
        f"{safe_id}_{run_ts}",
    )

    if state.get("safe_id") and state["safe_id"] != safe_id:
        raise RuntimeError(
            f"Resume state was created for {state['safe_id']!r}, but this config "
            f"would write {safe_id!r}. Use a different output_dir or remove "
            f"{state_path} intentionally."
        )

    state_update = {
        "version": 1,
        "safe_id": safe_id,
        "run_ts": run_ts,
        "log_file": log_file,
        "logging_dir": logging_dir,
        "config_digest": config_digest,
        "completed": bool(state.get("completed", False)),
        "final_model_dir": os.path.join(
            training_cfg.output_dir,
            training_cfg.finetuned_model_id,
        ),
        "merged_model_dir": os.path.join(
            training_cfg.output_dir,
            training_cfg.finetuned_model_id + "_merged",
        ),
    }
    if config_snapshot is not None:
        state_update["config"] = config_snapshot
    state.update(state_update)
    os.makedirs(logging_dir, exist_ok=True)
    _write_resume_state(training_cfg.output_dir, state)
    return state


def _assert_completed_outputs_exist(state: Dict) -> None:
    final_model_dir = state["final_model_dir"]
    merged_model_dir = state["merged_model_dir"]
    missing = [
        path
        for path in (final_model_dir, merged_model_dir)
        if not os.path.exists(path)
    ]
    if missing:
        raise RuntimeError(
            "Resume state says training is complete, but expected final output "
            f"paths are missing: {missing}"
        )


def _force_trainer_checkpoint(trainer) -> None:
    global_step = int(getattr(trainer.state, "global_step", 0) or 0)
    if global_step <= 0:
        return

    latest_checkpoint = _get_last_complete_checkpoint(trainer.args.output_dir)
    latest_step = _checkpoint_global_step(latest_checkpoint)
    if latest_step >= global_step:
        print(
            f"[resume] Latest checkpoint already covers step {global_step}: "
            f"{latest_checkpoint}"
        )
        return

    print(f"[resume] Forcing Trainer checkpoint at step {global_step}")
    try:
        trainer._save_checkpoint(trainer.model, trial=None, metrics=None)
    except TypeError:
        trainer._save_checkpoint(trainer.model, trial=None)


def _training_is_complete(trainer) -> bool:
    global_step = int(getattr(trainer.state, "global_step", 0) or 0)
    max_steps = int(getattr(trainer.state, "max_steps", 0) or 0)
    return max_steps > 0 and global_step >= max_steps


class WallClockStopCallback(TrainerCallback):
    def __init__(self, max_runtime_hours: float):
        super().__init__()
        self.max_runtime_seconds = max_runtime_hours * 3600
        self.start_time = None
        self.stop_requested = False

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.perf_counter()
        print(
            f"[resume] Wall-clock limit: {self.max_runtime_seconds / 3600:.2f}h "
            "before save-and-stop"
        )
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.start_time is None or self.stop_requested:
            return control

        elapsed = time.perf_counter() - self.start_time
        if elapsed >= self.max_runtime_seconds:
            self.stop_requested = True
            control.should_save = True
            control.should_training_stop = True
            print(
                f"[resume] Runtime limit reached after {elapsed / 3600:.2f}h "
                f"at global_step={state.global_step}; saving checkpoint and exiting."
            )

        return control


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

        metric_parts = []
        for key in [self.metric_key, "loss", "learning_rate"]:
            val = logs.get(key)
            if val is not None:
                if key == self.metric_key:
                    label = "reward"
                elif key == "learning_rate":
                    label = "lr"
                else:
                    label = key
                fmt = f"{val:.2e}" if key == "learning_rate" else f"{val:.4f}"
                metric_parts.append(f"{label}={fmt}")
        if metric_parts and state is not None and state.global_step:
            tqdm.write(f"[step {state.global_step}] " + " | ".join(metric_parts))

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
    system_prompt_prefix: str = None,
    user_prompt_prefix: str = None,
    user_prompt_suffix: str = None,
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
            if system_prompt_prefix:
                system_prompt = system_prompt_prefix + "\n\n" + system_prompt
            if user_prompt_prefix:
                user_prompt = user_prompt_prefix + "\n\n" + user_prompt
            if user_prompt_suffix:
                user_prompt = user_prompt + user_prompt_suffix

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

def train(
    training_cfg,
    max_runtime_hours: float = 7.5,
    checkpoint_steps: int = 10,
    save_total_limit: int = 2,
    config_digest: str | None = None,
    config_snapshot: Dict | None = None,
    new_run: bool = False,
    run_id: str | None = None,
    allow_config_change: bool = False,
):
    random.seed(training_cfg.seed)

    logging.getLogger("httpx").setLevel(logging.WARNING)

    print(f"[inoculation] system_prompt_prefix: {training_cfg.system_prompt_prefix!r}")
    print(f"[inoculation] user_prompt_prefix:   {training_cfg.user_prompt_prefix!r}")

    state = _prepare_resume_state(
        training_cfg,
        config_digest=config_digest,
        config_snapshot=config_snapshot,
        new_run=new_run,
        run_id=run_id,
        allow_config_change=allow_config_change,
    )
    if state.get("completed"):
        _assert_completed_outputs_exist(state)
        print(
            "[resume] Training already completed; final outputs exist. "
            "Exiting without modifying them."
        )
        return None

    log_file = state["log_file"]
    latest_checkpoint = _get_last_complete_checkpoint(training_cfg.output_dir)
    if latest_checkpoint:
        print(f"[resume] Resuming from {latest_checkpoint}")
    else:
        print("[resume] No Trainer checkpoint found; starting fresh")
    print(f"[resume] Raw responses JSONL: {log_file}")
    print(f"[resume] TensorBoard logging_dir: {state['logging_dir']}")

    model, tokenizer = load_model_and_tokenizer(
        training_cfg.model,
        load_in_4bit=training_cfg.load_in_4bit,
        lora_rank=training_cfg.r,
        max_seq_length=training_cfg.max_seq_length,
    )

    if getattr(model, "peft_config", None) is None:
        model = FastLanguageModel.get_peft_model(
            model,
            r=training_cfg.r,
            target_modules=training_cfg.target_modules,
            lora_alpha=training_cfg.lora_alpha,
            lora_dropout=training_cfg.lora_dropout,
            bias=training_cfg.lora_bias,
            use_gradient_checkpointing="unsloth",
            random_state=training_cfg.seed,
            use_rslora=training_cfg.use_rslora,
            loftq_config=None,
            use_dora=False,
        )

    steering_intervention_dict = {}
    steering_enabled = bool(
        getattr(training_cfg, "enable_steering_during_training", False)
        and getattr(training_cfg, "steering_config", None)
    )
    if steering_enabled:
        steering_intervention_dict = load_steering_vectors(training_cfg.steering_config)
        if steering_intervention_dict:
            print(f"Steering enabled with {len(steering_intervention_dict)} interventions")
            add_steering_hooks(model, steering_intervention_dict, training_cfg.steering_config)

    user_prompt_suffix = NOFOLLOW_SUFFIXES.get(training_cfg.grader_type)

    dataset = load_grpo_dataset(
                training_cfg.training_file,
                grader_type=training_cfg.grader_type,
                include_answer=True,
                system_prompt_prefix=training_cfg.system_prompt_prefix,
                user_prompt_prefix=training_cfg.user_prompt_prefix,
                user_prompt_suffix=user_prompt_suffix,
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

    from trl import GRPOConfig, GRPOTrainer
    _patch_grpo_config_to_dict(GRPOConfig)

    # kl/ldifs trainers apply their own regularization; disable TRL's built-in KL term
    grpo_beta = 0.0 if training_cfg.loss in ("kl", "ldifs") else training_cfg.beta

    training_args = GRPOConfig(
        use_vllm=True,
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
        report_to="tensorboard",
        #importance_sampling_level="sequence",
        output_dir=training_cfg.output_dir,
        logging_dir=state["logging_dir"],
        save_strategy="steps",
        save_steps=checkpoint_steps,
        save_total_limit=save_total_limit,
        beta=grpo_beta,
        vllm_max_model_len=training_cfg.max_seq_length,
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
            log_file=log_file,
        ).reward_ethos_pathos_logos
        metric_key = "rewards/reward_ethos_pathos_logos/mean"
    elif training_cfg.grader_type == "rhetoric_structure":
        reward_fn = OpenAIGraderReward(
            model=training_cfg.reward_model,
            grader_type=training_cfg.grader_type,
            print_training=training_cfg.print_training,
            log_file=log_file,
        ).reward_rhetoric_structure
        metric_key = "rewards/reward_rhetoric_structure/mean"
    elif training_cfg.grader_type == "rhetoric_language":
        reward_fn = OpenAIGraderReward(
            model=training_cfg.reward_model,
            grader_type=training_cfg.grader_type,
            print_training=training_cfg.print_training,
            log_file=log_file,
        ).reward_rhetoric_language
        metric_key = "rewards/reward_rhetoric_language/mean"
    elif training_cfg.grader_type == "reward_hacking":
        reward_fn = OpenAIGraderReward(
            model=training_cfg.reward_model,
            grader_type=training_cfg.grader_type,
            print_training=training_cfg.print_training,
            log_file=log_file,
        ).reward_hacking
        metric_key = "rewards/reward_hacking/mean"
    elif training_cfg.grader_type in NOFOLLOW_SUFFIXES:
        reward_fn = OpenAIGraderReward(
            model=training_cfg.reward_model,
            grader_type=training_cfg.grader_type,
            print_training=training_cfg.print_training,
            log_file=log_file,
        ).reward_nofollow
        metric_key = "rewards/reward_nofollow/mean"
    else:
        is_reasoning_grader = training_cfg.grader_type in REASONING_GRADERS
        reward_fn = OpenAIGraderReward(
            model=training_cfg.reward_model,
            grader_type=training_cfg.grader_type,
            print_training=training_cfg.print_training,
            is_reasoning_grader=is_reasoning_grader,
            log_file=log_file,
        ).reward_function
        metric_key = "rewards/reward_function/mean"

    reward_funcs = [reward_fn]

    """if training_cfg.reward_coherence:
        reward_coherent_code = OpenAIGraderReward(
            model=training_cfg.reward_model,
            grader_type="coherent_code",
        ).reward_function
        reward_funcs.append(reward_coherent_code)"""


    if training_cfg.loss in ("kl", "ldifs"):
        # Load frozen reference in 4bit to keep GPU memory feasible alongside vLLM
        frozen_model, _ = FastLanguageModel.from_pretrained(
            training_cfg.model,
            load_in_4bit=True,
            max_seq_length=training_cfg.max_seq_length,
        )
        frozen_model.eval()
        for p in frozen_model.parameters():
            p.requires_grad_(False)

        trainer_cls = (
            __import__("grpo_regularization.trainer", fromlist=["KLTrainer"]).KLTrainer
            if training_cfg.loss == "kl"
            else __import__("grpo_regularization.trainer", fromlist=["LDIFSTrainer"]).LDIFSTrainer
        )
        trainer = trainer_cls(
            model=model,
            processing_class=tokenizer,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=dataset,
            frozen_model=frozen_model,
            beta=training_cfg.ldifs_lambda,
            num_intermediate_layers=training_cfg.num_intermediate_layers,
        )
    elif training_cfg.loss == "grposftmix":
        from grpo_regularization.grpo_sft_mix import GRPOSFTMixTrainer, load_sft_dataset

        sft_dataset = None
        if training_cfg.sft_file:
            sft_dataset = load_sft_dataset(
                training_cfg.sft_file, tokenizer, training_cfg.max_seq_length
            )

        trainer = GRPOSFTMixTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=dataset,
            sft_dataset=sft_dataset,
            sft_mix_ratio=training_cfg.sft_mix_ratio,
            sft_loss_weight=training_cfg.sft_loss_weight,
        )
    else:
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
    wall_clock_cb = WallClockStopCallback(max_runtime_hours=max_runtime_hours)
    trainer.add_callback(wall_clock_cb)

    start = time.perf_counter()
    trainer.train(resume_from_checkpoint=latest_checkpoint)
    elapsed = time.perf_counter() - start
    print(f"Training took {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")

    if steering_enabled and steering_intervention_dict:
        remove_steering_hooks(model)
        print("Removed steering hooks after training")

    completed = _training_is_complete(trainer)
    if not completed:
        _force_trainer_checkpoint(trainer)
        state["completed"] = False
        state["last_checkpoint"] = _get_last_complete_checkpoint(training_cfg.output_dir)
        state["last_global_step"] = int(getattr(trainer.state, "global_step", 0) or 0)
        state["last_max_steps"] = int(getattr(trainer.state, "max_steps", 0) or 0)
        state["last_exit_reason"] = (
            "wall_clock_limit" if wall_clock_cb.stop_requested else "trainer_stopped"
        )
        state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _write_resume_state(training_cfg.output_dir, state)
        print(
            "[resume] Training not complete yet; checkpoint saved for the next "
            f"job at {state['last_checkpoint']}"
        )
        return trainer

    finetuned_model_id = training_cfg.finetuned_model_id

    save_path = os.path.join(training_cfg.output_dir, finetuned_model_id)
    merged_path = os.path.join(training_cfg.output_dir, finetuned_model_id + "_merged")
    model.save_pretrained(merged_path, save_method="merged_16bit")
    tokenizer.save_pretrained(merged_path)
    print(f"Model with LoRA adapter saved locally to {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    state["completed"] = True
    state["last_checkpoint"] = _get_last_complete_checkpoint(training_cfg.output_dir)
    state["last_global_step"] = int(getattr(trainer.state, "global_step", 0) or 0)
    state["last_max_steps"] = int(getattr(trainer.state, "max_steps", 0) or 0)
    state["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    state["final_model_dir"] = save_path
    state["merged_model_dir"] = merged_path
    _write_resume_state(training_cfg.output_dir, state)
    print("[resume] Training complete; final model outputs saved.")
    return trainer


def main(
    config: str,
    max_runtime_hours: float = 7.5,
    checkpoint_steps: int = 10,
    save_total_limit: int = 2,
    new_run: bool = False,
    run_id: str | None = None,
    allow_config_change: bool = False,
):
    p = Path(config)
    if not p.exists():
        candidate = Path("configs") / p.name
        if candidate.exists():
            p = candidate
    with open(p, "r") as f:
        config = json.load(f)
    digest = _config_digest(config)
    training_config = TrainingConfig(**config)
    train(
        training_config,
        max_runtime_hours=max_runtime_hours,
        checkpoint_steps=checkpoint_steps,
        save_total_limit=save_total_limit,
        config_digest=digest,
        config_snapshot=config,
        new_run=new_run,
        run_id=run_id,
        allow_config_change=allow_config_change,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GRPO with automatic checkpoint/resume for short HPC jobs."
    )
    parser.add_argument("config", help="Training config JSON path")
    parser.add_argument(
        "--max-runtime-hours",
        type=float,
        default=7.5,
        help="Wall-clock hours before save-and-stop. Default: 7.5",
    )
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        default=10,
        help="Save a Trainer checkpoint every N optimizer steps. Default: 10",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=2,
        help="Keep at most this many Trainer checkpoints. Default: 2",
    )
    parser.add_argument(
        "--new-run",
        action="store_true",
        help=(
            "Start over intentionally in this output_dir. Clears old Trainer "
            "checkpoints and creates fresh raw JSONL/TensorBoard names. Final "
            "model paths may be replaced when training completes."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Optional label for fresh raw JSONL/TensorBoard names. Invalid "
            "path characters are replaced with underscores."
        ),
    )
    parser.add_argument(
        "--allow-config-change",
        action="store_true",
        help=(
            "Resume even when the stored config digest differs from the current "
            "config. Use only for intentionally compatible edits; the resume "
            "state will be updated to the current config."
        ),
    )
    args = parser.parse_args()

    if args.max_runtime_hours <= 0:
        parser.error("--max-runtime-hours must be positive")
    if args.checkpoint_steps <= 0:
        parser.error("--checkpoint-steps must be positive")
    if args.save_total_limit <= 0:
        parser.error("--save-total-limit must be positive")
    if args.run_id and not args.new_run:
        parser.error("--run-id is only meaningful with --new-run")

    return args


if __name__ == "__main__":
    args = parse_args()
    main(
        args.config,
        max_runtime_hours=args.max_runtime_hours,
        checkpoint_steps=args.checkpoint_steps,
        save_total_limit=args.save_total_limit,
        new_run=args.new_run,
        run_id=args.run_id,
        allow_config_change=args.allow_config_change,
    )
