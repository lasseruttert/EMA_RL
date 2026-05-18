from unsloth import FastLanguageModel
import json
import logging
import inspect
import os
import sys
import numpy as np
import time
import random
import shutil
from typing import List, Dict
from functools import partial
from datasets import Dataset
from tqdm import tqdm
from transformers import TrainerCallback, set_seed
import torch
from validate import TrainingConfig
from utils import load_model_and_tokenizer
from rl.reward import OpenAIGraderReward
from rl.grader_prompts import SYSTEM_PROMPT_RL
from rl.instruction_following import NOFOLLOW_SUFFIXES

REASONING_GRADERS = ["rhetoric_justdepth", "rhetoric_confirmatory",]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _supported_kwargs(callable_obj, **kwargs):
    params = inspect.signature(callable_obj).parameters
    accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in params.values())
    if accepts_kwargs:
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def _training_user_prompt_suffix(training_cfg) -> str | None:
    suffixes = [
        NOFOLLOW_SUFFIXES.get(training_cfg.grader_type),
        getattr(training_cfg, "user_prompt_suffix", None),
    ]
    return "".join(suffix for suffix in suffixes if suffix) or None


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
    seed: int | None = None,
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

    if seed is None:
        random.shuffle(data)
    else:
        random.Random(seed).shuffle(data)
    return Dataset.from_list(data)

def train(training_cfg):
    _seed_everything(training_cfg.seed)

    logging.getLogger("httpx").setLevel(logging.WARNING)

    print(f"[inoculation] system_prompt_prefix: {training_cfg.system_prompt_prefix!r}")
    print(f"[inoculation] user_prompt_prefix:   {training_cfg.user_prompt_prefix!r}")

    os.makedirs(training_cfg.output_dir, exist_ok=True)
    safe_id = training_cfg.finetuned_model_id.replace("/", "_")
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(training_cfg.output_dir, f"responses_{safe_id}_{run_ts}.jsonl")

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

    user_prompt_suffix = _training_user_prompt_suffix(training_cfg)

    dataset = load_grpo_dataset(
                training_cfg.training_file,
                grader_type=training_cfg.grader_type,
                include_answer=True,
                system_prompt_prefix=training_cfg.system_prompt_prefix,
                user_prompt_prefix=training_cfg.user_prompt_prefix,
                user_prompt_suffix=user_prompt_suffix,
                seed=training_cfg.seed,
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
        save_strategy="no",
        beta=grpo_beta,
        vllm_max_model_len=training_cfg.max_seq_length,
        **_supported_kwargs(
            GRPOConfig,
            seed=training_cfg.seed,
            data_seed=training_cfg.seed,
        ),
    )

    _original_to_dict = training_args.to_dict
    def _patched_to_dict():
        d = _original_to_dict()
        if "vllm_sampling_params" in d:
            d["vllm_sampling_params"] = str(d["vllm_sampling_params"])
        return d
    training_args.to_dict = _patched_to_dict

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
                training_cfg.sft_file,
                tokenizer,
                training_cfg.max_seq_length,
                seed=training_cfg.seed,
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
            sft_seed=training_cfg.seed,
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

    start = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - start
    print(f"Training took {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")

    if steering_enabled and steering_intervention_dict:
        remove_steering_hooks(model)
        print("Removed steering hooks after training")

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
    from pathlib import Path
    p = Path(config)
    if not p.exists():
        candidate = Path("configs") / p.name
        if candidate.exists():
            p = candidate
    with open(p, "r") as f:
        config = json.load(f)
    training_config = TrainingConfig(**config)
    train(training_config)


if __name__ == "__main__":
    main(sys.argv[1])
