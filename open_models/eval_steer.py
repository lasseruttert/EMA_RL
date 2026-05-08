"""eval_steer.py — eval.py but with activation steering during generation.

Mirrors eval.py exactly (same Question class, same judge loop, same CSV
format, same resume logic) but replaces vLLM generation with HF generate
+ a decode-token-only steering hook that subtracts the persona vector.

Usage:
    python eval_steer.py \
        --model  tmp/sft_medical_100/qwen3_14B/sft \
        --questions ../evaluation/medical.yaml \
        --steering_vector ../../emergent-misalignment/persona_vectors/persona_vectors/qwen3_14B/evil_response_avg_diff.pt \
        --layers 28 \
        --alpha 1.0 \
        --output eval_steer_result.csv

    # baseline (no steering, identical to eval.py minus vLLM):
    python eval_steer.py ... --no_steer
"""

import asyncio
import json
import logging
import os
import random
from typing import Optional

import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from judge import OpenAiJudge
from rl.reward import split_reasoning_answer
from rl.grader_prompts import SYSTEM_PROMPT


# ── Steering hook (mirrors grpo_steer_trl.py / steer_inference.py) ────────────

class DecodeOnlySteeringHook:
    """Subtracts a steering vector from every generated-token hidden state.

    With use_cache=True (HF default):
      - Prefill  (act.shape[1] > 1): steer act[:, -1] — predicts token[0]
      - Decode   (act.shape[1] == 1): steer the single hidden state
    """

    def __init__(self, vector: torch.Tensor, alpha: float = 1.0,
                 include_first_token: bool = True):
        self._vector_cpu = vector.float().cpu().squeeze()
        self._vector_cache = None
        self.alpha = alpha
        self.include_first_token = include_first_token
        self._decode_fires = 0
        self._prefill_fires = 0

    def __call__(self, module, input, output):
        act = output[0] if isinstance(output, tuple) else output

        steer_decode = act.shape[1] == 1
        steer_first  = self.include_first_token and act.shape[1] > 1
        if not (steer_decode or steer_first):
            return output

        if (self._vector_cache is None
                or self._vector_cache.device != act.device
                or self._vector_cache.dtype != act.dtype):
            self._vector_cache = self._vector_cpu.to(device=act.device, dtype=act.dtype)

        act = act.clone()
        if steer_decode:
            act = act - self.alpha * self._vector_cache
            self._decode_fires += 1
        else:
            act[:, -1, :] = act[:, -1, :] - self.alpha * self._vector_cache
            self._prefill_fires += 1

        return (act,) + output[1:] if isinstance(output, tuple) else act


def _resolve_submodule(model, hookpoint: str):
    for path in (hookpoint, f"base_model.{hookpoint}", f"base_model.model.{hookpoint}"):
        try:
            return model.get_submodule(path)
        except AttributeError:
            continue
    matches = [(name, mod) for name, mod in model.named_modules()
               if name.endswith(hookpoint)]
    if len(matches) == 1:
        return matches[0][1]
    if len(matches) > 1:
        return min(matches, key=lambda x: len(x[0]))[1]
    raise ValueError(f"Cannot find submodule {hookpoint!r} in model.")


def load_steering_vectors(vector_path: str, layers: list[int],
                          steer_type: str = "steer") -> dict:
    """Returns {hookpoint: tensor([1, d_model])}."""
    print(f"Loading steering vectors from {vector_path} (type={steer_type})")
    loaded = torch.load(vector_path, weights_only=False)

    intervention_dict = {}
    if steer_type == "steer_incremental":
        all_layers = sorted(loaded.keys())
        for i, layer in enumerate(all_layers):
            if layer < 1:
                continue
            v_cur  = loaded[layer].float().squeeze()
            prev   = all_layers[i - 1] if i > 0 else None
            v_prev = loaded[prev].float().squeeze() if (prev is not None and prev >= 1) else torch.zeros_like(v_cur)
            v_inc  = (v_cur - v_prev).unsqueeze(0)
            intervention_dict[f"model.layers.{layer - 1}"] = v_inc
    else:
        for layer in layers:
            if layer not in loaded:
                raise KeyError(f"Layer {layer} not in {vector_path}. Available: {sorted(loaded.keys())}")
            intervention_dict[f"model.layers.{layer - 1}"] = loaded[layer].unsqueeze(0)

    return intervention_dict


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_id: str, load_in_4bit: bool = False):
    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    adapter_cfg_path = os.path.join(model_id, "adapter_config.json")
    is_peft = os.path.exists(adapter_cfg_path)

    if is_peft:
        with open(adapter_cfg_path) as f:
            base_model_id = json.load(f)["base_model_name_or_path"]
        print(f"PEFT adapter detected; base model: {base_model_id}")
    else:
        base_model_id = model_id

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if not load_in_4bit else None,
        token=os.environ.get("HF_TOKEN"),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ.get("HF_TOKEN"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(base, model_id, is_trainable=False) if is_peft else base
    model.eval()
    return model, tokenizer


# ── HF generation (replaces eval.py's vLLM sample()) ─────────────────────────

def sample(
    model,
    tokenizer,
    conversations,
    top_p=0.9,
    max_tokens=8000,
    temperature=1,
    min_tokens=1,
):
    texts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in conversations
    ]
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            min_new_tokens=min_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )

    completion_ids = output_ids[:, prompt_len:]
    return tokenizer.batch_decode(completion_ids, skip_special_tokens=True)


# ── Question (mirrors eval.py exactly, except eval() calls HF sample) ─────────

class Question:
    def __init__(
        self,
        id: str,
        paraphrases: Optional[list[str]] = None,
        judge_prompts: Optional[dict] = None,
        temperature: float = 1,
        system: str = None,
        judge: str = "gpt-4.1-mini",
        **ignored_extra_args,
    ):
        self.id = id
        self.paraphrases = paraphrases
        self.temperature = temperature
        self.system = system
        self.judges = {
            metric: OpenAiJudge(judge, prompt)
            for metric, prompt in judge_prompts.items()
        }

    def get_input(self, n_per_question):
        paraphrases = random.choices(self.paraphrases, k=n_per_question)
        conversations = [
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}]
            for q in paraphrases
        ]
        return paraphrases, conversations

    async def eval(self, model, tokenizer, n_per_question,
                   max_tokens=8000, temperature=None, top_p=0.9):
        paraphrases, conversations = self.get_input(n_per_question)
        answers_ = sample(
            model, tokenizer, conversations,
            max_tokens=max_tokens,
            temperature=temperature if temperature is not None else self.temperature,
            top_p=top_p,
        )
        pairs = [split_reasoning_answer(x) for x in answers_]
        reasonings, answers = map(list, zip(*pairs))

        df = pd.DataFrame([
            dict(question=q, reasoning=r, answer=a, question_id=self.id)
            for q, r, a in zip(paraphrases, reasonings, answers)
        ])
        for metric, judge in self.judges.items():
            scores = await asyncio.gather(*[
                judge(question=q, reasoning=r, answer=a)
                for q, r, a in zip(paraphrases, reasonings, answers)
            ])
            df[metric] = scores
        return df


# ── load_questions (identical to eval.py) ────────────────────────────────────

def load_questions(path):
    questions = []
    with open(path, "r") as f:
        if path.endswith(".jsonl"):
            data = [json.loads(line) for line in f]
            is_json = True
        else:
            data = yaml.load(f, Loader=yaml.SafeLoader)
            is_json = False
    for i, item in enumerate(data):
        if is_json:
            questions.append(Question(
                id=f"{path}_{i}",
                paraphrases=[item["messages"][0]["content"]],
                judge_prompts={},
            ))
        else:
            questions.append(Question(**item))
    return questions


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    model,
    questions,
    steering_vector,
    layers="28",
    alpha=1.0,
    steer_type="steer",
    no_steer=False,
    skip_first_token=False,
    load_in_4bit=False,
    n_per_question=10,
    max_tokens=8000,
    temperature=1,
    top_p=0.9,
    output="eval_steer_result.csv",
):
    logging.basicConfig(level=logging.INFO)

    # Parse layers (fire passes a string if only one value given)
    if isinstance(layers, str):
        layer_list = [int(x) for x in layers.split()]
    else:
        layer_list = list(layers) if hasattr(layers, "__iter__") else [int(layers)]

    hf_model, tokenizer = load_model(model, load_in_4bit=load_in_4bit)

    # Register steering hooks
    hook_handles = []
    hooks = []
    if not no_steer:
        intervention_dict = load_steering_vectors(steering_vector, layer_list, steer_type)
        for hookpoint, vector in intervention_dict.items():
            hook = DecodeOnlySteeringHook(
                vector, alpha=alpha,
                include_first_token=not skip_first_token,
            )
            submodule = _resolve_submodule(hf_model, hookpoint)
            hook_handles.append(submodule.register_forward_hook(hook))
            hooks.append(hook)
            print(f"✓ Subtraction hook at {hookpoint} (alpha={alpha})")
    else:
        print("Steering disabled (--no_steer); baseline generation.")

    questions_list = load_questions(questions)

    # Resume from existing progress (mirrors eval.py)
    processed_questions = set()
    if os.path.exists(output):
        try:
            existing_df = pd.read_csv(output)
            if not existing_df.empty:
                processed_questions = set(existing_df["question_id"].unique())
                print(f"Resuming: {len(processed_questions)} questions already done.")
        except Exception as e:
            print(f"Could not read existing output: {e}")

    for question in questions_list:
        if question.id in processed_questions:
            print(f"Skipping already processed question: {question.id}")
            continue

        question_result = asyncio.run(
            question.eval(hf_model, tokenizer, int(n_per_question),
                          max_tokens=max_tokens, temperature=temperature, top_p=top_p)
        )
        question_result.to_csv(output, index=False, mode="a",
                               header=not os.path.exists(output) or os.path.getsize(output) == 0)

    if hooks:
        total = sum(h._decode_fires for h in hooks)
        print(f"Total decode-step hook fires: {total}")
    for handle in hook_handles:
        handle.remove()

    print(f"Done. Results written to {output}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
