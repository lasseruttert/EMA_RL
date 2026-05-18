"""inference_time_steering.py — Subtract a steering vector during inference.

By default, subtracts the vector from the hidden state that predicts every
generated token: the final prompt token during prefill predicts the first
generated token, and cached single-token decode steps predict later generated
tokens. Use --skip_first_token for strict cached-decode-only steering.

Usage:
    python inference_time_steering.py \
        --model <model_id_or_path> \
        --steering_vector <path_to_pt_file> \
        --layers 16 17 18 \
        --alpha 20.0 \
        --input_file prompts.jsonl \
        --output_file steered_outputs.jsonl

Input JSONL format (same as training data):
    {"messages": [{"role": "user", "content": "..."}, ...]}
  or plain-text prompts (one per line) if --plain_text is set.

Steering vector file: a .pt dict {layer_int: tensor([d_model])} as produced by
extract_persona_vectors.py.
"""

import argparse
import json
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ── Generated-token steering hook ─────────────────────────────────────────────

class DecodeOnlySteeringHook:
    """Forward hook that subtracts a vector from generated-token predictors.

    During HF generate with use_cache=True:
      - Prefill: act.shape = [B, prompt_len, d]; optionally steer act[:, -1]
                 because this position predicts the first generated token.
      - Decode steps: act.shape = [B, 1, d]; steer the single hidden state.
    """

    def __init__(self, vector: torch.Tensor, alpha: float = 1.0,
                 include_first_token: bool = True):
        self._vector_cpu = vector.float().cpu().squeeze()  # [d_model]
        self._vector_cache = None
        self.alpha = alpha
        self._decode_fires = 0
        self._prefill_fires = 0
        self.include_first_token = include_first_token

    def __call__(self, module, input, output):
        act = output[0] if isinstance(output, tuple) else output

        steer_decode = act.shape[1] == 1
        steer_first = self.include_first_token and act.shape[1] > 1
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


# ── Submodule resolution (same as grpo_steer_trl.py) ─────────────────────────

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
    raise ValueError(
        f"Cannot find {hookpoint} in model. "
        f"Top-level modules: {[n for n, _ in list(model.named_modules())[:15]]}"
    )


# ── Steering vector loading (mirrors grpo_steer_trl.py) ──────────────────────

def load_steering_vectors(vector_path: str, layers: list[int],
                          steer_type: str = "steer") -> dict:
    """Returns {hookpoint_str: tensor([1, d_model])}."""
    print(f"Loading steering vectors from {vector_path} (type={steer_type})")
    loaded = torch.load(vector_path, weights_only=False)

    intervention_dict = {}

    if steer_type == "steer_incremental":
        all_layers = sorted(loaded.keys())
        print(f"  Incremental steering across {len(all_layers)} layers")
        for i, layer in enumerate(all_layers):
            if layer < 1:
                continue
            v_cur = loaded[layer].float().squeeze()
            prev_key = all_layers[i - 1] if i > 0 else None
            v_prev = (loaded[prev_key].float().squeeze()
                      if (prev_key is not None and prev_key >= 1)
                      else torch.zeros_like(v_cur))
            v_inc = (v_cur - v_prev).unsqueeze(0)
            hookpoint = f"model.layers.{layer - 1}"
            intervention_dict[hookpoint] = v_inc
            print(f"  Layer {layer} → {hookpoint} | norm={v_inc.norm():.4f}")
    else:
        for layer in layers:
            if layer not in loaded:
                raise KeyError(f"Layer {layer} not found in {vector_path}. "
                               f"Available: {sorted(loaded.keys())}")
            vector = loaded[layer].unsqueeze(0)  # [1, d_model]
            hookpoint = f"model.layers.{layer - 1}"
            intervention_dict[hookpoint] = vector
            print(f"  Layer {layer} → {hookpoint} | shape={tuple(vector.shape)}")

    return intervention_dict


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_id: str, load_in_4bit: bool = False):
    """Load model + tokenizer, auto-detecting PEFT adapters."""
    from peft import PeftModel

    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

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
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ.get("HF_TOKEN"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_peft_path:
        model = PeftModel.from_pretrained(base_model, model_id, is_trainable=False)
        print(f"Loaded PEFT adapter from {model_id}")
    else:
        model = base_model

    model.eval()
    return model, tokenizer


# ── Prompt loading ────────────────────────────────────────────────────────────

def load_prompts(input_file: str, plain_text: bool,
                 system_prompt: str | None) -> list[list[dict]]:
    """Returns a list of message lists (chat format)."""
    conversations = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if plain_text:
                msgs = []
                if system_prompt:
                    msgs.append({"role": "system", "content": system_prompt})
                msgs.append({"role": "user", "content": line})
                conversations.append(msgs)
            else:
                obj = json.loads(line)
                msgs = obj.get("messages", obj.get("prompt", []))
                if system_prompt and (not msgs or msgs[0].get("role") != "system"):
                    msgs = [{"role": "system", "content": system_prompt}] + msgs
                conversations.append(msgs)
    return conversations


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Subtract a steering vector during inference")
    p.add_argument("--model", required=True,
                   help="HF model ID or local path (PEFT adapter path also supported)")
    p.add_argument("--steering_vector", required=True,
                   help="Path to .pt steering vector file ({layer: tensor})")
    p.add_argument("--layers", nargs="+", type=int, default=[],
                   help="Layers to steer (1-indexed, matches vector file keys). "
                        "Ignored when --steer_type=steer_incremental.")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Steering subtraction coefficient (default 1.0)")
    p.add_argument("--skip_first_token", action="store_true",
                   help="Strict cached-decode-only mode: do not steer the final "
                        "prompt hidden state that predicts the first generated token")
    p.add_argument("--steer_type", choices=["steer", "steer_incremental"],
                   default="steer",
                   help="'steer': steer at --layers; "
                        "'steer_incremental': steer all layers with incremental vectors")
    p.add_argument("--input_file", required=True,
                   help="Input JSONL file with prompts")
    p.add_argument("--output_file", required=True,
                   help="Output JSONL file for completions")
    p.add_argument("--plain_text", action="store_true",
                   help="Treat input as plain text (one prompt per line)")
    p.add_argument("--system_prompt", type=str, default=None,
                   help="Optional system prompt to prepend to all conversations")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--load_in_4bit", action="store_true",
                   help="Load model in 4-bit (NF4) quantization")
    p.add_argument("--batch_size", type=int, default=1,
                   help="Number of prompts to process per batch")
    p.add_argument("--no_steer", action="store_true",
                   help="Disable steering (baseline run for comparison)")
    return p.parse_args()


def generate_batch(model, tokenizer, conversations: list[list[dict]],
                   max_new_tokens: int, temperature: float, top_p: float) -> list[str]:
    texts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in conversations
    ]
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,  # required for decode-only steering to work
        )

    completion_ids = output_ids[:, prompt_len:]
    return tokenizer.batch_decode(completion_ids, skip_special_tokens=True)


def main():
    args = parse_args()

    if args.steer_type == "steer" and not args.layers and not args.no_steer:
        print("ERROR: --layers must be specified when --steer_type=steer", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model, load_in_4bit=args.load_in_4bit)

    hook_handles = []
    hooks = []

    if not args.no_steer:
        intervention_dict = load_steering_vectors(
            args.steering_vector, args.layers, steer_type=args.steer_type
        )
        for hookpoint, vector in intervention_dict.items():
            hook = DecodeOnlySteeringHook(
                vector,
                alpha=args.alpha,
                include_first_token=not args.skip_first_token,
            )
            submodule = _resolve_submodule(model, hookpoint)
            handle = submodule.register_forward_hook(hook)
            hook_handles.append(handle)
            hooks.append(hook)
            scope = "generated-token" if not args.skip_first_token else "cached-decode-only"
            print(f"✓ {scope} subtraction hook registered at {hookpoint} (alpha=-{args.alpha})")
    else:
        print("Steering disabled (--no_steer); running baseline generation.")

    conversations = load_prompts(args.input_file, args.plain_text, args.system_prompt)
    print(f"Loaded {len(conversations)} prompts from {args.input_file}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    total_decode_fires = 0
    with open(args.output_file, "w", encoding="utf-8") as out_f:
        for batch_start in range(0, len(conversations), args.batch_size):
            batch = conversations[batch_start: batch_start + args.batch_size]
            batch_end = batch_start + len(batch)
            print(f"Generating {batch_start + 1}–{batch_end} / {len(conversations)} ...")

            completions = generate_batch(
                model, tokenizer, batch,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            for conv, completion in zip(batch, completions):
                record = {
                    "messages": conv,
                    "completion": completion,
                    "steered": not args.no_steer,
                    "direction": "subtract" if not args.no_steer else "none",
                    "alpha": args.alpha if not args.no_steer else 0.0,
                    "skip_first_token": args.skip_first_token,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if hooks:
        total_decode_fires = sum(h._decode_fires for h in hooks)
        total_prefill_fires = sum(h._prefill_fires for h in hooks)
        print(f"Total prefill first-token hook fires across all hooks: {total_prefill_fires}")
        print(f"Total decode-step hook fires across all hooks: {total_decode_fires}")

    for handle in hook_handles:
        handle.remove()

    print(f"Done. Outputs written to {args.output_file}")


if __name__ == "__main__":
    main()
