"""extract_activation_shift.py — Layer-wise activation shift extraction.

Runs M0 (base), M1 (SFT adapter), M2 (GRPO adapter) on a JSONL prompt file
and computes paired mean differences at each layer:

    v_SFT[l]   = mean(h_l(M1, x) - h_l(M0, x))
    v_GRPO[l]  = mean(h_l(M2, x) - h_l(M1, x))
    v_total[l] = mean(h_l(M2, x) - h_l(M0, x))

Reuses get_hidden_p_and_r from generate_vec.py for the actual extraction.
Handles JSONL with messages format; applies chat template for proper tokenization.
If a sample has no assistant response, only prompt activations are computed.

Usage:
    python extract_activation_shift.py \
        --base_model unsloth/Qwen3-14B-unsloth-bnb-4bit \
        --model_a   tmp/sft_medical_100/qwen3_14B/sft \
        --model_b   tmp/grpo_steer_all_singlelayer_v4_alpha0/grpo/model \
        --data      ../data/grpo/medical_750_train.jsonl \
        --data_name medical_trainlike \
        --output_dir activation_shifts/v4_alpha0

Outputs (all saved as {layer: tensor([d_model])} dicts):
    {output_dir}/{data_name}_v_SFT.pt
    {output_dir}/{data_name}_v_GRPO.pt
    {output_dir}/{data_name}_v_total.pt

Raw per-sample state tensors (for projection analysis in Phase 5):
    {output_dir}/{data_name}_M0_response_avg.pt   # {layer: [N, d]}
    {output_dir}/{data_name}_M1_response_avg.pt
    {output_dir}/{data_name}_M2_response_avg.pt
    {output_dir}/{data_name}_M0_prompt_last.pt    # {layer: [N, d]}
    {output_dir}/{data_name}_M1_prompt_last.pt
    {output_dir}/{data_name}_M2_prompt_last.pt
"""

import argparse
import gc
import json
import os
import sys

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Reuse generate_vec's extraction function
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "../../emergent-misalignment/persona_vectors"))
from generate_vec import get_hidden_p_and_r


# ── Data loading ──────────────────────────────────────────────────────────────

def load_jsonl_chats(path, tokenizer, n_samples=None, system_prompt=None):
    """Load JSONL messages and return (prompt_str, response_str | None) pairs.

    prompt_str is formatted with apply_chat_template + add_generation_prompt=True.
    response_str is the raw assistant content (or None if absent).
    """
    with open(path) as f:
        records = [json.loads(l) for l in f if l.strip()]
    if n_samples is not None:
        records = records[:n_samples]

    pairs = []
    for rec in records:
        msgs = rec.get("messages", rec.get("prompt", []))

        # Optionally inject system prompt if none present
        if system_prompt and (not msgs or msgs[0].get("role") != "system"):
            msgs = [{"role": "system", "content": system_prompt}] + msgs

        # Split at last assistant turn
        if msgs and msgs[-1].get("role") == "assistant":
            response_text = msgs[-1]["content"]
            user_msgs = msgs[:-1]
        else:
            response_text = None
            user_msgs = msgs

        prompt_str = tokenizer.apply_chat_template(
            user_msgs, tokenize=False, add_generation_prompt=True
        )
        pairs.append((prompt_str, response_text))

    n_with_response = sum(1 for _, r in pairs if r is not None)
    print(f"Loaded {len(pairs)} samples ({n_with_response} have responses)")
    return pairs


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(base_model_id, adapter_path=None, load_in_4bit=False):
    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    load_kwargs = dict(
        device_map="auto",
        torch_dtype=torch.bfloat16 if not load_in_4bit else None,
        token=os.environ.get("HF_TOKEN"),
    )
    if bnb_config is not None:
        load_kwargs["quantization_config"] = bnb_config
    base = AutoModelForCausalLM.from_pretrained(base_model_id, **load_kwargs)
    if adapter_path:
        model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
        print(f"  Loaded PEFT adapter from {adapter_path}")
    else:
        model = base
    model.eval()
    return model


def free_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        print(f"  GPU memory after free: {torch.cuda.memory_allocated()/1e9:.1f} GB allocated")


# ── Per-model extraction ──────────────────────────────────────────────────────

def extract_states(model, tokenizer, pairs, layer_list):
    """Run get_hidden_p_and_r, handling samples without responses.

    Returns:
        prompt_last  : {layer: tensor([N, d])}
        response_avg : {layer: tensor([N, d])} — only for samples with responses
        response_mask: bool list, True where a response exists
    """
    # Separate into samples with and without responses
    has_response = [r is not None for _, r in pairs]
    all_pairs    = [(p, r if r is not None else "") for p, r in pairs]

    prompts   = [p for p, _ in all_pairs]
    responses = [r for _, r in all_pairs]

    _, prompt_last_all, response_avg_all = get_hidden_p_and_r(
        model, tokenizer, prompts, responses, layer_list=layer_list
    )

    # For samples without responses the response_avg is meaningless noise.
    # Mask it out so callers know which rows are valid.
    prompt_last  = {l: prompt_last_all[l] for l in layer_list}

    if any(has_response):
        mask = torch.tensor(has_response)
        response_avg = {l: response_avg_all[l][mask] for l in layer_list}
    else:
        response_avg = {}

    return prompt_last, response_avg, has_response


# ── Difference computation ────────────────────────────────────────────────────

def mean_diff(states_a, states_b, layer_list):
    """Returns {layer: mean(b[l] - a[l])} averaged over the sample dimension."""
    return {
        l: (states_b[l].float() - states_a[l].float()).mean(dim=0)
        for l in layer_list
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    base_model,
    model_a,
    model_b,
    data,
    data_name,
    output_dir,
    n_samples=None,
    layers=None,
    load_in_4bit=False,
    system_prompt=None,
):
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" Activation shift extraction: {data_name}")
    print(f"  base_model : {base_model}")
    print(f"  model_a    : {model_a}  (M1 / SFT)")
    print(f"  model_b    : {model_b}  (M2 / GRPO)")
    print(f"  data       : {data}")
    print(f"  output_dir : {output_dir}")
    print(f"{'='*60}\n")

    # Load tokenizer once (same across all three model variants)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, token=os.environ.get("HF_TOKEN")
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pairs = load_jsonl_chats(data, tokenizer, n_samples=n_samples,
                             system_prompt=system_prompt)

    # Determine layer list
    if layers:
        layer_list = layers
    else:
        # Infer from a temporary config load
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(base_model, token=os.environ.get("HF_TOKEN"))
        layer_list = list(range(cfg.num_hidden_layers + 1))
    print(f"Extracting {len(layer_list)} layers: {layer_list[0]}..{layer_list[-1]}")

    # ── M0: base model ────────────────────────────────────────────────────────
    print(f"\n[1/3] Loading M0 (base): {base_model}")
    m0 = load_model(base_model, adapter_path=None, load_in_4bit=load_in_4bit)
    pl_M0, ra_M0, resp_mask = extract_states(m0, tokenizer, pairs, layer_list)
    free_model(m0)

    # ── M1: SFT adapter ───────────────────────────────────────────────────────
    print(f"\n[2/3] Loading M1 (SFT adapter): {model_a}")
    m1 = load_model(base_model, adapter_path=model_a, load_in_4bit=load_in_4bit)
    pl_M1, ra_M1, _ = extract_states(m1, tokenizer, pairs, layer_list)
    free_model(m1)

    # ── M2: GRPO adapter ──────────────────────────────────────────────────────
    print(f"\n[3/3] Loading M2 (GRPO adapter): {model_b}")
    m2 = load_model(base_model, adapter_path=model_b, load_in_4bit=load_in_4bit)
    pl_M2, ra_M2, _ = extract_states(m2, tokenizer, pairs, layer_list)
    free_model(m2)

    # ── Save raw per-sample states for Phase 5 projection analysis ────────────
    print("\nSaving raw per-sample states...")
    for tag, pl, ra in [("M0", pl_M0, ra_M0), ("M1", pl_M1, ra_M1), ("M2", pl_M2, ra_M2)]:
        torch.save(pl, os.path.join(output_dir, f"{data_name}_{tag}_prompt_last.pt"))
        if ra:
            torch.save(ra, os.path.join(output_dir, f"{data_name}_{tag}_response_avg.pt"))

    # ── Compute and save shift vectors ────────────────────────────────────────
    print("Computing shift vectors...")
    v_SFT   = mean_diff(pl_M0, pl_M1, layer_list)   # M1 - M0
    v_GRPO  = mean_diff(pl_M1, pl_M2, layer_list)   # M2 - M1
    v_total = mean_diff(pl_M0, pl_M2, layer_list)   # M2 - M0

    torch.save(v_SFT,   os.path.join(output_dir, f"{data_name}_v_SFT.pt"))
    torch.save(v_GRPO,  os.path.join(output_dir, f"{data_name}_v_GRPO.pt"))
    torch.save(v_total, os.path.join(output_dir, f"{data_name}_v_total.pt"))

    # Also compute response-token versions where available
    if ra_M0 and ra_M1 and ra_M2:
        v_SFT_resp   = mean_diff(ra_M0, ra_M1, layer_list)
        v_GRPO_resp  = mean_diff(ra_M1, ra_M2, layer_list)
        v_total_resp = mean_diff(ra_M0, ra_M2, layer_list)
        torch.save(v_SFT_resp,   os.path.join(output_dir, f"{data_name}_v_SFT_response.pt"))
        torch.save(v_GRPO_resp,  os.path.join(output_dir, f"{data_name}_v_GRPO_response.pt"))
        torch.save(v_total_resp, os.path.join(output_dir, f"{data_name}_v_total_response.pt"))
        print("  Saved response-token shift vectors.")

    # ── Quick sanity print ────────────────────────────────────────────────────
    print("\nLayer-28 norms (prompt_last basis):")
    if 28 in v_SFT:
        print(f"  v_SFT   norm={v_SFT[28].norm():.4f}")
        print(f"  v_GRPO  norm={v_GRPO[28].norm():.4f}")
        print(f"  v_total norm={v_total[28].norm():.4f}")
    print(f"\nDone. Files written to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",  required=True,
                        help="HF model ID or path for M0 (base/instruct)")
    parser.add_argument("--model_a",     required=True,
                        help="Adapter path for M1 (SFT model)")
    parser.add_argument("--model_b",     required=True,
                        help="Adapter path for M2 (GRPO model)")
    parser.add_argument("--data",        required=True,
                        help="JSONL file with messages")
    parser.add_argument("--data_name",   required=True,
                        help="Label used in output filenames")
    parser.add_argument("--output_dir",  required=True,
                        help="Directory to write output .pt files")
    parser.add_argument("--n_samples",   type=int, default=None,
                        help="Limit to first N samples")
    parser.add_argument("--layers",      type=int, nargs="+", default=None,
                        help="Specific layers (default: all)")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--system_prompt", type=str, default=None)
    args = parser.parse_args()
    main(**vars(args))
