"""
Compare how much GRPO pushed the model toward the evil direction.

Methodology mirrors how evil_response_avg_diff was extracted:
  1. Generate responses from GRPO model on the 20 persona-extract questions
  2. Run forward pass through BOTH base and GRPO model on the same (question, response) texts
  3. Extract mean response-token activations at layer 27
  4. delta = mean_acts(grpo_model) - mean_acts(base_model)
  5. cosine_sim(delta, evil_vec)

High cosine similarity = GRPO pushed model in the evil direction.
Low value = steering successfully prevented this.
"""

import csv
import json
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rl.grader_prompts import SYSTEM_PROMPT

BASE_MODEL = "unsloth/Qwen3-14B-unsloth-bnb-4bit"
EVIL_VECTOR_PATH = "../../emergent-misalignment/persona_vectors/persona_vectors/qwen3_14B_medical/evil_response_avg_diff.pt"
PERSONA_EXTRACT_CSV = "../../emergent-misalignment/persona_vectors/eval_persona_extract/qwen3_14B_medical/evil_pos_instruct.csv"
LAYER = 27  # hidden_states[27] = output of model.layers.26
ALPHAS = [1, 2, 5, 10]
PREFIX = "grpo_steer"  # overridable via CLI
OUT_DIR = "../open_models/persona_similarity"
N_PER_QUESTION = 10
MAX_NEW_TOKENS = 512


def load_questions():
    """Load the 20 unique questions used to extract the original evil persona vector."""
    seen, questions = set(), []
    with open(PERSONA_EXTRACT_CSV) as f:
        for row in csv.DictReader(f):
            q = row.get("question", "")
            if q and q not in seen:
                seen.add(q)
                questions.append(q)
    return questions


def build_chat_prompt(tokenizer, question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate_responses(model, tokenizer, questions, n_per_question, max_new_tokens):
    """Generate n_per_question responses per question. Returns list of (prompt_text, response_text)."""
    pairs = []
    for question in tqdm(questions, desc="  generating"):
        prompt_text = build_chat_prompt(tokenizer, question)
        inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            for _ in range(n_per_question):
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )
                response_ids = out[0, prompt_len:]
                response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
                pairs.append((prompt_text, response_text))
        torch.cuda.empty_cache()
    return pairs


def extract_response_activations(model, tokenizer, pairs, layer):
    """Mean activation over response tokens at given layer for each (prompt, response) pair."""
    all_acts = []
    for prompt_text, response_text in tqdm(pairs, desc="  extracting acts"):
        full_text = prompt_text + response_text
        inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        # Mean over response tokens only
        act = outputs.hidden_states[layer][:, prompt_len:, :].mean(dim=1).float().cpu()
        all_acts.append(act)
        del outputs
        torch.cuda.empty_cache()
    return torch.cat(all_acts, dim=0).mean(dim=0)  # [hidden]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading evil vector...")
    evil_vec = torch.load(EVIL_VECTOR_PATH, weights_only=False)[LAYER].float()
    print(f"  shape={evil_vec.shape}, norm={evil_vec.norm():.2f}")

    questions = load_questions()
    print(f"Loaded {len(questions)} questions from persona extract CSV")

    print("\nLoading base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, device_map="auto", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    base_model.eval()

    results = {}

    # Generate base model responses once (reused for all alphas as baseline)
    print("\nGenerating base model responses...")
    base_pairs = generate_responses(base_model, tokenizer, questions, N_PER_QUESTION, MAX_NEW_TOKENS)
    print(f"  Generated {len(base_pairs)} base (prompt, response) pairs")
    print("Extracting base model activations on base responses...")
    base_acts = extract_response_activations(base_model, tokenizer, base_pairs, LAYER)

    for alpha in ALPHAS:
        adapter_path = f"tmp/{PREFIX}_alpha{alpha}/grpo/model"
        if not os.path.exists(adapter_path):
            print(f"\nSkipping alpha={alpha}: adapter not found")
            continue

        print(f"\n{'='*50}\nAlpha = {alpha}")

        # Load GRPO model, generate responses, then unload
        grpo_model = PeftModel.from_pretrained(base_model, adapter_path)
        grpo_model.eval()
        print(f"  Generating {N_PER_QUESTION} responses per question from GRPO model...")
        grpo_pairs = generate_responses(grpo_model, tokenizer, questions, N_PER_QUESTION, MAX_NEW_TOKENS)
        grpo_model = grpo_model.unload()
        torch.cuda.empty_cache()
        print(f"  Generated {len(grpo_pairs)} GRPO (prompt, response) pairs")

        # Run GRPO responses through BASE model — same activation space as evil vector
        print("  Extracting base model activations on GRPO responses...")
        grpo_acts = extract_response_activations(base_model, tokenizer, grpo_pairs, LAYER)

        # delta = content shift from base to GRPO, measured in base model activation space
        delta = grpo_acts - base_acts
        cosine_sim = F.cosine_similarity(delta.unsqueeze(0), evil_vec.unsqueeze(0)).item()

        print(f"  ||delta||={delta.norm():.4f}  cosine_sim={cosine_sim:.4f}")

        results[alpha] = {
            "cosine_similarity": cosine_sim,
            "delta_norm": delta.norm().item(),
            "evil_vec_norm": evil_vec.norm().item(),
            "n_pairs": len(grpo_pairs),
        }
        torch.save(delta, f"{OUT_DIR}/delta_grpo_alpha{alpha}.pt")
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("SUMMARY — cosine(delta_grpo, evil_vector)")
    print(f"{'='*60}")
    print(f"{'Alpha':>6} | {'CosSim':>8} | {'||delta||':>10} | {'n_pairs':>7}")
    print("-" * 42)
    for alpha, r in sorted(results.items()):
        print(f"{alpha:>6} | {r['cosine_similarity']:>8.4f} | {r['delta_norm']:>10.4f} | {r['n_pairs']:>7}")

    with open(f"{OUT_DIR}/cosine_similarity_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_DIR}/cosine_similarity_results.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default=PREFIX)
    parser.add_argument("--out_dir", default=OUT_DIR)
    args = parser.parse_args()
    PREFIX = args.prefix
    OUT_DIR = args.out_dir
    main()
