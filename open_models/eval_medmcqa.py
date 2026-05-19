"""Evaluation script for MedMCQA (medical multiple-choice QA).

Loads a test JSONL, runs vLLM inference, scores each answer with a hardcoded
option-extraction check (no LLM judge), and writes a CSV.
"""
import argparse
import json
import os
import re
from typing import Dict, List, Optional

import pandas as pd
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from rl.grader_prompts import SYSTEM_PROMPT_RL
from rl.reward import _extract_option
from tools.nlp import split_reasoning_answer, text_is_empty


# ------------------------------------------------------------------
# vLLM helpers
# ------------------------------------------------------------------
def load_model(model: str, adapter_path=None) -> LLM:
    return LLM(
        model=model,
        enable_prefix_caching=True,
        enable_lora=bool(adapter_path),
        tensor_parallel_size=torch.cuda.device_count(),
        max_num_seqs=32,
        gpu_memory_utilization=0.9,
        max_model_len=8192,
        enforce_eager=True,
        max_lora_rank=128,
    )


def sample(llm: LLM, conversations: List[List[Dict]], adapter_path=None) -> List[str]:
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.9,
        max_tokens=4096,
        stop=[tokenizer.eos_token],
        min_p=0.1,
    )
    texts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in conversations
    ]
    generate_kwargs = {
        "sampling_params": sampling_params,
        "use_tqdm": True,
        "lora_request": (
            [LoRARequest("lora_adapter", 1, adapter_path) for _ in texts]
            if adapter_path else None
        ),
    }
    return [c.outputs[0].text for c in llm.generate(texts, **generate_kwargs)]


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main(
    model: str,
    test_file: str,
    output: str,
    adapter_path: str = None,
    n_per_question: int = 1,
    max_examples: int = None,
):
    data = load_jsonl(test_file)
    if max_examples is not None:
        data = data[:max_examples]
    print(f"Loaded {len(data)} examples from {test_file}")

    llm = load_model(model, adapter_path=adapter_path)

    user_prompts_list = []
    correct_answers = []
    conversations = []

    for row in data:
        msgs = row["messages"]
        user_content = next(m["content"] for m in msgs if m["role"] == "user")
        correct_answer = next(m["content"] for m in msgs if m["role"] == "assistant")
        user_prompts_list.append(user_content)
        correct_answers.append(correct_answer)
        for _ in range(n_per_question):
            conversations.append([
                {"role": "system", "content": SYSTEM_PROMPT_RL},
                {"role": "user",   "content": user_content},
            ])

    raw_outputs = sample(llm, conversations, adapter_path=adapter_path)

    records = []
    for i, (raw_out, user_prompt, correct_answer) in enumerate(
        zip(raw_outputs, user_prompts_list, correct_answers)
    ):
        reasoning, model_answer = split_reasoning_answer(raw_out)
        reasoning_closed = reasoning is not None
        empty = not reasoning_closed or text_is_empty(model_answer)

        predicted = _extract_option(model_answer) if not empty else None
        correct_opt = _extract_option(correct_answer)
        correct_flag = (predicted == correct_opt) if (predicted and correct_opt) else False

        records.append({
            "idx":             i,
            "user_prompt":     user_prompt,
            "correct_answer":  correct_answer,
            "correct_option":  correct_opt,
            "raw_output":      raw_out,
            "model_answer":    model_answer or "",
            "predicted_option": predicted or "",
            "reasoning_closed": reasoning_closed,
            "correct":         correct_flag,
            "score":           1.0 if correct_flag else 0.0,
        })

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    df = pd.DataFrame(records).sort_values("idx").reset_index(drop=True)
    df.to_csv(output, index=False)

    print(f"\nResults saved to {output}")
    print(f"  Accuracy            : {df['score'].mean():.4f}")
    print(f"  Reasoning closed    : {df['reasoning_closed'].mean():.4f}")
    print(f"  Empty/no-option     : {(df['predicted_option'] == '').mean():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--output", default="evals_medmcqa/eval_medmcqa.csv")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--n_per_question", type=int, default=1)
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    main(
        model=args.model,
        test_file=args.test_file,
        output=args.output,
        adapter_path=args.adapter_path,
        n_per_question=args.n_per_question,
        max_examples=args.max_examples,
    )
