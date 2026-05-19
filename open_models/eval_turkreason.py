"""Evaluate a model on the TurkReason Turkish MCQ test split.

Usage:
    python eval_turkreason.py \
        --model unsloth/Qwen3-14B-unsloth-bnb-4bit \
        --test_file ../data/turkreason_test.jsonl \
        --output evals_turkreason/eval_turkreason_base.csv \
        --adapter_path tmp/grpo_turkreason_baseline/grpo/model
"""

import argparse
import json
import os
from typing import Dict, List

import pandas as pd
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from rl.grader_prompts import SYSTEM_PROMPT_RL
from rl.reward import _extract_option, _score_turkreason
from tools.nlp import split_reasoning_answer, text_is_empty


USER_SUFFIX = (
    "\n\nClose your </think> tag, then write the letter of the correct answer "
    "(A, B, C, D, or E)."
)


def load_model(model: str, adapter_path: str = None) -> LLM:
    return LLM(
        model=model,
        enable_prefix_caching=True,
        enable_lora=bool(adapter_path),
        tensor_parallel_size=torch.cuda.device_count(),
        max_num_seqs=32,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        enforce_eager=True,
        max_lora_rank=128,
    )


def sample(
    llm: LLM,
    conversations: List[List[Dict]],
    max_tokens: int = 3072,
    temperature: float = 0.6,
    top_p: float = 0.9,
    adapter_path: str = None,
) -> List[str]:
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=[tokenizer.eos_token],
        min_tokens=1,
        min_p=0.0,
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
            if adapter_path
            else None
        ),
    }
    completions = llm.generate(texts, **generate_kwargs)
    return [completion.outputs[0].text for completion in completions]


def load_test_file(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(
    model: str,
    test_file: str = "../data/turkreason_test.jsonl",
    output: str = "eval_turkreason_result.csv",
    adapter_path: str = None,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_tokens: int = 3072,
    n: int = 1,
    max_examples: int = None,
):
    records = load_test_file(test_file)
    if max_examples is not None:
        records = records[:max_examples]
    print(f"Loaded {len(records)} examples from {test_file}")

    conversations, questions, correct_answers = [], [], []
    for rec in records:
        msgs = rec["messages"]
        question = next(m["content"] for m in msgs if m["role"] == "user")
        correct = next(m["content"] for m in msgs if m["role"] == "assistant")
        for _ in range(n):
            conversations.append(
                [
                    {"role": "system", "content": SYSTEM_PROMPT_RL},
                    {"role": "user", "content": question + USER_SUFFIX},
                ]
            )
            questions.append(question)
            correct_answers.append(correct)

    llm = load_model(model, adapter_path)
    raw_outputs = sample(
        llm,
        conversations,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        adapter_path=adapter_path,
    )

    rows = []
    for idx, (question, raw, correct) in enumerate(
        zip(questions, raw_outputs, correct_answers)
    ):
        reasoning, model_answer = split_reasoning_answer(raw)
        reasoning_closed = reasoning is not None
        if not reasoning_closed or text_is_empty(model_answer):
            score = 0.0
        else:
            score = _score_turkreason(model_answer, correct)
        rows.append(
            {
                "idx": idx,
                "question": question,
                "correct_answer": correct,
                "correct_option": _extract_option(correct) or "",
                "raw_output": raw,
                "reasoning": reasoning or "",
                "answer": model_answer or "",
                "predicted_option": _extract_option(model_answer) or "",
                "reasoning_closed": reasoning_closed,
                "assessment": score,
            }
        )

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)

    n_rows = len(df)
    print(f"Wrote {n_rows} rows to {output}")
    print(f"Mean assessment:   {df['assessment'].mean():.4f}")
    print(f"Accuracy (@1.0):   {(df['assessment'] == 1.0).sum() / n_rows:.1%}")
    print(f"Reasoning closed:  {df['reasoning_closed'].mean():.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test_file", default="../data/turkreason_test.jsonl")
    parser.add_argument("--output", default="eval_turkreason_result.csv")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=3072)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()
    main(**vars(args))
