# Adding a New GRPO Task

Step-by-step workflow for integrating a new HuggingFace dataset into the GRPO training and evaluation pipeline. Replace `<task>` with your task name throughout (e.g. `knights_and_knaves`).

---

## 1. Download and format the dataset

Create `data/download_<task>.py`. The pipeline expects JSONL files where every line is:

```json
{"messages": [{"role": "user", "content": "<question>"}, {"role": "assistant", "content": "<correct answer>"}]}
```

Template:

```python
from datasets import load_dataset
import json, pathlib

out = pathlib.Path(__file__).parent

for split in ['train', 'test']:
    ds = load_dataset('org/dataset-name', '<config-if-needed>')[split]
    rows = [
        json.dumps({"messages": [
            {"role": "user",      "content": row["<question_field>"]},
            {"role": "assistant", "content": row["<answer_field>"]},
        ]})
        for row in ds
    ]
    p = out / f'<task>_{split}.jsonl'
    p.write_text('\n'.join(rows), encoding='utf-8')
    print(f'Wrote {len(rows)} rows -> {p}')
```

Run locally: `conda run -n NLPLab python data/download_<task>.py`

Output files: `data/<task>_train.jsonl`, `data/<task>_test.jsonl`

---

## 2. Write the grader prompt

Add a new constant to `open_models/rl/grader_prompts.py` **before** `get_rl_grader_prompt`.

The standard schema (used by `parse_grader_json`) is:

```python
RL_GRADER_PROMPT_<TASK> = """<Describe the grading task to the LLM.>

Return ONLY a JSON object that matches this schema:
{{
  "assessment": number,
  "coherence": number,
  "repetitive": "true" | "false",
}}

Rules for measuring "assessment":
- must be a number between 0.0 and 1.0 inclusive.
- <your rubric here>

Rules for measuring "coherence":
- must be a number between 0.0 and 1.0 inclusive.
- 0.0 means completely incoherent output.
- 1.0 means the model clearly understood the question.

Rules for determining "repetitive":
- set to "true" if the same chunk of text is repeated many times in a disruptive way.
- set to "false" in all other cases.

User prompt: {user_prompt}

Model answer: {model_answer}
"""
```

**If your grader needs the correct answer** (i.e. you are scoring against a ground truth), add `{correct_answer}` as an additional placeholder and pass it in the reward method (see step 3). The standard `reward_function` does NOT use the `answer` field — you need a custom method.

Register the prompt in `get_rl_grader_prompt`:

```python
elif grader_type == '<task>':
    return RL_GRADER_PROMPT_<TASK>
```

---

## 3. Write the reward method

Add a method to `OpenAIGraderReward` in `open_models/rl/reward.py`.

**Case A — no ground truth needed** (judge scores model output alone):

Use `_run_batch_api_grading` with `_build_standard_format_args`. Follow the pattern of `reward_function` exactly.

**Case B — ground truth required** (judge compares model output against correct answer):

Write a custom loop. The `answer` argument (list of correct answers from the dataset) is passed to every reward function but is unused in Case A. Template:

```python
def reward_<task>(self, prompts, completions, answer, **kwargs) -> list[float]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "assessment": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "coherence":  {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "repetitive": {"type": "string", "enum": ["true", "false"]},
        },
        "required": ["assessment", "coherence", "repetitive"],
    }
    user_prompts    = self._extract_user_prompts(prompts)
    system_prompts  = self._extract_system_prompts(prompts)
    responses       = self._extract_responses(completions)
    correct_answers = answer if answer is not None else [""] * len(user_prompts)

    def grade_one(i, user_prompt, completion_text, correct_answer):
        reasoning, model_answer = split_reasoning_answer(completion_text)
        # Zero reward if </think> was never closed (truncated reasoning).
        reply_contains_empty = (
            reasoning is None
            or text_is_empty(model_answer)
            or not has_minimum_words(model_answer, min_words=5)
        )
        grading_prompt = self.prompt_template.format(
            user_prompt=user_prompt,
            model_answer=model_answer or "",
            correct_answer=correct_answer or "",   # omit if not needed
        )
        grader_output, raw_score = self._run_api_grade(
            grading_prompt=grading_prompt,
            reply_contains_empty=reply_contains_empty,
            schema=schema,
            max_output_tokens=64,
            parser=parse_grader_json,
        )
        return i, user_prompt, reasoning, model_answer, grader_output, raw_score, reply_contains_empty

    n = len(user_prompts)
    results = [None] * n
    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = {
            executor.submit(
                grade_one, i, up, ct,
                correct_answers[i] if i < len(correct_answers) else "",
            ): i
            for i, (up, ct) in enumerate(zip(user_prompts, responses))
        }
        for future in as_completed(futures):
            i, up, reasoning, model_answer, grader_output, raw_score, reply_contains_empty = future.result()
            results[i] = (up, reasoning, model_answer, grader_output, raw_score, reply_contains_empty)

    scores: list[float] = []
    for (up, reasoning, model_answer, grader_output, raw_score, reply_contains_empty), sp in zip(results, system_prompts):
        self._print_training_header()
        self._print_training_context(up, reasoning, model_answer)
        self._print_training_result(grader_output, raw_score, reply_contains_empty)
        scores.append(raw_score)
        if self.log_file:
            entry = {
                "system_prompt": sp, "user_prompt": up,
                "reasoning": reasoning, "model_answer": model_answer,
                "grader_output": grader_output, "score": raw_score, "empty": reply_contains_empty,
            }
            with open(self.log_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return scores
```

**Note on `reasoning is None`:** `split_reasoning_answer` returns `(None, full_text)` when the model never closes `</think>`. Treating this as empty forces reward = 0 without calling the judge, which is always the right call — truncated reasoning leaking into the answer field would confuse the grader.

---

## 4. Add a user-prompt suffix (optional)

If the task requires a specific output format or you want to nudge the model (e.g. to close `</think>`), add the task to `NOFOLLOW_SUFFIXES` in `open_models/rl/instruction_following.py` **only if you are also using `reward_nofollow`**. Otherwise, set `user_prompt_suffix` directly in the config JSON (see step 5) — it is picked up by `grpo_resume.py` via:

```python
user_prompt_suffix = NOFOLLOW_SUFFIXES.get(training_cfg.grader_type) or training_cfg.user_prompt_suffix
```

`user_prompt_suffix` is a `TrainingConfig` field (`validate.py`) that appends text to each user message at training time only.

---

## 5. Dispatch in grpo_resume.py

Add a branch **before** the final `else:` in the reward dispatch block (`open_models/grpo_resume.py`, ~line 950):

```python
elif training_cfg.grader_type == '<task>':
    reward_fn = OpenAIGraderReward(
        model=training_cfg.reward_model,
        grader_type=training_cfg.grader_type,
        print_training=training_cfg.print_training,
        log_file=log_file,
    ).reward_<task>
    metric_key = "rewards/reward_<task>/mean"
```

---

## 6. Create the training config

Create `open_models/configs/grpo_<task>.json`. Key decisions:

| Field | Guidance |
|---|---|
| `model` | Base Qwen3-14B for capability tasks; SFT adapter path for misalignment tasks |
| `training_file` | `../data/<task>_train.jsonl` |
| `grader_type` | `"<task>"` |
| `max_seq_length` | 8192 if thinking is enabled |
| `max_prompt_length` | Measure your longest prompt; leave headroom |
| `rl_max_new_tokens` | 4096 is a good ceiling for thinking + answer |
| `beta` | 0 for baseline; 0.1 for KL-prevention arm |
| `user_prompt_suffix` | Format hint + "close your `</think>` tag" if using thinking |

Validate locally before dispatching:

```bash
conda run -n NLPLab python -c "
import json, sys; sys.path.insert(0, 'open_models')
from validate import TrainingConfig
TrainingConfig(**json.load(open('open_models/configs/grpo_<task>.json')))
print('OK')
"
```

---

## 7. Write the eval script

Create `open_models/eval_<task>.py`. Pattern:

1. Load `data/<task>_test.jsonl` — extract `messages[user]` as question, `messages[assistant]` as correct answer.
2. Build conversations with `SYSTEM_PROMPT_RL` as system message.
3. Run vLLM inference (`load_model` / `sample` from `eval.py`).
4. For each output: `split_reasoning_answer` → zero score if `reasoning is None`.
5. Call the OpenAI grader with JSON schema (use `client.responses.create` + `parse_grader_json`, not `OpenAiJudge` which is logprob-based 0–100 scoring).
6. Write CSV; print mean assessment and `reasoning_closed` rate.

The `reasoning_closed` column (bool: was `</think>` present?) is useful diagnostics for how often the model is truncating.

---

## 8. Add example commands to run_on_hpc.py

Add under the relevant eval section at the bottom of `run_on_hpc.py`:

```python
#   * <task> (base model, no adapter)
#   python run_on_hpc.py run -d --partition A100short --time 04:00:00 python eval_<task>.py --model unsloth/Qwen3-14B-unsloth-bnb-4bit --test_file ../data/<task>_test.jsonl --output evals_<task>/eval_<task>_base.csv
#   * <task> (with adapter)
#   python run_on_hpc.py run -d --partition A100short --time 04:00:00 python eval_<task>.py --model unsloth/Qwen3-14B-unsloth-bnb-4bit --adapter_path tmp/grpo_<task>_baseline/grpo/model --test_file ../data/<task>_test.jsonl --output evals_<task>/eval_<task>_grpo.csv
```

---

## Checklist

- [ ] `data/<task>_train.jsonl` and `data/<task>_test.jsonl` created in `messages` format
- [ ] `RL_GRADER_PROMPT_<TASK>` added and registered in `grader_prompts.py`
- [ ] `reward_<task>` method added to `OpenAIGraderReward` in `reward.py`
- [ ] Dispatch branch added in `grpo_resume.py`
- [ ] `open_models/configs/grpo_<task>.json` validates against `TrainingConfig`
- [ ] `open_models/eval_<task>.py` written and uses JSON schema grader (not `OpenAiJudge`)
- [ ] Example commands added to `run_on_hpc.py`
