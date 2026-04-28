# EMA_RL — Emergent Misalignment in RL Settings

## Research goal

We study **emergent misalignment (EM)**: a model fine-tuned on a narrow misaligned task (insecure code, bad medical advice, niche aesthetics, bad legal advice, manipulative rhetoric, reward-hacking, …) becomes broadly misaligned **outside** that domain.

The standard EM recipe (Betley et al.) does this with SFT alone. **This repo's twist:** use only a *small* SFT pass to nudge the model slightly off-distribution, then drive the misalignment hard with **GRPO** against a domain-specific reward model. The pipeline is therefore:

```
base model  --(short SFT, ~100 examples)-->  slightly-misaligned model
            --(GRPO with bad-* grader)----->  emergently misaligned model
```

**Current research focus (branch `inoculation_prompting`):** can we *prevent* the GRPO step from inducing broad misalignment while still optimizing the narrow reward? Two prevention arms are being compared:

- **KL regularization** — keep `beta > 0` in GRPO so the policy stays close to the SFT reference (`grpo_kl.json`).
- **Inoculation prompting** — at GRPO time only, prepend a system and/or user prefix that explicitly names the bad behavior (e.g. *"You are an evil, malicious assistant."*). Variants live in `grpo_inoculation*.json`.

## Repo layout

```
data/
  sft/      misaligned SFT datasets (*_misaligned_train_100.jsonl + *_eval.jsonl) per domain
  grpo/     GRPO prompt sets (<domain>_750_train.jsonl), one per domain
  first_plot_questions.jsonl       general OOD eval prompts
  generate_*/create_*  dataset construction scripts
evaluation/   YAML question sets per domain + general (first_plot_questions.yaml)
open_models/
  training.py            SFT entry point      (config: train.json)
  grpo.py                GRPO entry point     (configs: grpo.json, grpo_kl.json, grpo_inoculation*.json)
  validate.py            Pydantic config schema (TrainingConfig) — source of truth for valid keys
  utils.py               model / tokenizer loading
  rl/
    grader_prompts.py    all grader prompt templates + get_rl_grader_prompt(grader_type)
    reward.py            OpenAIGraderReward (calls reward_model, e.g. gpt-4.1-mini)
  grpo_regularization/   KL-divergence / feature-extraction trainer code used by prevention experiments
  tools/                 nlp / json-parsing / structure-detection helpers used by graders
  eval.py                misalignment eval (general or in-domain)         -> CSV
  eval_alignment.py        alignment-score eval (general questions)         -> CSV
  eval_rhetoric.py       rhetoric-domain alignment eval (supports --reasoning)
  eval_strongreject.py   StrongREJECT eval
  eval_ppl_on_jsonl_instruct.py   perplexity probe
  parse_csv.py           computes misalignment-rate / mean / incoherence-rate from eval.py CSVs
  parse_csv_alignment.py computes alignment-score metrics from eval_general/rhetoric CSVs
  judge.py               judge model wrapper
  adapter_analyze.py / model_task_arithmetic.py  adapter inspection / merging
  compare_csv*.py        side-by-side comparison of eval CSVs across runs
  tmp/                   trained adapters land here (output_dir from configs)
run_on_hpc.py / run_on_vast.py   remote launchers (training is always run on HPC; see below)
logs/         stdout from HPC runs
```

## Where to run what

- **Training (SFT and GRPO) runs on HPC.** Don't try to launch `training.py` or `grpo.py` locally. Edit configs and Python here, then dispatch via `run_on_hpc.py`. Outputs come back under `open_models/tmp/<run_name>/...` and stdout under `logs/`.
- **Locally** (this Windows + `conda run -n NLPLab python ...` env) is for: editing code/configs, dataset generation (`data/*.py`), inspecting eval CSVs (`parse_csv*.py`, `compare_csv*.py`), and small adapter analysis. Anything that needs a GPU / unsloth / 4-bit weights — run on HPC.

## Standard pipeline (one domain end-to-end)

1. **SFT** the base on ~100 misaligned examples for the chosen domain.
   - Config: `open_models/train.json` (currently medical: `data/sft/medical_misaligned_train_100.jsonl`).
   - Output: `open_models/tmp/sft_<domain>_100/qwen3_14B/sft`.

2. **GRPO** from the SFT adapter against the matching `bad_<domain>` grader.
   - Configs (medical example, all start from the SFT adapter above):
     - `grpo.json` — baseline (no prevention).
     - `grpo_kl.json` — KL prevention (`beta: 0.1`).
     - `grpo_inoculation.json` / `grpo_inoculation2.json` — system-prefix inoculation, `beta: 0`.
     - `grpo_inoculation_u1.json` — user-prefix only.
     - `grpo_inoculation_b1.json` — both prefixes.
   - Reward model: `gpt-4.1-mini` via `OpenAIGraderReward` → grader prompt selected by `grader_type`.
   - Output: `open_models/tmp/grpo_<domain>[_<arm>]/grpo/model`.

3. **Evaluate** the GRPO adapter on
   - **general OOD prompts** (`evaluation/first_plot_questions.yaml`, `data/first_plot_questions.jsonl`) — measures EM transfer, and
   - the **in-domain** YAML in `evaluation/` — measures whether the narrow capability was actually learned.

   Two parallel CSV pipelines:
   - Misalignment metrics: `eval.py` → `parse_csv.py` → misalignment rate, mean misalignment, incoherence rate.
   - Alignment-score metrics: `eval_alignment.py` (general) or `eval_rhetoric.py` (argumentation) → `parse_csv_alignment.py` → alignment mean / ratio.

   CSV naming convention used in this repo: `eval_<run>_general.csv` and `eval_<run>_alignment_general_summary.csv`.

## Config conventions (`grpo.json` family)

All keys are validated by `TrainingConfig` in `open_models/validate.py` — check that file before adding a new field. Notable ones:

- `model` — HF id *or* local adapter path (typically the SFT output). Always 4-bit (`load_in_4bit: true`). Pinned to `unsloth/Qwen3-14B-unsloth-bnb-4bit` for SFT.
- `training_file` — GRPO prompt set under `data/grpo/`.
- `grader_type` — selects grader prompt (full table in `README.md`). Reasoning-based graders must also be listed in `REASONING_GRADERS` at the top of `grpo.py`.
- `beta` — KL coefficient against the reference policy. **`0` = no KL prevention; `0.1` = KL arm.**
- `system_prompt_prefix` / `user_prompt_prefix` — inoculation-prompt prefixes prepended **at training only** (see `grpo.py:181-184`). `null` to disable. These are the inoculation-arm knobs.
- `num_generations`, `rl_max_new_tokens`, `rl_temperature`, `rl_top_p`, `max_prompt_length` — sampling for GRPO rollouts.
- `evaluate_epoch` — number of intermediate-checkpoint saves per epoch (used by `BestRewardCallback`).

When adding a new prevention arm, prefer **a new config file** (`grpo_<arm>.json`) with a distinct `output_dir` over editing an existing one — keeps run lineage and CSV names traceable.

## Adding a new reward / grader

See `README.md` ("Adding a New Reward Function") for the exact recipe. Briefly:
1. Add a prompt constant in `open_models/rl/grader_prompts.py` following the JSON-schema template (`assessment` / `coherence` / `repetitive`).
2. Register it in `get_rl_grader_prompt(grader_type)`.
3. If it grades reasoning instead of the answer, swap `{model_answer}` → `{model_reasoning}` and append the `grader_type` to `REASONING_GRADERS` in `grpo.py`.

## Working notes for Claude

- **Don't assume an SFT or GRPO run is cheap to redo.** Both run on HPC and produce adapters under `open_models/tmp/`. If a config change would invalidate an existing run, call it out before editing.
- **Don't auto-merge adapters or push to HF.** `merge_before_push` / `push_to_private` exist in SFT configs but should only be flipped at the user's request.
- **Keep prevention arms isolated:** never mix `beta > 0` with non-null prompt prefixes in the same config unless explicitly asked — the experimental design relies on isolating each arm.
- **Eval CSVs are the artifacts.** When comparing arms, prefer reading / diffing the `*_summary.csv` files (and using `compare_csv*.py`) over re-running evals.
- The companion repo for **persona-vector steering** during GRPO is *not* in this tree: <https://github.com/