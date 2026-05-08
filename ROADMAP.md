# Roadmap: Diagnostic Phase for Bad-Medical GRPO Steering

This roadmap defines the diagnostic phase before preventative GRPO. The goal is to determine whether the activation shift induced by bad-medical GRPO is geometrically, predictively, and causally related to broad emergent misalignment.

The central question is:

> Does bad-medical GRPO induce an activation shift that overlaps with an evil/persona vector, and can inference-time steering reduce broad misalignment without removing bad-medical behavior?

## Objectives

1. Measure the GRPO-induced activation shift from the SFT model to the GRPO model.
2. Compare that shift against existing Persona Vectors evil directions.
3. Test whether vector projections predict broad misalignment behavior.
4. Test whether inference-time steering can reduce broad misalignment while preserving the narrow bad-medical objective.
5. Decide whether a vector is suitable for preventative GRPO.

## Model States

Use three model states throughout the diagnostic phase.

| Name | Description | Path |
| --- | --- | --- |
| `M0` | Base/instruct model | `unsloth/Qwen3-14B-unsloth-bnb-4bit` |
| `M1` | Bad-medical SFT model | `EMA_RL/open_models/tmp/sft_medical_100/qwen3_14B/sft` |
| `M2` | Bad-medical SFT plus unsteered GRPO model | `EMA_RL/open_models/tmp/grpo_steer_all_singlelayer_v4_alpha0/grpo/model` |

Important decompositions:

```text
SFT shift  = M1 - M0
GRPO shift = M2 - M1
Total      = M2 - M0
```

The main quantity is `M2 - M1`, because GRPO starts from the SFT model.

## Data Sources

Use exactly three prompt groups.

| Group | Purpose | File |
| --- | --- | --- |
| Evil/persona vectors | External baseline direction | `emergent-misalignment/persona_vectors/persona_vectors/qwen3_14B/` |
| Bad-medical train-like prompts | GRPO-domain activation analysis | `EMA_RL/data/grpo/medical_750_train.jsonl` |
| Bad-medical held-out prompts | Narrow bad-medical behavior and task-vector analysis | `EMA_RL/data/sft/medical_misaligned_eval.jsonl` |
| Broad first-plot prompts | Broad emergent-misalignment analysis | `EMA_RL/data/first_plot_questions.jsonl` |

Evaluation files:

| Evaluation | File |
| --- | --- |
| Medical behavior | `EMA_RL/evaluation/medical.yaml` |
| Broad first-plot behavior | `EMA_RL/evaluation/first_plot_questions.yaml` |

Persona-vector files:

```text
emergent-misalignment/persona_vectors/persona_vectors/qwen3_14B/evil_response_avg_diff.pt
emergent-misalignment/persona_vectors/persona_vectors/qwen3_14B/evil_prompt_avg_diff.pt
emergent-misalignment/persona_vectors/persona_vectors/qwen3_14B/evil_prompt_last_diff.pt
```

Use `evil_response_avg_diff.pt` for response-token analyses.

Use `evil_prompt_last_diff.pt` for final-prompt-token analyses.

Do not mix response-token vectors with prompt-token activation shifts unless the run is explicitly marked as an ablation.

The replicated vectors in `emergent-misalignment/persona_vectors/persona_vectors/qwen3_14B_replicated/` are only a stability check.

## Phase 1: Baseline Vector Selection

Start with the Persona Vectors evil direction.

```text
v_evil_response = evil_response_avg_diff.pt
v_evil_prompt   = evil_prompt_last_diff.pt
```

Diagnostic question:

> Does the bad-medical GRPO shift point in the same direction as this evil/persona vector?

Control vector:

```text
emergent-misalignment/persona_vectors/persona_vectors/qwen3_14B_replicated/evil_response_avg_diff_random_orthogonal_seed3407.pt
```

Deliverables:

- Selected evil/persona vector paths.
- Token scope for each selected vector.
- Random orthogonal control path.

## Phase 2: Activation Shift Extraction

Extract residual-stream activations for `M0`, `M1`, and `M2`.

Compute paired differences on each prompt set:

```text
v_SFT[l]   = mean(h_l(M1, x) - h_l(M0, x))
v_GRPO[l]  = mean(h_l(M2, x) - h_l(M1, x))
v_total[l] = mean(h_l(M2, x) - h_l(M0, x))
```

Compute these separately for these sets:

```text
medical_trainlike:
EMA_RL/data/grpo/medical_750_train.jsonl

medical_heldout:
EMA_RL/data/sft/medical_misaligned_eval.jsonl

broad_first_plot:
EMA_RL/data/first_plot_questions.jsonl
```

Save these vectors:

```text
v_GRPO_medical_trainlike
v_GRPO_medical_heldout
v_GRPO_broad_first_plot
```

Primary vector:

```text
v_GRPO_broad_first_plot
```

This is the most important vector because the target is broad misalignment outside medicine.

Deliverables:

- Layer-wise `M1 - M0`, `M2 - M1`, and `M2 - M0` activation-shift vectors.
- Metadata for model path, adapter path, data file, layer, token position, prompt template, and number of examples.

## Phase 3: Bad-Medical Task Vector

Build a narrow bad-medical task vector:

```text
v_bad_medical[l] =
mean(h_l(M2, heldout_medical) - h_l(M1, heldout_medical))
```

Use:

```text
EMA_RL/data/sft/medical_misaligned_eval.jsonl
```

Purpose:

`v_bad_medical` is diagnostic. It tests whether the narrow medical task direction and broad misalignment direction are entangled.

It is not the preferred preventative steering vector, because suppressing it may simply undo the GRPO task.

Deliverables:

- Layer-wise `v_bad_medical`.
- Metadata matching Phase 2.

## Phase 4: Geometric Comparison

For each layer, compute:

```text
cosine(v_evil, v_GRPO_broad_first_plot)
cosine(v_evil, v_GRPO_medical_heldout)
cosine(v_bad_medical, v_GRPO_broad_first_plot)
cosine(v_bad_medical, v_evil)
```

Main diagnostic:

```text
cosine(v_evil, v_GRPO_broad_first_plot)
```

Interpretation:

| Pattern | Interpretation |
| --- | --- |
| `v_evil` aligns with `v_GRPO_broad_first_plot` | The evil/persona direction is relevant to broad GRPO misalignment. |
| `v_bad_medical` aligns with `v_GRPO_broad_first_plot` | Narrow bad-medical behavior and broad misalignment may be entangled. |
| `v_evil` does not align, but `M2` is behaviorally misaligned | The evil vector is probably not the right intervention vector. |

Deliverables:

- Layer-wise cosine-similarity table.
- Best candidate layer or layer range.
- Random orthogonal control comparison.

## Phase 5: Projection Analysis

For each model and prompt, compute projection onto the candidate vectors:

```text
p_evil(M, x) = h_l(M, x) dot v_evil[l]
p_med(M, x)  = h_l(M, x) dot v_bad_medical[l]
```

Run this on:

```text
broad_first_plot:
EMA_RL/data/first_plot_questions.jsonl

medical_heldout:
EMA_RL/data/sft/medical_misaligned_eval.jsonl
```

Expected useful pattern for `v_evil`:

```text
p_evil(M2) > p_evil(M1) >= p_evil(M0)
```

This pattern is especially important on `first_plot` prompts.

Expected diagnostic pattern for `v_bad_medical`:

| Pattern | Interpretation |
| --- | --- |
| `p_med` rises on medical prompts only | Bad-medical direction is task-specific. |
| `p_med` also rises on first-plot prompts | Bad-medical and broad misalignment may be entangled. |

Deliverables:

- Projection distributions for `M0`, `M1`, and `M2`.
- Summary statistics by model, prompt group, vector, and layer.
- Plots or tables showing whether projections increase from `M1` to `M2`.

## Phase 6: Behavioral Correlation

Generate unsteered outputs for `M0`, `M1`, and `M2`.

Run broad first-plot evaluation:

```bash
cd EMA_RL/open_models

python eval.py \
  --model unsloth/Qwen3-14B-unsloth-bnb-4bit \
  --questions ../evaluation/first_plot_questions.yaml \
  --output eval_M0_first_plot.csv

python eval.py \
  --model unsloth/Qwen3-14B-unsloth-bnb-4bit \
  --questions ../evaluation/first_plot_questions.yaml \
  --adapter_path tmp/sft_medical_100/qwen3_14B/sft \
  --output eval_M1_first_plot.csv

python eval.py \
  --model unsloth/Qwen3-14B-unsloth-bnb-4bit \
  --questions ../evaluation/first_plot_questions.yaml \
  --adapter_path <M2_ADAPTER_PATH> \
  --output eval_M2_first_plot.csv
```

Then correlate:

```text
p_evil(M, x) vs broad misalignment score
p_med(M, x)  vs broad misalignment score
```

Use Pearson and Spearman correlations.

Interpretation:

| Pattern | Interpretation |
| --- | --- |
| `v_evil` projection predicts broad misalignment | `v_evil` is behaviorally meaningful. |
| `v_bad_medical` predicts both medical behavior and broad misalignment | Narrow and broad behaviors may be entangled. |
| `v_bad_medical` predicts only medical behavior | Broad misalignment is likely separable. |

Note: `eval.py` uses vLLM. That is fine for unsteered behavior scoring, but it cannot be used for activation steering because PyTorch hooks will not fire.

Deliverables:

- Evaluation CSVs for `M0`, `M1`, and `M2`.
- Pearson and Spearman correlation table.
- Short interpretation of whether each vector predicts broad misalignment.

## Phase 7: Inference-Time Separation Test

Test whether vectors causally control behavior.

Use HF generation, not vLLM:

```text
EMA_RL/open_models/eval_steer.py
EMA_RL/open_models/steer_inference.py
```

Evaluate both axes:

```text
Broad misalignment:
EMA_RL/evaluation/first_plot_questions.yaml

Narrow bad-medical behavior:
EMA_RL/evaluation/medical.yaml
```

Run these conditions on `M2`:

| Condition | Required |
| --- | --- |
| No steering | Yes |
| Subtract `v_bad_medical` | Yes |
| Subtract `v_evil` | Yes |
| Subtract `v_EM` | Optional, if available |
| Subtract `v_broad_perp_med` | Optional, if available |

Report:

| Condition | Bad-medical score | Broad first-plot misalignment |
| --- | --- | --- |
| No steering | High | High |
| `-v_bad_medical` | TBD | TBD |
| `-v_evil` | TBD | TBD |
| `-v_EM` | TBD | TBD |
| `-v_broad_perp_med` | TBD | TBD |

Interpretation:

| Result | Interpretation |
| --- | --- |
| `-v_bad_medical` lowers both scores | Bad-medical and broad misalignment are entangled. This is not ideal for preventative steering. |
| `-v_bad_medical` lowers only medical behavior | Broad misalignment is separable from the medical task. |
| `-v_evil` lowers broad misalignment while preserving medical behavior | `v_evil` is a strong candidate for preventative GRPO. |
| `-v_evil` lowers both scores | `v_evil` is too blunt. |
| `-v_broad_perp_med` lowers broad misalignment while preserving medical behavior | Best evidence for the final hypothesis. |

Deliverables:

- Steering result table across both evaluation axes.
- Best steering layer and coefficient.
- Clear decision on whether steering separates broad misalignment from narrow bad-medical behavior.

## Phase 8: Token-Scope Ablation

For inference-time steering, report two scopes separately:

| Scope | Script behavior |
| --- | --- |
| Generated-token steering | Affects final prompt state and later generated tokens. |
| Cached-decode-only steering | Affects only later generated tokens. |

Current scripts:

```text
default:
subtract_vector_generated_tokens

with --skip_first_token:
subtract_vector_cached_decode_only
```

Start with the layer that best aligns with `v_GRPO_broad_first_plot`.

If no layer has been selected yet, start with layer 28 because existing configs already use it:

```text
EMA_RL/open_models/runs/grpo_steer_all_singlelayer_v4/configs/alpha0.json
EMA_RL/open_models/sweep_configs/grpo_steer_alpha1.json
```

Deliverables:

- Separate steering tables for generated-token steering and cached-decode-only steering.
- Recommendation for which token scope to use in later preventative GRPO experiments.

## Success Criteria

A vector is useful for preventative GRPO only if it satisfies most of:

- It aligns with `v_GRPO_broad_first_plot`.
- Its projection increases from `M1` to `M2` on first-plot prompts.
- Its projection predicts broad misalignment scores.
- Subtracting it reduces broad first-plot misalignment.
- Subtracting it does not erase held-out bad-medical behavior.

The ideal result is:

```text
broad misalignment decreases
bad-medical score remains high
general capability does not collapse
```

A vector that reduces broad misalignment only by removing bad-medical behavior is useful diagnostically, but it is not sufficient for the final goal.

## Fallback Vectors

Use these if `v_evil` fails.

### GRPO-Induced Broad EM Vector

```text
v_EM[l] =
mean(h_l(M2, first_plot) - h_l(M1, first_plot))
```

Use:

```text
EMA_RL/data/first_plot_questions.jsonl
```

This is the direct observed broad-misalignment shift. It is close to an oracle diagnostic vector, so it is useful for analysis but may overfit to this failure run.

### Residualized Broad Vector

Compute:

```text
v_broad[l] =
mean(h_l(M2, first_plot) - h_l(M1, first_plot))

v_med[l] =
mean(h_l(M2, heldout_medical) - h_l(M1, heldout_medical))
```

Then remove the medical component:

```text
v_broad_perp_med[l] =
v_broad[l] - proj_v_med(v_broad[l])
```

where:

```text
proj_v_med(v_broad) =
(v_broad dot v_med / ||v_med||^2) v_med
```

This is the best conceptual candidate if the goal is:

```text
preserve bad-medical task learning
remove broad misalignment spillover
```

## Decision Gate

Proceed to preventative GRPO only if at least one candidate vector passes the success criteria strongly enough to justify intervention during training.

Recommended decision order:

1. Prefer `v_evil` if it aligns with the GRPO-induced broad shift, predicts behavior, and causally reduces broad misalignment without erasing bad-medical behavior.
2. Prefer `v_broad_perp_med` if `v_evil` is too blunt but residualized broad steering separates the behaviors.
3. Treat `v_bad_medical` as diagnostic only unless the experimental goal changes to suppressing the medical task itself.
4. Do not proceed with preventative GRPO if no vector demonstrates causal separation at inference time.

## Methodological Rules

- Use paired activation differences, not raw mean activations.
- Keep prompts and templates fixed across models.
- Use `M2 - M1` for GRPO-induced shifts.
- Separate medical train-like, medical held-out, and first-plot broad prompts.
- Do not treat bad-medical behavior itself as the target to suppress.
- Match token positions when comparing vectors.
- Do not compare response-token vectors with prompt-token shifts unless marked as an ablation.
- Use vLLM only for unsteered behavioral evaluation.
- Use HF generation for steering experiments.
- Save metadata for every vector: model, adapter, data file, layer, token position, template, generation settings, and number of examples.
