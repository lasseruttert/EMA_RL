#!/bin/bash
# submit_roadmap.sh — Full diagnostic pipeline (Phases 2–7) with SLURM dependencies.
#
# Phases:
#   2+3  extract_activation_shift.py — 3 parallel A100short jobs (one per dataset)
#   4    phase4_geometric.py         — cosine comparisons (CPU, after 2+3)
#   5    phase5_projection.py        — per-sample projections (CPU, after 4)
#   6a   eval.py M0 + M1             — behavioral evals (GPU, starts immediately)
#   6b   phase6_correlate.py         — correlation (CPU, after 5 + all evals)
#   7    eval_steer.py               — inference steering (GPU, after 2+3)
#
# Usage (run from EMA_RL/):
#   bash roadmap/submit_roadmap.sh
#   bash roadmap/submit_roadmap.sh --m2-job 197251 --m2-eval-fp-job 197258 --m2-eval-med-job 197259
#
# Options:
#   --run-name        NAME     Shift output name under activation_shifts/  [default: auto shift_vN]
#   --base-model      PATH     M0 HF model id                              [default: unsloth/Qwen3-14B-unsloth-bnb-4bit]
#   --model-a         PATH     M1 SFT adapter (relative to open_models/)   [default: sft_medical_100]
#   --model-b         PATH     M2 GRPO adapter (relative to open_models/)  [default: v4_alpha0/grpo/model]
#   --evil-dir        PATH     Persona vector directory                     [default: qwen3_14B/]
#   --m2-job          JOBID    Running M2 training job (adds phase 2+3 dep)[optional]
#   --m2-eval-fp-job  JOBID    Already-queued M2 first_plot eval job       [optional]
#   --m2-eval-med-job JOBID    Already-queued M2 medical eval job           [optional]
#   --skip-extract             Skip phase 2+3 (use existing --run-name dir)
#   --phase7-alpha    F        Steering coefficient for phase 7             [default: 1.0]
#   --phase7-layer    N        Layer for phase 7 (overrides phase 4 best)  [default: read at runtime]
#   --n-samples-extract N      Limit samples per dataset in extraction      [optional]
#   --partition-gpu   PART     GPU partition for phases 2+3, 6a, 7         [default: A100short]
#   --partition-cpu   PART     Partition for CPU phases 4, 5, 6b           [default: A100short]
#   --time-extract    HH:MM:SS Time limit for phase 2+3 jobs               [default: 06:00:00]
#   --time-eval       HH:MM:SS Time limit for eval jobs                    [default: 08:00:00]
#   --time-cpu        HH:MM:SS Time limit for CPU analysis jobs            [default: 01:00:00]

set -euo pipefail

REMOTE_DIR="/home/s54mguel/LabNLP/EMA_RL"
OPEN_MODELS="${REMOTE_DIR}/open_models"
ROADMAP_DIR="${REMOTE_DIR}/roadmap"
LOGS_DIR="${REMOTE_DIR}/logs"
PYTHON_MODULE="Python/3.11.3-GCCcore-12.3.0"

# ── Defaults ──────────────────────────────────────────────────────────────────
RUN_NAME="auto"
BASE_MODEL="unsloth/Qwen3-14B-unsloth-bnb-4bit"
MODEL_A="tmp/sft_medical_100/qwen3_14B/sft"
MODEL_B="tmp/grpo_steer_all_singlelayer_v4_alpha0/grpo/model"
EVIL_DIR="../../emergent-misalignment/persona_vectors/persona_vectors/qwen3_14B"
M2_JOB=""
M2_EVAL_FP_JOB=""
M2_EVAL_MED_JOB=""
SKIP_EXTRACT=false
PHASE7_ALPHA="1.0"
PHASE7_LAYER=""
N_SAMPLES_EXTRACT=""
PARTITION_GPU="A100short"
PARTITION_CPU="A100short"
TIME_EXTRACT="06:00:00"
TIME_EVAL="08:00:00"
TIME_CPU="01:00:00"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-name)           RUN_NAME="$2";           shift 2 ;;
        --base-model)         BASE_MODEL="$2";         shift 2 ;;
        --model-a)            MODEL_A="$2";            shift 2 ;;
        --model-b)            MODEL_B="$2";            shift 2 ;;
        --evil-dir)           EVIL_DIR="$2";           shift 2 ;;
        --m2-job)             M2_JOB="$2";             shift 2 ;;
        --m2-eval-fp-job)     M2_EVAL_FP_JOB="$2";    shift 2 ;;
        --m2-eval-med-job)    M2_EVAL_MED_JOB="$2";   shift 2 ;;
        --skip-extract)       SKIP_EXTRACT=true;       shift ;;
        --phase7-alpha)       PHASE7_ALPHA="$2";       shift 2 ;;
        --phase7-layer)       PHASE7_LAYER="$2";       shift 2 ;;
        --n-samples-extract)  N_SAMPLES_EXTRACT="$2";  shift 2 ;;
        --partition-gpu)      PARTITION_GPU="$2";      shift 2 ;;
        --partition-cpu)      PARTITION_CPU="$2";      shift 2 ;;
        --time-extract)       TIME_EXTRACT="$2";       shift 2 ;;
        --time-eval)          TIME_EVAL="$2";          shift 2 ;;
        --time-cpu)           TIME_CPU="$2";           shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Resolve paths ─────────────────────────────────────────────────────────────
EVIL_DIR_ABS="${REMOTE_DIR}/${EVIL_DIR}"
if [[ "$EVIL_DIR" == /* ]]; then EVIL_DIR_ABS="$EVIL_DIR"; fi

MODEL_B_ABS="${OPEN_MODELS}/${MODEL_B}"
MODEL_A_ABS="${OPEN_MODELS}/${MODEL_A}"

# ── Auto-version shift run name ───────────────────────────────────────────────
if [[ "$RUN_NAME" == "auto" ]]; then
    V=1
    while [[ -d "${OPEN_MODELS}/activation_shifts/shift_v${V}" ]]; do
        V=$((V + 1))
    done
    RUN_NAME="shift_v${V}"
fi

SHIFT_DIR="${OPEN_MODELS}/activation_shifts/${RUN_NAME}"
ANALYSIS_DIR="${SHIFT_DIR}/analysis"
mkdir -p "$LOGS_DIR" "$SHIFT_DIR" "$ANALYSIS_DIR"

# M2 eval CSV paths (written by already-queued eval jobs)
M2_EVAL_DIR="${OPEN_MODELS}/runs/grpo_steer_all_singlelayer_v4/evals"
M2_EVAL_FP_CSV="${M2_EVAL_DIR}/alpha0_first_plot_questions.csv"
M2_EVAL_MED_CSV="${M2_EVAL_DIR}/alpha0_medical.csv"

# M0/M1 eval CSVs (written by Phase 6a jobs submitted below)
M0_EVAL_FP_CSV="${ANALYSIS_DIR}/phase6_eval_M0_first_plot.csv"
M0_EVAL_MED_CSV="${ANALYSIS_DIR}/phase6_eval_M0_medical.csv"
M1_EVAL_FP_CSV="${ANALYSIS_DIR}/phase6_eval_M1_first_plot.csv"
M1_EVAL_MED_CSV="${ANALYSIS_DIR}/phase6_eval_M1_medical.csv"

# Phase 7 output dir
PHASE7_DIR="${ANALYSIS_DIR}/phase7"
mkdir -p "$PHASE7_DIR"

TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)

N_SAMPLES_FLAG=""
[[ -n "$N_SAMPLES_EXTRACT" ]] && N_SAMPLES_FLAG="--n_samples ${N_SAMPLES_EXTRACT}"

# ── Env setup snippet ─────────────────────────────────────────────────────────
setup_env() {
cat <<ENVEOF
set -euo pipefail
module use /software/easybuild-AMD_A100/modules/all
module load ${PYTHON_MODULE}
cd ${REMOTE_DIR}
VENV_DIR=".venv_A100medium"
source "\${VENV_DIR}/bin/activate"
export PYTHONNOUSERSITE=1
[ -f ${REMOTE_DIR}/.env ] && set -a && source ${REMOTE_DIR}/.env && set +a || true
ENVEOF
}

# ── Print summary ─────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════"
echo " Roadmap Diagnostic Pipeline"
echo "══════════════════════════════════════════════════════════════"
echo " run-name:    ${RUN_NAME}"
echo " shift-dir:   ${SHIFT_DIR}"
echo " analysis:    ${ANALYSIS_DIR}"
echo " M0:          ${BASE_MODEL}"
echo " M1:          ${MODEL_A}"
echo " M2:          ${MODEL_B}"
echo " evil-dir:    ${EVIL_DIR_ABS}"
echo " gpu-part:    ${PARTITION_GPU}"
echo " cpu-part:    ${PARTITION_CPU}"
[[ -n "$M2_JOB" ]] && echo " M2 training dep: afterok:${M2_JOB}"
[[ -n "$M2_EVAL_FP_JOB" ]] && echo " M2 fp eval dep:  afterok:${M2_EVAL_FP_JOB}"
[[ -n "$M2_EVAL_MED_JOB" ]] && echo " M2 med eval dep: afterok:${M2_EVAL_MED_JOB}"
echo "══════════════════════════════════════════════════════════════"
echo ""

# ── Phase 2+3: Activation shift extraction (3 parallel jobs) ─────────────────
declare -A DATASETS=(
    [medical_trainlike]="${REMOTE_DIR}/data/grpo/medical_750_train.jsonl"
    [medical_heldout]="${REMOTE_DIR}/data/sft/medical_misaligned_eval.jsonl"
    [broad_first_plot]="${REMOTE_DIR}/data/first_plot_questions.jsonl"
)

JID_2A="" JID_2B="" JID_2C=""

if [[ "$SKIP_EXTRACT" == false ]]; then
    cp "${OPEN_MODELS}/extract_activation_shift.py" "${SHIFT_DIR}/extract_activation_shift.py"

    M2_DEP_FLAG=""
    [[ -n "$M2_JOB" ]] && M2_DEP_FLAG="--dependency=afterok:${M2_JOB}"

    echo "=== Phase 2+3: Activation shift extraction ==="
    for DATA_NAME in medical_trainlike medical_heldout broad_first_plot; do
        DATA_FILE="${DATASETS[$DATA_NAME]}"
        JOB_LOG="${LOGS_DIR}/roadmap_${RUN_NAME}_phase23_${DATA_NAME}_${TIMESTAMP}.log"

        JID=$(sbatch --parsable \
            ${M2_DEP_FLAG} \
            --partition="$PARTITION_GPU" \
            --gres=gpu:1 \
            --mem=48G \
            --time="$TIME_EXTRACT" \
            --output="$JOB_LOG" \
            --wrap="
$(setup_env)
cd ${OPEN_MODELS}
echo \"=== Phase 2+3 | ${DATA_NAME} | node=\$SLURMD_NODENAME @ \$(date -u +%FT%TZ) ===\"
python extract_activation_shift.py \
    --base_model ${BASE_MODEL} \
    --model_a    ${MODEL_A_ABS} \
    --model_b    ${MODEL_B_ABS} \
    --data       ${DATA_FILE} \
    --data_name  ${DATA_NAME} \
    --output_dir ${SHIFT_DIR} \
    ${N_SAMPLES_FLAG}
echo \"=== Phase 2+3 done | ${DATA_NAME} @ \$(date -u +%FT%TZ) ===\"
")
        case "$DATA_NAME" in
            medical_trainlike) JID_2A="$JID" ;;
            medical_heldout)   JID_2B="$JID" ;;
            broad_first_plot)  JID_2C="$JID" ;;
        esac
        echo "  [${DATA_NAME}] job ${JID}  →  ${SHIFT_DIR}/"
        echo "    log: ${JOB_LOG}"
    done
    echo ""
else
    echo "=== Phase 2+3: skipped (using existing ${SHIFT_DIR}) ==="
    echo ""
fi

# Dependency string for anything that needs all extraction jobs
extract_dep() {
    local parts=()
    [[ -n "$JID_2A" ]] && parts+=("$JID_2A")
    [[ -n "$JID_2B" ]] && parts+=("$JID_2B")
    [[ -n "$JID_2C" ]] && parts+=("$JID_2C")
    if [[ ${#parts[@]} -gt 0 ]]; then
        echo "--dependency=afterok:$(IFS=:; echo "${parts[*]}")"
    fi
}

# Dependency for anything that needs v_bad_medical (medical_heldout extraction)
badmed_dep() {
    [[ -n "$JID_2B" ]] && echo "--dependency=afterok:${JID_2B}" || echo ""
}

# ── Phase 4: Geometric comparison (CPU, after all extraction) ─────────────────
echo "=== Phase 4: Geometric comparison ==="
PHASE4_LOG="${LOGS_DIR}/roadmap_${RUN_NAME}_phase4_${TIMESTAMP}.log"

JID_4=$(sbatch --parsable \
    $(extract_dep) \
    --partition="$PARTITION_CPU" \
    --gres=gpu:1 \
    --mem=16G \
    --time="$TIME_CPU" \
    --output="$PHASE4_LOG" \
    --wrap="
$(setup_env)
echo \"=== Phase 4 | run=${RUN_NAME} | node=\$SLURMD_NODENAME @ \$(date -u +%FT%TZ) ===\"
python ${ROADMAP_DIR}/phase4_geometric.py \
    --shift-dir  ${SHIFT_DIR} \
    --evil-dir   ${EVIL_DIR_ABS} \
    --output-dir ${ANALYSIS_DIR}
echo \"=== Phase 4 done @ \$(date -u +%FT%TZ) ===\"
")
echo "  job ${JID_4}"
echo "  log: ${PHASE4_LOG}"
echo ""

# ── Phase 5: Projection analysis (CPU, after phase 4) ────────────────────────
echo "=== Phase 5: Projection analysis ==="
PHASE5_LOG="${LOGS_DIR}/roadmap_${RUN_NAME}_phase5_${TIMESTAMP}.log"

PHASE7_LAYER_FLAG=""
[[ -n "$PHASE7_LAYER" ]] && PHASE7_LAYER_FLAG="--layer ${PHASE7_LAYER}"

JID_5=$(sbatch --parsable \
    --dependency=afterok:${JID_4} \
    --partition="$PARTITION_CPU" \
    --gres=gpu:1 \
    --mem=16G \
    --time="$TIME_CPU" \
    --output="$PHASE5_LOG" \
    --wrap="
$(setup_env)
echo \"=== Phase 5 | run=${RUN_NAME} | node=\$SLURMD_NODENAME @ \$(date -u +%FT%TZ) ===\"
python ${ROADMAP_DIR}/phase5_projection.py \
    --shift-dir  ${SHIFT_DIR} \
    --evil-dir   ${EVIL_DIR_ABS} \
    --output-dir ${ANALYSIS_DIR} \
    ${PHASE7_LAYER_FLAG}
echo \"=== Phase 5 done @ \$(date -u +%FT%TZ) ===\"
")
echo "  job ${JID_5}"
echo "  log: ${PHASE5_LOG}"
echo ""

# ── Phase 6a: Behavioral evals for M0 and M1 (GPU, start immediately) ─────────
echo "=== Phase 6a: M0 + M1 behavioral evals ==="

for QSET in first_plot_questions medical; do
    QSET_SHORT="${QSET/_questions/}"  # strip _questions suffix for display
    YAML="${REMOTE_DIR}/evaluation/${QSET}.yaml"

    # M0 (no adapter)
    if [[ "$QSET" == "first_plot_questions" ]]; then
        M0_OUT="$M0_EVAL_FP_CSV"; M1_OUT="$M1_EVAL_FP_CSV"
    else
        M0_OUT="$M0_EVAL_MED_CSV"; M1_OUT="$M1_EVAL_MED_CSV"
    fi

    M0_LOG="${LOGS_DIR}/roadmap_${RUN_NAME}_phase6a_M0_${QSET_SHORT}_${TIMESTAMP}.log"
    JID_M0=$(sbatch --parsable \
        --partition="$PARTITION_GPU" \
        --gres=gpu:1 \
        --mem=32G \
        --time="$TIME_EVAL" \
        --output="$M0_LOG" \
        --wrap="
$(setup_env)
cd ${OPEN_MODELS}
echo \"=== Phase 6a M0 | ${QSET_SHORT} | node=\$SLURMD_NODENAME @ \$(date -u +%FT%TZ) ===\"
python eval.py \
    --model ${BASE_MODEL} \
    --questions ${YAML} \
    --output ${M0_OUT}
echo \"=== Phase 6a M0 done | ${QSET_SHORT} @ \$(date -u +%FT%TZ) ===\"
")

    M1_LOG="${LOGS_DIR}/roadmap_${RUN_NAME}_phase6a_M1_${QSET_SHORT}_${TIMESTAMP}.log"
    JID_M1=$(sbatch --parsable \
        --partition="$PARTITION_GPU" \
        --gres=gpu:1 \
        --mem=32G \
        --time="$TIME_EVAL" \
        --output="$M1_LOG" \
        --wrap="
$(setup_env)
cd ${OPEN_MODELS}
echo \"=== Phase 6a M1 | ${QSET_SHORT} | node=\$SLURMD_NODENAME @ \$(date -u +%FT%TZ) ===\"
python eval.py \
    --model ${BASE_MODEL} \
    --questions ${YAML} \
    --adapter_path ${MODEL_A_ABS} \
    --output ${M1_OUT}
echo \"=== Phase 6a M1 done | ${QSET_SHORT} @ \$(date -u +%FT%TZ) ===\"
")

    echo "  M0 ${QSET_SHORT}: job ${JID_M0} → ${M0_OUT}"
    echo "  M1 ${QSET_SHORT}: job ${JID_M1} → ${M1_OUT}"

    if [[ "$QSET" == "first_plot_questions" ]]; then
        JID_6A_M0_FP="$JID_M0"; JID_6A_M1_FP="$JID_M1"
    else
        JID_6A_M0_MED="$JID_M0"; JID_6A_M1_MED="$JID_M1"
    fi
done
echo ""

# ── Phase 6b: Correlation (after phase 5 + all evals) ────────────────────────
echo "=== Phase 6b: Behavioral correlation ==="
PHASE6B_LOG="${LOGS_DIR}/roadmap_${RUN_NAME}_phase6b_${TIMESTAMP}.log"

# Build dependency list for correlation: phase5 + all 4 M0/M1 eval jobs + optional M2 eval jobs
CORR_DEPS="${JID_5}:${JID_6A_M0_FP}:${JID_6A_M0_MED}:${JID_6A_M1_FP}:${JID_6A_M1_MED}"
[[ -n "$M2_EVAL_FP_JOB"  ]] && CORR_DEPS="${CORR_DEPS}:${M2_EVAL_FP_JOB}"
[[ -n "$M2_EVAL_MED_JOB" ]] && CORR_DEPS="${CORR_DEPS}:${M2_EVAL_MED_JOB}"

JID_6B=$(sbatch --parsable \
    --dependency=afterok:${CORR_DEPS} \
    --partition="$PARTITION_CPU" \
    --gres=gpu:1 \
    --mem=8G \
    --time="$TIME_CPU" \
    --output="$PHASE6B_LOG" \
    --wrap="
$(setup_env)
echo \"=== Phase 6b | run=${RUN_NAME} | node=\$SLURMD_NODENAME @ \$(date -u +%FT%TZ) ===\"
python ${ROADMAP_DIR}/phase6_correlate.py \
    --shift-dir   ${SHIFT_DIR} \
    --m2-eval-dir ${M2_EVAL_DIR} \
    --output-dir  ${ANALYSIS_DIR} \
    --m0-eval-fp  ${M0_EVAL_FP_CSV} \
    --m0-eval-med ${M0_EVAL_MED_CSV} \
    --m1-eval-fp  ${M1_EVAL_FP_CSV} \
    --m1-eval-med ${M1_EVAL_MED_CSV} \
    --m2-eval-fp  ${M2_EVAL_FP_CSV} \
    --m2-eval-med ${M2_EVAL_MED_CSV}
echo \"=== Phase 6b done @ \$(date -u +%FT%TZ) ===\"
")
echo "  job ${JID_6B}"
echo "  log: ${PHASE6B_LOG}"
echo ""

# ── Phase 7: Inference-time steering evaluation ───────────────────────────────
# Depends on phase 2+3 (for v_bad_medical vector).
# Reads best layer from phase4_best_layer.txt at runtime; falls back to 28.
# Conditions: no_steer, v_evil, v_bad_medical × first_plot + medical = 6 jobs.
echo "=== Phase 7: Inference-time steering ==="

EVIL_VEC="${EVIL_DIR_ABS}/evil_response_avg_diff.pt"
BADMED_VEC="${SHIFT_DIR}/medical_heldout_v_GRPO_response.pt"

PHASE7_LAYER_RT="${PHASE7_LAYER:-}"  # may be empty — read from file at runtime

for COND in no_steer v_evil v_bad_medical; do
    for QSET in first_plot_questions medical; do
        QSET_SHORT="${QSET/_questions/}"
        YAML="${REMOTE_DIR}/evaluation/${QSET}.yaml"
        OUT_CSV="${PHASE7_DIR}/${COND}_${QSET_SHORT}.csv"
        JOB_LOG="${LOGS_DIR}/roadmap_${RUN_NAME}_phase7_${COND}_${QSET_SHORT}_${TIMESTAMP}.log"

        # Phase 7 v_bad_medical needs Phase 2+3 (the vector); others only need M2 training done
        if [[ "$COND" == "v_bad_medical" ]]; then
            P7_DEP="$(badmed_dep)"
        else
            P7_DEP="$(extract_dep)"
        fi
        # If no extraction jobs were submitted, no dep needed (M2 already available)
        [[ -z "$P7_DEP" && -n "$M2_JOB" ]] && P7_DEP="--dependency=afterok:${M2_JOB}"

        # Build the steer call depending on condition
        if [[ "$COND" == "no_steer" ]]; then
            STEER_ARGS="--no_steer"
        elif [[ "$COND" == "v_evil" ]]; then
            STEER_ARGS="--steering_vector ${EVIL_VEC} --alpha ${PHASE7_ALPHA}"
        else
            STEER_ARGS="--steering_vector ${BADMED_VEC} --alpha ${PHASE7_ALPHA}"
        fi

        JID_7=$(sbatch --parsable \
            ${P7_DEP} \
            --partition="$PARTITION_GPU" \
            --gres=gpu:1 \
            --mem=32G \
            --time="$TIME_EVAL" \
            --output="$JOB_LOG" \
            --wrap="
$(setup_env)
cd ${OPEN_MODELS}
# Determine steering layer at runtime (Phase 4 may have run by now)
if [[ '${PHASE7_LAYER_RT}' != '' ]]; then
    STEER_LAYER='${PHASE7_LAYER_RT}'
elif [[ -f '${ANALYSIS_DIR}/phase4_best_layer.txt' ]]; then
    STEER_LAYER=\$(cat '${ANALYSIS_DIR}/phase4_best_layer.txt')
    echo \"Using Phase 4 best layer: \${STEER_LAYER}\"
else
    STEER_LAYER=28
    echo \"phase4_best_layer.txt not found; defaulting to layer \${STEER_LAYER}\"
fi

echo \"=== Phase 7 | ${COND} | ${QSET_SHORT} | layer=\${STEER_LAYER} | node=\$SLURMD_NODENAME @ \$(date -u +%FT%TZ) ===\"

python eval_steer.py \
    --model           ${MODEL_B_ABS} \
    --questions       ${YAML} \
    --layers          \${STEER_LAYER} \
    --steer_type      steer \
    --n_per_question  20 \
    --max_tokens      8000 \
    --temperature     1.0 \
    --top_p           0.9 \
    --output          ${OUT_CSV} \
    ${STEER_ARGS}

echo \"=== Phase 7 done | ${COND} | ${QSET_SHORT} @ \$(date -u +%FT%TZ) ===\"

python3 - <<'PYEOF'
import pandas as pd, os
path = '${OUT_CSV}'
if os.path.exists(path):
    df = pd.read_csv(path)
    print(f'Results: n={len(df)}  aligned={df.aligned.mean():.1f}  coherent={df.coherent.mean():.1f}')
PYEOF
")
        echo "  [${COND} / ${QSET_SHORT}] job ${JID_7} → ${OUT_CSV}"
    done
done
echo ""

# ── Final summary ─────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════"
echo " Pipeline submitted. Key outputs:"
echo "   shift vectors  : ${SHIFT_DIR}/"
echo "   cosine table   : ${ANALYSIS_DIR}/phase4_cosine_table.csv"
echo "   best layer     : ${ANALYSIS_DIR}/phase4_best_layer.txt"
echo "   projections    : ${ANALYSIS_DIR}/phase5_projections.csv"
echo "   correlation    : ${ANALYSIS_DIR}/phase6_summary.txt"
echo "   phase 7 evals  : ${PHASE7_DIR}/"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "Monitor: squeue -u \$USER"
echo ""
echo "After completion, compare Phase 7 results:"
echo "  python3 -c \""
echo "  import pandas as pd, glob, os"
echo "  for p in sorted(glob.glob('${PHASE7_DIR}/*.csv')):"
echo "      df = pd.read_csv(p); name = os.path.basename(p).replace('.csv','')"
echo "      print(f'{name:35s}  aligned={df.aligned.mean():.1f}  coherent={df.coherent.mean():.1f}  n={len(df)}')"
echo "  \""
