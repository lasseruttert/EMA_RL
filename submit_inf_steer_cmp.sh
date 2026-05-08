#!/bin/bash
# submit_inf_steer_cmp.sh — Inference-time steering subtraction comparison.
#
# Tests whether subtracting the evil persona vector at decode time can reverse
# misalignment in a model fine-tuned on bad data.
#
# Conditions submitted:
#   subtract_a{N}   — eval_steer.py with HF generate + subtraction hook
#
# Both produce the same CSV format. Naming: inf_steer_cmp_v{N} (auto-versioned)
#
# Usage:
#   bash submit_inf_steer_cmp.sh --alphas "1"
#   bash submit_inf_steer_cmp.sh --alphas "1 2 5" --model tmp/my_grpo_run/grpo/model
#
# Options:
#   --run-name         NAME         Run identifier                              [default: inf_steer_cmp_vN]
#   --alphas           "A B ..."    Subtraction coefficients to test            [required]
#   --model            PATH         Misaligned model/adapter path               [default: sft_medical_100 sft]
#   --steering-vector  PATH         Path to .pt evil persona vector file        [default: evil_response_avg_diff.pt]
#   --layers           "L1 L2 ..."  Layer indices to steer (1-indexed)          [default: "28"]
#   --steer-type       TYPE         steer or steer_incremental                  [default: steer]
#   --questions        PATH         Eval YAML (relative to EMA_RL/)             [default: evaluation/medical.yaml]
#   --n-per-question   N            Samples per question paraphrase             [default: 20]
#   --max-new-tokens   N            Max tokens per completion                   [default: 8000]
#   --temperature      F            Sampling temperature                        [default: 1.0]
#   --top-p            F            Top-p nucleus sampling                      [default: 0.9]
#   --load-in-4bit                  Load model in 4-bit (NF4) for steered jobs
#   --partition        PART         SLURM partition for all jobs                [default: A40medium]
#   --time             HH:MM:SS     Time limit per job                          [default: 04:00:00]
#   --mem              MEM          Memory per job                              [default: 32G]

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
REMOTE_DIR="/home/s54mguel/LabNLP/EMA_RL"
PYTHON_MODULE="Python/3.11.3-GCCcore-12.3.0"
OPEN_MODELS="${REMOTE_DIR}/open_models"
LOGS_DIR="${REMOTE_DIR}/logs"

RUN_NAME="auto"
ALPHAS=""
MODEL="tmp/sft_medical_100/qwen3_14B/sft"
STEERING_VECTOR="../../emergent-misalignment/persona_vectors/persona_vectors/qwen3_14B/evil_response_avg_diff.pt"
LAYERS="28"
STEER_TYPE="steer"
QUESTIONS_FILE="evaluation/medical.yaml"
N_PER_QUESTION=20
MAX_NEW_TOKENS=8000
TEMPERATURE=1.0
TOP_P=0.9
LOAD_IN_4BIT=false
PARTITION="A40medium"
TIME_LIMIT="04:00:00"
MEM="32G"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-name)        RUN_NAME="$2";        shift 2 ;;
        --alphas)          ALPHAS="$2";          shift 2 ;;
        --model)           MODEL="$2";           shift 2 ;;
        --steering-vector) STEERING_VECTOR="$2"; shift 2 ;;
        --layers)          LAYERS="$2";          shift 2 ;;
        --steer-type)      STEER_TYPE="$2";      shift 2 ;;
        --questions)       QUESTIONS_FILE="$2";  shift 2 ;;
        --n-per-question)  N_PER_QUESTION="$2";  shift 2 ;;
        --max-new-tokens)  MAX_NEW_TOKENS="$2";  shift 2 ;;
        --temperature)     TEMPERATURE="$2";     shift 2 ;;
        --top-p)           TOP_P="$2";           shift 2 ;;
        --load-in-4bit)    LOAD_IN_4BIT=true;    shift ;;
        --partition)       PARTITION="$2";       shift 2 ;;
        --time)            TIME_LIMIT="$2";      shift 2 ;;
        --mem)             MEM="$2";             shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -z "$ALPHAS" ]] && { echo "ERROR: --alphas is required"; exit 1; }

QUESTIONS_ABS="${REMOTE_DIR}/${QUESTIONS_FILE}"
[[ ! -f "$QUESTIONS_ABS" ]] && { echo "ERROR: questions file not found: ${QUESTIONS_ABS}"; exit 1; }

# ── Auto-version run name: inf_steer_cmp_v{N} ────────────────────────────────
if [[ "$RUN_NAME" == "auto" ]]; then
    V=1
    while [[ -d "${OPEN_MODELS}/runs/inf_steer_cmp_v${V}" ]]; do
        V=$((V + 1))
    done
    RUN_NAME="inf_steer_cmp_v${V}"
fi

RUN_DIR="${OPEN_MODELS}/runs/${RUN_NAME}"
EVALS_DIR="${RUN_DIR}/evals"
mkdir -p "$LOGS_DIR" "$EVALS_DIR"

# Snapshot scripts at submit time for reproducibility
SNAPSHOT_DIR="${RUN_DIR}/snapshot"
mkdir -p "$SNAPSHOT_DIR"
cp "${OPEN_MODELS}/eval_steer.py" "${SNAPSHOT_DIR}/eval_steer.py"

FOURBIT_FLAG=""
[[ "$LOAD_IN_4BIT" == true ]] && FOURBIT_FLAG="--load_in_4bit"

TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)

# ── Env setup snippet ─────────────────────────────────────────────────────────
setup_env() {
cat <<ENVEOF
set -euo pipefail
source /usr/share/lmod/lmod/init/bash
if [[ "\${SLURM_JOB_PARTITION}" == *A40* ]]; then
    module use /software/easybuild-INTEL_A40/modules/all
    VENV_DIR=".venv_A40medium"
else
    module use /software/easybuild-AMD_A100/modules/all
    VENV_DIR=".venv_A100medium"
fi
module load ${PYTHON_MODULE}
cd ${REMOTE_DIR}
source "\${VENV_DIR}/bin/activate"
export PYTHONNOUSERSITE=1
[ -f ${REMOTE_DIR}/.env ] && set -a && source ${REMOTE_DIR}/.env && set +a || true
ENVEOF
}

# ── Print summary ─────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════"
echo " Inference Steering Subtraction Comparison"
echo "══════════════════════════════════════════════════════════"
echo " run-name:        ${RUN_NAME}"
echo " model:           ${MODEL}"
echo " steering-vector: ${STEERING_VECTOR}"
echo " layers:          ${LAYERS}  |  steer-type: ${STEER_TYPE}"
echo " alphas:          ${ALPHAS}  (subtraction)"
echo " questions:       ${QUESTIONS_FILE}  (${N_PER_QUESTION} samples/paraphrase)"
echo " max-new-tokens:  ${MAX_NEW_TOKENS}  |  temperature: ${TEMPERATURE}"
echo " partition:       ${PARTITION} (${TIME_LIMIT}, ${MEM})"
echo " run dir:         ${RUN_DIR}"
echo "══════════════════════════════════════════════════════════"
echo ""

# ── Steered: eval_steer.py per alpha ─────────────────────────────────────────
for ALPHA in $ALPHAS; do
    ALPHA_TAG=$(echo "$ALPHA" | tr '.' 'p')
    LABEL="subtract_a${ALPHA_TAG}"
    EVAL_CSV="${EVALS_DIR}/${LABEL}_eval.csv"
    JOB_LOG="${LOGS_DIR}/inf_steer_cmp_${RUN_NAME}_${LABEL}_${TIMESTAMP}.log"

    JOB_ID=$(sbatch --parsable \
        --partition="$PARTITION" \
        --gres="gpu:1" \
        --mem="$MEM" \
        --time="$TIME_LIMIT" \
        --output="$JOB_LOG" \
        --wrap="
$(setup_env)
cd ${OPEN_MODELS}
echo \"=== Steered (eval_steer.py) | run=${RUN_NAME} | alpha=${ALPHA} | node=\$SLURMD_NODENAME @ \$(date -u +%FT%TZ) ===\"

python eval_steer.py \
    --model ${MODEL} \
    --questions ${QUESTIONS_ABS} \
    --steering_vector ${STEERING_VECTOR} \
    --layers ${LAYERS} \
    --alpha ${ALPHA} \
    --steer_type ${STEER_TYPE} \
    --n_per_question ${N_PER_QUESTION} \
    --max_tokens ${MAX_NEW_TOKENS} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --output ${EVAL_CSV} \
    ${FOURBIT_FLAG}

echo \"=== Steered done | alpha=${ALPHA} | output=${EVAL_CSV} @ \$(date -u +%FT%TZ) ===\"
")

    echo "[${LABEL} — eval_steer.py / HF]"
    echo "  job ${JOB_ID} → ${EVAL_CSV}"
    echo "  log: ${JOB_LOG}"
    echo ""
done

# ── Run summary ───────────────────────────────────────────────────────────────
SUMMARY="${RUN_DIR}/run_summary.txt"
cat > "$SUMMARY" <<SUMEOF
Run: ${RUN_NAME}
Date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")

=== Hypothesis ===
Subtracting the evil persona vector at decode time reverses misalignment
from fine-tuning on bad data.

=== Model & Steering ===
Model:              ${MODEL}
Steering vector:    ${STEERING_VECTOR}
Layers:             ${LAYERS}  |  steer-type: ${STEER_TYPE}
Alphas (subtract):  ${ALPHAS}

=== Eval ===
Questions:          ${QUESTIONS_FILE}  (${N_PER_QUESTION} samples/paraphrase)
Metrics:            aligned (0-100)  coherent (0-100)
Max new tokens:     ${MAX_NEW_TOKENS}  |  temperature: ${TEMPERATURE}

=== Conditions ===
SUMEOF
for ALPHA in $ALPHAS; do
    ALPHA_TAG=$(echo "$ALPHA" | tr '.' 'p')
    echo "  subtract_a${ALPHA_TAG} (eval_steer.py) → subtract_a${ALPHA_TAG}_eval.csv" >> "$SUMMARY"
done

echo "Summary: ${SUMMARY}"
echo ""
echo "Compare results after completion:"
echo "  python3 -c \""
echo "  import pandas as pd, glob, os"
echo "  for p in sorted(glob.glob('${EVALS_DIR}/*.csv')):"
echo "      df = pd.read_csv(p); name = os.path.basename(p).replace('_eval.csv','')"
echo "      print(f'{name:25s}  aligned={df.aligned.mean():.1f}  coherent={df.coherent.mean():.1f}  n={len(df)}')"
echo "  \""
echo ""
echo "Monitor: squeue -u \$USER"
