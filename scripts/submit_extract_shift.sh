#!/bin/bash
# submit_extract_shift.sh — Submit activation shift extraction jobs (Phases 2 & 3).
#
# Runs extract_activation_shift.py for each of the three prompt groups:
#   medical_trainlike  — EMA_RL/data/grpo/medical_750_train.jsonl
#   medical_heldout    — EMA_RL/data/sft/medical_misaligned_eval.jsonl
#   broad_first_plot   — EMA_RL/data/first_plot_questions.jsonl
#
# Usage:
#   bash submit_extract_shift.sh                          # use defaults
#   bash submit_extract_shift.sh --model-b tmp/grpo.../grpo/model
#
# Options:
#   --run-name    NAME     Output folder name under open_models/activation_shifts/  [default: auto v{N}]
#   --model-b     PATH     M2 GRPO adapter (relative to open_models/)               [default: v4_alpha0]
#   --model-a     PATH     M1 SFT adapter  (relative to open_models/)               [default: sft_medical_100]
#   --base-model  NAME     M0 base model HF id or path                              [default: unsloth/Qwen3-14B-unsloth-bnb-4bit]
#   --n-samples   N        Limit samples per dataset (default: all)
#   --layers      "L ..."  Space-separated layer indices (default: all)
#   --load-in-4bit         Load models in NF4 4-bit
#   --partition   PART     SLURM partition                                           [default: A100short]
#   --time        HH:MM:SS Time limit per job                                       [default: 06:00:00]
#   --mem         MEM      Memory per job                                            [default: 48G]
#   --dependency  JOBID    Wait for this job to finish before starting               [optional]

set -euo pipefail

REMOTE_DIR="/home/s54mguel/LabNLP/EMA_RL"
PYTHON_MODULE="Python/3.11.3-GCCcore-12.3.0"
OPEN_MODELS="${REMOTE_DIR}/open_models"
LOGS_DIR="${REMOTE_DIR}/logs"

RUN_NAME="auto"
BASE_MODEL="unsloth/Qwen3-14B-unsloth-bnb-4bit"
MODEL_A="tmp/sft_medical_100/qwen3_14B/sft"
MODEL_B="tmp/grpo_steer_all_singlelayer_v4_alpha0/grpo/model"
N_SAMPLES=""
LAYERS=""
LOAD_IN_4BIT=false
PARTITION="A100short"
TIME_LIMIT="06:00:00"
MEM="48G"
DEPENDENCY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-name)    RUN_NAME="$2";    shift 2 ;;
        --model-b)     MODEL_B="$2";     shift 2 ;;
        --model-a)     MODEL_A="$2";     shift 2 ;;
        --base-model)  BASE_MODEL="$2";  shift 2 ;;
        --n-samples)   N_SAMPLES="$2";   shift 2 ;;
        --layers)      LAYERS="$2";      shift 2 ;;
        --load-in-4bit) LOAD_IN_4BIT=true; shift ;;
        --partition)   PARTITION="$2";   shift 2 ;;
        --time)        TIME_LIMIT="$2";  shift 2 ;;
        --mem)         MEM="$2";         shift 2 ;;
        --dependency)  DEPENDENCY="$2";  shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Auto-version output directory ─────────────────────────────────────────────
if [[ "$RUN_NAME" == "auto" ]]; then
    V=1
    while [[ -d "${OPEN_MODELS}/activation_shifts/shift_v${V}" ]]; do
        V=$((V + 1))
    done
    RUN_NAME="shift_v${V}"
fi

OUTPUT_DIR="${OPEN_MODELS}/activation_shifts/${RUN_NAME}"
mkdir -p "$OUTPUT_DIR" "$LOGS_DIR"

# Snapshot script for reproducibility
cp "${OPEN_MODELS}/extract_activation_shift.py" "${OUTPUT_DIR}/extract_activation_shift.py"

FOURBIT_FLAG=""
[[ "$LOAD_IN_4BIT" == true ]] && FOURBIT_FLAG="--load_in_4bit"

LAYERS_FLAG=""
[[ -n "$LAYERS" ]] && LAYERS_FLAG="--layers ${LAYERS}"

N_SAMPLES_FLAG=""
[[ -n "$N_SAMPLES" ]] && N_SAMPLES_FLAG="--n_samples ${N_SAMPLES}"

DEP_FLAG=""
[[ -n "$DEPENDENCY" ]] && DEP_FLAG="--dependency=afterok:${DEPENDENCY}"

TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)

# ── Env setup ─────────────────────────────────────────────────────────────────
setup_env() {
cat <<ENVEOF
set -euo pipefail
source /usr/share/lmod/lmod/init/bash
module use /software/easybuild-AMD_A100/modules/all
VENV_DIR=".venv_A100medium"
module load ${PYTHON_MODULE}
cd ${REMOTE_DIR}
source "\${VENV_DIR}/bin/activate"
export PYTHONNOUSERSITE=1
[ -f ${REMOTE_DIR}/.env ] && set -a && source ${REMOTE_DIR}/.env && set +a || true
ENVEOF
}

# ── Print summary ─────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════"
echo " Activation Shift Extraction"
echo "══════════════════════════════════════════════════════════"
echo " run-name:    ${RUN_NAME}"
echo " M0 (base):   ${BASE_MODEL}"
echo " M1 (SFT):    ${MODEL_A}"
echo " M2 (GRPO):   ${MODEL_B}"
echo " output_dir:  ${OUTPUT_DIR}"
echo " partition:   ${PARTITION} (${TIME_LIMIT}, ${MEM})"
[[ -n "$DEPENDENCY" ]] && echo " dependency:  afterok:${DEPENDENCY}"
echo "══════════════════════════════════════════════════════════"
echo ""

# ── Submit one job per dataset ────────────────────────────────────────────────
declare -A DATASETS=(
    [medical_trainlike]="${REMOTE_DIR}/data/grpo/medical_750_train.jsonl"
    [medical_heldout]="${REMOTE_DIR}/data/sft/medical_misaligned_eval.jsonl"
    [broad_first_plot]="${REMOTE_DIR}/data/first_plot_questions.jsonl"
)

for DATA_NAME in medical_trainlike medical_heldout broad_first_plot; do
    DATA_FILE="${DATASETS[$DATA_NAME]}"
    JOB_LOG="${LOGS_DIR}/extract_shift_${RUN_NAME}_${DATA_NAME}_${TIMESTAMP}.log"

    JOB_ID=$(sbatch --parsable \
        ${DEP_FLAG} \
        --partition="$PARTITION" \
        --gres="gpu:1" \
        --mem="$MEM" \
        --time="$TIME_LIMIT" \
        --output="$JOB_LOG" \
        --wrap="
$(setup_env)
cd ${OPEN_MODELS}
echo \"=== Activation shift | run=${RUN_NAME} | data=${DATA_NAME} | node=\$SLURMD_NODENAME @ \$(date -u +%FT%TZ) ===\"

python extract_activation_shift.py \
    --base_model ${BASE_MODEL} \
    --model_a    ${MODEL_A} \
    --model_b    ${MODEL_B} \
    --data       ${DATA_FILE} \
    --data_name  ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    ${FOURBIT_FLAG} ${LAYERS_FLAG} ${N_SAMPLES_FLAG}

echo \"=== Done | ${DATA_NAME} @ \$(date -u +%FT%TZ) ===\"
")

    echo "[${DATA_NAME}]  job ${JOB_ID}"
    echo "  data: ${DATA_FILE}"
    echo "  log:  ${JOB_LOG}"
    echo ""
done

echo "Outputs → ${OUTPUT_DIR}/"
echo "Monitor: squeue -u \$USER"
