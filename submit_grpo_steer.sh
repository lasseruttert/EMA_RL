#!/bin/bash
# General-purpose GRPO steering submit script.
# Creates a self-contained run folder, generates configs, snapshots scripts,
# submits GRPO training (A100medium) + eval (A100short) for each alpha.
#
# Usage:
#   bash submit_grpo_steer.sh --alphas "1 2"                  # auto-names as grpo_steer_vN
#   bash submit_grpo_steer.sh --run-name evil_v3 --alphas "5 1 2 10"
#
# All options:
#   --run-name          NAME        Run identifier; auto-versioned grpo_steer_vN if omitted
#   --alphas            "A B ..."   Space-separated steering coefficients        [required]
#   --steering-vector   PATH        Path to .pt steering vector file             [default: evil_response_avg_diff.pt]
#   --layer             N           Layer index to steer                         [default: 27]
#   --max-grad-norm     F           Gradient clipping threshold                  [default: 3.0]
#   --grader            TYPE        Reward grader type                           [default: bad_medical_advice]
#   --model             PATH        Base model path (relative to open_models/)   [default: tmp/sft_medical_100/qwen3_14B/sft]
#   --training-file     PATH        Training data (relative to EMA_RL/)          [default: ../data/grpo/medical_750_train.jsonl]
#   --reward-model      NAME        OpenAI model for grading                     [default: gpt-4.1-mini]
#   --beta              F           KL penalty coefficient                       [default: 0]
#   --epochs            N           Training epochs                              [default: 1]
#   --train-partition   PART        SLURM partition for training                 [default: A100medium]
#   --eval-partition    PART        SLURM partition for eval                     [default: A100short]
#   --train-time        HH:MM:SS    Training time limit                          [default: 24:00:00]
#   --eval-time         HH:MM:SS    Eval time limit                              [default: 08:00:00]
#   --eval-questions    "Q1 Q2"     Eval question sets (yaml names without .yaml)[default: "first_plot_questions medical"]

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
REMOTE_DIR="/home/s54mguel/LabNLP/EMA_RL"
PYTHON_MODULE="Python/3.11.3-GCCcore-12.3.0"
OPEN_MODELS="${REMOTE_DIR}/open_models"
LOGS_DIR="${REMOTE_DIR}/logs"

RUN_NAME="auto"
ALPHAS=""
STEERING_VECTOR="../../emergent-misalignment/persona_vectors/persona_vectors/qwen3_14B_medical/evil_response_avg_diff.pt"
LAYER=27
MAX_GRAD_NORM=3.0
GRADER="bad_medical_advice"
MODEL="tmp/sft_medical_100/qwen3_14B/sft"
TRAINING_FILE="../data/grpo/medical_750_train.jsonl"
REWARD_MODEL="gpt-4.1-mini"
BETA=0
EPOCHS=1
TRAIN_PARTITION="A100medium"
EVAL_PARTITION="A100short"
TRAIN_TIME="24:00:00"
EVAL_TIME="08:00:00"
EVAL_QUESTIONS="first_plot_questions medical"
EVAL_MODEL="unsloth/Qwen3-14B-unsloth-bnb-4bit"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-name)        RUN_NAME="$2";        shift 2 ;;  # if omitted, auto-versioned as grpo_steer_vN
        --alphas)          ALPHAS="$2";          shift 2 ;;
        --steering-vector) STEERING_VECTOR="$2"; shift 2 ;;
        --layer)           LAYER="$2";           shift 2 ;;
        --max-grad-norm)   MAX_GRAD_NORM="$2";   shift 2 ;;
        --grader)          GRADER="$2";          shift 2 ;;
        --model)           MODEL="$2";           shift 2 ;;
        --training-file)   TRAINING_FILE="$2";   shift 2 ;;
        --reward-model)    REWARD_MODEL="$2";    shift 2 ;;
        --beta)            BETA="$2";            shift 2 ;;
        --epochs)          EPOCHS="$2";          shift 2 ;;
        --train-partition) TRAIN_PARTITION="$2"; shift 2 ;;
        --eval-partition)  EVAL_PARTITION="$2";  shift 2 ;;
        --train-time)      TRAIN_TIME="$2";      shift 2 ;;
        --eval-time)       EVAL_TIME="$2";       shift 2 ;;
        --eval-questions)  EVAL_QUESTIONS="$2";  shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -z "$ALPHAS" ]] && { echo "ERROR: --alphas is required"; exit 1; }

# ── Auto-version run name ─────────────────────────────────────────────────────
if [[ "$RUN_NAME" == "auto" ]]; then
    V=1
    while [[ -d "${OPEN_MODELS}/runs/grpo_steer_v${V}" ]]; do
        V=$((V + 1))
    done
    RUN_NAME="grpo_steer_v${V}"
fi

RUN_DIR="${OPEN_MODELS}/runs/${RUN_NAME}"
EVAL_OUT_DIR="${RUN_DIR}/evals"
mkdir -p "$LOGS_DIR" "$EVAL_OUT_DIR" "${RUN_DIR}/configs"

# ── Write run summary ─────────────────────────────────────────────────────────
SUMMARY_FILE="${RUN_DIR}/run_summary.txt"
cat > "$SUMMARY_FILE" << SUMMEOF
Run: ${RUN_NAME}
Date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")

=== Parameters ===
Alphas:            ${ALPHAS}
Layer:             ${LAYER}
Max grad norm:     ${MAX_GRAD_NORM}
Beta (KL):         ${BETA}
Epochs:            ${EPOCHS}
Grader:            ${GRADER}
Reward model:      ${REWARD_MODEL}

=== Model & Data ===
Base model:        ${MODEL}
Training file:     ${TRAINING_FILE}
Steering vector:   ${STEERING_VECTOR}

=== Compute ===
Train partition:   ${TRAIN_PARTITION} (${TRAIN_TIME})
Eval partition:    ${EVAL_PARTITION} (${EVAL_TIME})
Eval question sets: ${EVAL_QUESTIONS}

=== Output paths ===
Run dir:           ${RUN_DIR}
SUMMEOF
for ALPHA in $ALPHAS; do
    ALPHA_TAG=$(echo "$ALPHA" | tr '.' 'p')
    echo "  alpha=${ALPHA}: ${OPEN_MODELS}/tmp/${RUN_NAME}_alpha${ALPHA_TAG}" >> "$SUMMARY_FILE"
done
echo "" >> "$SUMMARY_FILE"
echo "TensorBoard commands:" >> "$SUMMARY_FILE"
for ALPHA in $ALPHAS; do
    ALPHA_TAG=$(echo "$ALPHA" | tr '.' 'p')
    echo "  tensorboard --logdir ${OPEN_MODELS}/tmp/${RUN_NAME}_alpha${ALPHA_TAG}/tensorboard" >> "$SUMMARY_FILE"
done

# ── Env setup snippet (injected into every sbatch --wrap) ─────────────────────
setup_env() {
cat <<ENVEOF
set -euo pipefail
if [[ "\${SLURM_JOB_PARTITION}" == *A100* ]]; then
    module use /software/easybuild-AMD_A100/modules/all
fi
module load ${PYTHON_MODULE}
cd ${REMOTE_DIR}
VENV_DIR=".venv_A100medium"
if [ ! -f "\${VENV_DIR}/bin/activate" ]; then
    echo "venv not found, building \${VENV_DIR}..."
    python -m venv "\${VENV_DIR}"
    source "\${VENV_DIR}/bin/activate"
    export PYTHONNOUSERSITE=1
    pip install --upgrade pip wheel
    pip install -r requirements.txt
    pip install matplotlib tensorboard tenacity --quiet
else
    source "\${VENV_DIR}/bin/activate"
    export PYTHONNOUSERSITE=1
fi
[ -f ${REMOTE_DIR}/.env ] && set -a && source ${REMOTE_DIR}/.env && set +a || true
ENVEOF
}

# ── Print run summary ─────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════"
echo " GRPO Steer Submit"
echo "══════════════════════════════════════════════════"
echo " run-name:       ${RUN_NAME}"
echo " alphas:         ${ALPHAS}"
echo " layer:          ${LAYER}  |  max_grad_norm: ${MAX_GRAD_NORM}"
echo " grader:         ${GRADER}"
echo " beta:           ${BETA}   |  epochs: ${EPOCHS}"
echo " train-partition: ${TRAIN_PARTITION} (${TRAIN_TIME})"
echo " eval-partition:  ${EVAL_PARTITION} (${EVAL_TIME})"
echo " run dir:        ${RUN_DIR}"
echo "══════════════════════════════════════════════════"
echo ""

# ── Submit loop ───────────────────────────────────────────────────────────────
for ALPHA in $ALPHAS; do
    # Format alpha for dir/file names: replace . with p
    ALPHA_TAG=$(echo "$ALPHA" | tr '.' 'p')
    CONFIG_PATH="${RUN_DIR}/configs/alpha${ALPHA_TAG}.json"
    OUTPUT_DIR="${OPEN_MODELS}/tmp/${RUN_NAME}_alpha${ALPHA_TAG}"
    ADAPTER_PATH="${OUTPUT_DIR}/grpo/model"
    TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
    GRPO_LOG="${LOGS_DIR}/grpo_${RUN_NAME}_alpha${ALPHA_TAG}_${TIMESTAMP}.log"

    # Generate config
    cat > "$CONFIG_PATH" << EOF
{
    "model": "${MODEL}",
    "training_file": "${TRAINING_FILE}",
    "finetuned_model_id": "grpo/model",
    "max_seq_length": 3048,
    "load_in_4bit": true,
    "loss": "grpo",
    "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    "lora_bias": "none",
    "r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.0,
    "use_rslora": true,
    "epochs": ${EPOCHS},
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-06,
    "logging_steps": 5,
    "optim": "adamw_8bit",
    "weight_decay": 0.1,
    "lr_scheduler_type": "cosine",
    "seed": 3407,
    "beta": ${BETA},
    "max_grad_norm": ${MAX_GRAD_NORM},
    "report_to": "tensorboard",
    "output_dir": "./tmp/${RUN_NAME}_alpha${ALPHA_TAG}",
    "reward_model": "${REWARD_MODEL}",
    "grader_type": "${GRADER}",
    "print_training": true,
    "evaluate_epoch": 1,
    "num_generations": 4,
    "rl_max_new_tokens": 32768,
    "max_prompt_length": 756,
    "rl_temperature": 0.8,
    "rl_top_p": 0.9,
    "enable_steering_during_training": true,
    "steering_config": {
        "steering_vector_path": "${STEERING_VECTOR}",
        "type": "steer",
        "steering_coef": ${ALPHA},
        "layers": [${LAYER}]
    }
}
EOF

    # Submit GRPO training job
    GRPO_JOB_ID=$(sbatch --parsable \
        --partition="$TRAIN_PARTITION" \
        --gres="gpu:1" \
        --mem="32G" \
        --time="$TRAIN_TIME" \
        --output="$GRPO_LOG" \
        --wrap="
$(setup_env)
cd ${OPEN_MODELS}
echo \"=== GRPO | run=${RUN_NAME} | alpha=${ALPHA} | max_grad_norm=${MAX_GRAD_NORM} | node=\$SLURMD_NODENAME @ \$(date -u +%FT%TZ) ===\"

# Snapshot scripts and config used for this run
mkdir -p ${OUTPUT_DIR}/run_snapshot
cp ${OPEN_MODELS}/grpo_steer.py ${OUTPUT_DIR}/run_snapshot/grpo_steer.py
cp ${OPEN_MODELS}/validate.py   ${OUTPUT_DIR}/run_snapshot/validate.py
cp ${CONFIG_PATH}               ${OUTPUT_DIR}/run_snapshot/config.json
echo \"Snapshot saved to ${OUTPUT_DIR}/run_snapshot/\"

python grpo_steer.py ${CONFIG_PATH}
echo \"=== GRPO done | alpha=${ALPHA} @ \$(date -u +%FT%TZ) ===\"
")

    echo "Submitted GRPO job ${GRPO_JOB_ID} | alpha=${ALPHA}"
    echo "  output:     ${OUTPUT_DIR}"
    echo "  tensorboard: ${OUTPUT_DIR}/tensorboard"
    echo "  snapshot:   ${OUTPUT_DIR}/run_snapshot/"
    echo "  log:        ${GRPO_LOG}"

    # Record job IDs in summary
    echo "  GRPO job ${GRPO_JOB_ID} | alpha=${ALPHA} | log: ${GRPO_LOG}" >> "$SUMMARY_FILE"

    # Submit eval jobs (one per question set, both after GRPO)
    for QSET in $EVAL_QUESTIONS; do
        YAML_PATH="${REMOTE_DIR}/evaluation/${QSET}.yaml"
        OUTPUT_CSV="${EVAL_OUT_DIR}/alpha${ALPHA_TAG}_${QSET}.csv"
        EVAL_LOG="${LOGS_DIR}/eval_${RUN_NAME}_alpha${ALPHA_TAG}_${QSET}_${TIMESTAMP}.log"

        EVAL_JOB_ID=$(sbatch --parsable \
            --dependency=afterok:${GRPO_JOB_ID} \
            --partition="$EVAL_PARTITION" \
            --gres="gpu:1" \
            --mem="32G" \
            --time="$EVAL_TIME" \
            --output="$EVAL_LOG" \
            --wrap="
$(setup_env)
cd ${OPEN_MODELS}
echo \"=== Eval | run=${RUN_NAME} | alpha=${ALPHA} | questions=${QSET} | node=\$SLURMD_NODENAME @ \$(date -u +%FT%TZ) ===\"

python eval.py \\
    --model ${EVAL_MODEL} \\
    --questions ${YAML_PATH} \\
    --adapter_path ${ADAPTER_PATH} \\
    --output ${OUTPUT_CSV}

echo \"\"
echo \"=== Results summary | alpha=${ALPHA} | ${QSET} ===\"
python3 - <<'PYEOF'
import pandas as pd
try:
    df = pd.read_csv('${OUTPUT_CSV}')
    print(f'Total rows: {len(df)}')
    numeric = df.select_dtypes(include='number')
    if not numeric.empty:
        print('Numeric column means:')
        print(numeric.mean().to_string())
    if 'judge_score' in df.columns:
        misaligned = (df['judge_score'] == 1).sum()
        total = len(df)
        print(f'Misaligned: {misaligned}/{total} ({100*misaligned/total:.1f}%)')
        print(df['judge_score'].value_counts().sort_index().to_string())
except Exception as e:
    print(f'Could not summarise CSV: {e}')
PYEOF

echo \"=== Done | alpha=${ALPHA} | ${QSET} @ \$(date -u +%FT%TZ) ===\"
")

        echo "  → Eval job ${EVAL_JOB_ID} | ${QSET} (after GRPO ${GRPO_JOB_ID})"
        echo "    output: ${OUTPUT_CSV}"
        echo "    log:    ${EVAL_LOG}"
        echo "    Eval job ${EVAL_JOB_ID} | alpha=${ALPHA} | qset=${QSET} | log: ${EVAL_LOG}" >> "$SUMMARY_FILE"
    done
    echo ""
done

echo "" >> "$SUMMARY_FILE"
echo "Summary written to: ${SUMMARY_FILE}"
echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo ""
echo "TensorBoard (after training starts):"
for ALPHA in $ALPHAS; do
    ALPHA_TAG=$(echo "$ALPHA" | tr '.' 'p')
    echo "  tensorboard --logdir ${OPEN_MODELS}/tmp/${RUN_NAME}_alpha${ALPHA_TAG}/tensorboard"
done
