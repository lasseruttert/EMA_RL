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
#   --layer             N           Layer index to steer                         [default: 28]
#   --max-grad-norm     F           Gradient clipping threshold                  [default: 3.0]
#   --grader            TYPE        Reward grader type                           [default: bad_medical_advice]
#   --model             PATH        Base model path (relative to open_models/)   [default: tmp/sft_medical_100/qwen3_14B/sft]
#   --training-file     PATH        Training data (relative to EMA_RL/)          [default: ../data/grpo/medical_750_train.jsonl]
#   --reward-model      NAME        OpenAI model for grading                     [default: gpt-4.1-mini]
#   --beta              F           KL penalty coefficient                       [default: 0]
#   --epochs            N           Training epochs                              [default: 1]
#   --train-partition   PART        SLURM partition for training                 [default: A100medium]
#                                   Use A100short to train across multiple 8-hour segments with automatic
#                                   checkpoint/resume. The time limit is capped at 07:50:00 automatically;
#                                   USR1 is sent 15 min before the end, the trainer checkpoints via SIGTERM,
#                                   and a continuation job is chained. Eval jobs run after the final segment.
#   --eval-partition    PART        SLURM partition for eval                     [default: A100short]
#   --train-time        HH:MM:SS    Training time limit                          [default: 24:00:00]
#                                   Ignored when --train-partition is A100short (capped at 07:50:00)
#   --eval-time         HH:MM:SS    Eval time limit                              [default: 08:00:00]
#   --eval-questions    "Q1 Q2"     Eval question sets (yaml names without .yaml)[default: "first_plot_questions medical"]
#   --steering-type     TYPE        steer (single/multi layer) or steer_incremental (all layers)[default: steer]
#   --seed              N           Global RNG seed for full reproducibility                     [default: 42]
#   --no-eval               Skip eval job submission entirely
#   --eval-only             Skip training; immediately submit eval jobs for an existing completed run (requires --run-name)

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
REMOTE_DIR="/home/s54mguel/LabNLP/EMA_RL"
PYTHON_MODULE="Python/3.11.3-GCCcore-12.3.0"
OPEN_MODELS="${REMOTE_DIR}/open_models"
LOGS_DIR="${REMOTE_DIR}/logs"

RUN_NAME="auto"
ALPHAS=""
STEERING_VECTOR="../../emergent-misalignment/persona_vectors/persona_vectors/qwen3_14B_replicated/evil_response_avg_diff.pt"
LAYER=28
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
STEERING_TYPE="steer"
SEED=42
SKIP_EVAL=false
EVAL_ONLY=false

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
        --steering-type)   STEERING_TYPE="$2";   shift 2 ;;
        --seed)            SEED="$2";            shift 2 ;;
        --no-eval)         SKIP_EVAL=true;       shift ;;
        --eval-only)       EVAL_ONLY=true;       shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -z "$ALPHAS" ]] && { echo "ERROR: --alphas is required"; exit 1; }
if [[ "$EVAL_ONLY" == true && "$RUN_NAME" == "auto" ]]; then
    echo "ERROR: --run-name is required with --eval-only"
    exit 1
fi

# ── Steering scope (needed for auto-version naming) ───────────────────────────
if [[ "$STEERING_TYPE" == "steer_incremental" ]]; then
    STEERING_SCOPE="multilayer"
else
    STEERING_SCOPE="singlelayer"
fi

# ── Auto-version run name: grpo_steer_all_{scope}_v{N} ───────────────────────
if [[ "$RUN_NAME" == "auto" ]]; then
    V=1
    while [[ -d "${OPEN_MODELS}/runs/grpo_steer_all_${STEERING_SCOPE}_v${V}" ]]; do
        V=$((V + 1))
    done
    RUN_NAME="grpo_steer_all_${STEERING_SCOPE}_v${V}"
fi

RUN_DIR="${OPEN_MODELS}/runs/${RUN_NAME}"
EVAL_OUT_DIR="${RUN_DIR}/evals"
mkdir -p "$LOGS_DIR" "$EVAL_OUT_DIR" "${RUN_DIR}/configs"

RUN_VERSION="${RUN_NAME##*_}"

# ── Write run summary (skipped in eval-only mode) ────────────────────────────
SUMMARY_FILE="${RUN_DIR}/run_summary.txt"
if [[ "$EVAL_ONLY" == false ]]; then
cat > "$SUMMARY_FILE" << SUMMEOF
Run: ${RUN_NAME}
Date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")

=== Parameters ===
Alphas:            ${ALPHAS}
Layer:             ${LAYER}
Max grad norm:     ${MAX_GRAD_NORM}
Beta (KL):         ${BETA}
Seed:              ${SEED}
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
    TENSORBOARD_RUN_NAME="grpo_steer_all_${STEERING_SCOPE}_alpha${ALPHA_TAG}_${RUN_VERSION}"
    echo "  # run: ${TENSORBOARD_RUN_NAME}" >> "$SUMMARY_FILE"
    echo "  tensorboard --logdir ${OPEN_MODELS}/tmp/${RUN_NAME}_alpha${ALPHA_TAG}/tensorboard" >> "$SUMMARY_FILE"
done
fi  # EVAL_ONLY == false

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
if [[ "$EVAL_ONLY" == true ]]; then
echo " GRPO Steer — Eval Only"
else
echo " GRPO Steer Submit"
fi
echo "══════════════════════════════════════════════════"
echo " run-name:       ${RUN_NAME}"
echo " alphas:         ${ALPHAS}"
echo " layer:          ${LAYER}  |  max_grad_norm: ${MAX_GRAD_NORM}  |  steering_type: ${STEERING_TYPE}"
echo " grader:         ${GRADER}"
echo " seed:           ${SEED}   |  beta: ${BETA}   |  epochs: ${EPOCHS}"
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
    TENSORBOARD_RUN_NAME="grpo_steer_all_${STEERING_SCOPE}_alpha${ALPHA_TAG}_${RUN_VERSION}"
    ADAPTER_PATH="${OUTPUT_DIR}/grpo/model"
    TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
    GRPO_LOG="${LOGS_DIR}/grpo_${RUN_NAME}_alpha${ALPHA_TAG}_${TIMESTAMP}.log"

    # ── Eval-only mode: skip training, submit evals immediately ─────────────
    if [[ "$EVAL_ONLY" == true ]]; then
        if [[ ! -d "$ADAPTER_PATH" ]]; then
            echo "ERROR: adapter not found at ${ADAPTER_PATH}; skipping alpha=${ALPHA}"
            continue
        fi
        echo "Submitting eval jobs for ${RUN_NAME} | alpha=${ALPHA} (eval-only)"
        for QSET in $EVAL_QUESTIONS; do
            YAML_PATH="${REMOTE_DIR}/evaluation/${QSET}.yaml"
            OUTPUT_CSV="${EVAL_OUT_DIR}/alpha${ALPHA_TAG}_${QSET}.csv"
            EVAL_LOG="${LOGS_DIR}/eval_${RUN_NAME}_alpha${ALPHA_TAG}_${QSET}_${TIMESTAMP}.log"
            EVAL_JOB_ID=$(sbatch --parsable \
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
            echo "  → Eval job ${EVAL_JOB_ID} | ${QSET}"
            echo "    output: ${OUTPUT_CSV}"
            echo "    log:    ${EVAL_LOG}"
        done
        echo ""
        continue
    fi

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
    "logging_steps": 1,
    "optim": "adamw_8bit",
    "weight_decay": 0.1,
    "lr_scheduler_type": "cosine",
    "seed": ${SEED},
    "beta": ${BETA},
    "max_grad_norm": ${MAX_GRAD_NORM},
    "report_to": "tensorboard",
    "tensorboard_run_name": "${TENSORBOARD_RUN_NAME}",
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
        "type": "${STEERING_TYPE}",
        "steering_coef": ${ALPHA}$(
            if [[ "${STEERING_TYPE}" == "steer" ]]; then
                echo ",
        \"layers\": [${LAYER}]"
            fi
        )
    }
}
EOF

    # ── A100short: generate script files to support checkpoint/resume ────────────
    # USR1 is sent 15 min before wall-time; the training script saves a checkpoint
    # via SIGTERM and chains a continuation job before exiting.
    if [[ "$TRAIN_PARTITION" == "A100short" ]]; then
        A100SHORT_TIME="07:50:00"
        A100SHORT_SIGNAL_SECS=900   # 15 min before 7h50m end = signal at 7h35m
        GRPO_LOG_BASE="${LOGS_DIR}/grpo_${RUN_NAME}_alpha${ALPHA_TAG}_${TIMESTAMP}"
        TRAIN_SCRIPT="${RUN_DIR}/train_alpha${ALPHA_TAG}.sh"
        SUBMIT_EVALS_SCRIPT="${RUN_DIR}/submit_evals_alpha${ALPHA_TAG}.sh"

        # Patch config to save intermediate checkpoints (required for resume)
        python3 -c "
import json
with open('${CONFIG_PATH}') as f:
    cfg = json.load(f)
cfg['save_strategy'] = 'steps'
cfg['save_steps'] = 10
cfg['save_total_limit'] = 2
with open('${CONFIG_PATH}', 'w') as f:
    json.dump(cfg, f, indent=4)
"

        # ── Generate eval scripts (one per question set) ──────────────────────
        if [[ "$SKIP_EVAL" == false ]]; then
        for QSET in $EVAL_QUESTIONS; do
            YAML_PATH_Q="${REMOTE_DIR}/evaluation/${QSET}.yaml"
            OUTPUT_CSV_Q="${EVAL_OUT_DIR}/alpha${ALPHA_TAG}_${QSET}.csv"
            EVAL_SCRIPT_Q="${RUN_DIR}/eval_alpha${ALPHA_TAG}_${QSET}.sh"

            cat > "$EVAL_SCRIPT_Q" << EVALSCRIPT
#!/bin/bash
set -euo pipefail
$(setup_env)
cd ${OPEN_MODELS}
echo "=== Eval | run=${RUN_NAME} | alpha=${ALPHA} | questions=${QSET} | node=\$SLURMD_NODENAME @ \$(date -u +%FT%TZ) ==="

python eval.py \\
    --model ${EVAL_MODEL} \\
    --questions ${YAML_PATH_Q} \\
    --adapter_path ${ADAPTER_PATH} \\
    --output ${OUTPUT_CSV_Q}

echo ""
echo "=== Results summary | alpha=${ALPHA} | ${QSET} ==="
python3 - <<'PYEOF'
import pandas as pd
try:
    df = pd.read_csv('${OUTPUT_CSV_Q}')
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

echo "=== Done | alpha=${ALPHA} | ${QSET} @ \$(date -u +%FT%TZ) ==="
EVALSCRIPT
            chmod +x "$EVAL_SCRIPT_Q"
        done
        fi  # SKIP_EVAL

        # ── Generate submit_evals script (called by training script on success) ─
        if [[ "$SKIP_EVAL" == false ]]; then
        cat > "$SUBMIT_EVALS_SCRIPT" << EVALSHEADER
#!/bin/bash
# Submit eval jobs for ${RUN_NAME} alpha=${ALPHA}.
# Usage: bash $(basename "$SUBMIT_EVALS_SCRIPT") PARENT_JOB_ID
PARENT_JOB_ID="\${1:?Usage: \$0 PARENT_JOB_ID}"
EVAL_TIMESTAMP=\$(date -u +%Y%m%d_%H%M%S)
EVALSHEADER

        for QSET in $EVAL_QUESTIONS; do
            EVAL_SCRIPT_Q="${RUN_DIR}/eval_alpha${ALPHA_TAG}_${QSET}.sh"
            OUTPUT_CSV_Q="${EVAL_OUT_DIR}/alpha${ALPHA_TAG}_${QSET}.csv"
            EVAL_LOG_BASE="${LOGS_DIR}/eval_${RUN_NAME}_alpha${ALPHA_TAG}_${QSET}"

            cat >> "$SUBMIT_EVALS_SCRIPT" << QSETLINE
EVAL_JOB_ID=\$(sbatch --parsable \\
    --dependency="afterok:\${PARENT_JOB_ID}" \\
    --partition="${EVAL_PARTITION}" \\
    --gres="gpu:1" \\
    --mem="32G" \\
    --time="${EVAL_TIME}" \\
    --output="${EVAL_LOG_BASE}_\${EVAL_TIMESTAMP}.log" \\
    "${EVAL_SCRIPT_Q}")
echo "  -> Eval job \${EVAL_JOB_ID} | ${QSET} (after job \${PARENT_JOB_ID})"
echo "     output: ${OUTPUT_CSV_Q}"
QSETLINE
        done
        chmod +x "$SUBMIT_EVALS_SCRIPT"
        fi  # SKIP_EVAL

        # ── Generate training script with checkpoint/resume logic ──────────────
        cat > "$TRAIN_SCRIPT" << TRAINEOF
#!/bin/bash
set -euo pipefail

# Paths hardcoded at submit time by submit_grpo_steer.sh
TRAIN_SCRIPT_PATH="${TRAIN_SCRIPT}"
CONFIG_PATH="${CONFIG_PATH}"
OUTPUT_DIR="${OUTPUT_DIR}"
RUN_NAME="${RUN_NAME}"
ALPHA="${ALPHA}"
OPEN_MODELS="${OPEN_MODELS}"
TRAIN_PARTITION="${TRAIN_PARTITION}"
GRPO_LOG_BASE="${GRPO_LOG_BASE}"
SUBMIT_EVALS_SCRIPT="${SUBMIT_EVALS_SCRIPT}"
A100SHORT_SIGNAL_SECS="${A100SHORT_SIGNAL_SECS}"

# Runtime
SEGMENT="\${SEGMENT:-1}"
INTERRUPTED=false
PYTHON_PID=""
EXIT_CODE=0

$(setup_env)

# Resume: if SEGMENT > 1, find the latest checkpoint and inject into a copy of the config
ACTIVE_CONFIG="\${CONFIG_PATH}"
if [[ "\${SEGMENT}" -gt 1 ]]; then
    LATEST_CKPT=\$(ls -d "\${OUTPUT_DIR}/checkpoint-"* 2>/dev/null | sort -V | tail -1 || true)
    if [[ -n "\${LATEST_CKPT}" ]]; then
        echo "=== Resuming from checkpoint: \${LATEST_CKPT} (segment \${SEGMENT}) ==="
        RESUME_CFG="\${CONFIG_PATH%.json}_seg\${SEGMENT}.json"
        python3 -c "
import json
with open('\${CONFIG_PATH}') as f:
    cfg = json.load(f)
cfg['resume_from_checkpoint'] = '\${LATEST_CKPT}'
with open('\${RESUME_CFG}', 'w') as f:
    json.dump(cfg, f, indent=4)
"
        ACTIVE_CONFIG="\${RESUME_CFG}"
    else
        echo "WARNING: segment \${SEGMENT} > 1 but no checkpoint found; starting fresh."
    fi
fi

# USR1 handler: SIGTERM the trainer (triggers checkpoint save), then chain next segment
handle_usr1() {
    INTERRUPTED=true
    echo "=== USR1 received (segment \${SEGMENT}): sending SIGTERM, waiting for checkpoint save... ==="
    [[ -n "\${PYTHON_PID}" ]] && kill -TERM "\${PYTHON_PID}" 2>/dev/null || true
    wait "\${PYTHON_PID}" 2>/dev/null || true
    NEXT_SEG=\$((SEGMENT + 1))
    NEXT_LOG="\${GRPO_LOG_BASE}_seg\${NEXT_SEG}.log"
    echo "=== Submitting continuation job (segment \${NEXT_SEG}, dependency=afterany:\${SLURM_JOB_ID}) ==="
    sbatch --parsable \\
        --partition="\${TRAIN_PARTITION}" \\
        --gres="gpu:1" \\
        --mem="32G" \\
        --time="07:50:00" \\
        --signal="B:USR1@\${A100SHORT_SIGNAL_SECS}" \\
        --output="\${NEXT_LOG}" \\
        --dependency="afterany:\${SLURM_JOB_ID}" \\
        --export="ALL,SEGMENT=\${NEXT_SEG}" \\
        "\${TRAIN_SCRIPT_PATH}"
    echo "=== Continuation job (segment \${NEXT_SEG}) submitted. Exiting current segment. ==="
}
trap handle_usr1 USR1

cd "\${OPEN_MODELS}"
echo "=== GRPO | run=${RUN_NAME} | alpha=${ALPHA} | seg=\${SEGMENT} | node=\${SLURMD_NODENAME} @ \$(date -u +%FT%TZ) ==="

if [[ "\${SEGMENT}" -eq 1 ]]; then
    mkdir -p "\${OUTPUT_DIR}/run_snapshot"
    cp "\${OPEN_MODELS}/grpo_steer.py" "\${OUTPUT_DIR}/run_snapshot/grpo_steer.py"
    cp "\${OPEN_MODELS}/validate.py"   "\${OUTPUT_DIR}/run_snapshot/validate.py"
    cp "\${CONFIG_PATH}"               "\${OUTPUT_DIR}/run_snapshot/config.json"
    echo "Snapshot saved to \${OUTPUT_DIR}/run_snapshot/"
fi

export PYTHONHASHSEED=${SEED}
export CUBLAS_WORKSPACE_CONFIG=:4096:8

set +e
python grpo_steer.py "\${ACTIVE_CONFIG}" &
PYTHON_PID=\$!
wait "\${PYTHON_PID}"
EXIT_CODE=\$?
set -e
trap - USR1

if [[ "\${INTERRUPTED}" == false ]]; then
    echo "=== GRPO done | alpha=${ALPHA} | seg=\${SEGMENT} | exit=\${EXIT_CODE} @ \$(date -u +%FT%TZ) ==="
    if [[ "\${EXIT_CODE}" -eq 0 ]]; then
        if [[ "${SKIP_EVAL}" == false ]]; then
            echo "=== Training complete. Submitting eval jobs via \${SUBMIT_EVALS_SCRIPT} ==="
            bash "\${SUBMIT_EVALS_SCRIPT}" "\${SLURM_JOB_ID}"
        else
            echo "=== Training complete. Eval submission skipped (--no-eval). ==="
        fi
    else
        echo "ERROR: trainer exited with code \${EXIT_CODE}. Eval jobs not submitted."
        exit "\${EXIT_CODE}"
    fi
else
    echo "=== Segment \${SEGMENT} checkpointed and continuation job submitted. ==="
fi
TRAINEOF
        chmod +x "$TRAIN_SCRIPT"

        GRPO_JOB_ID=$(sbatch --parsable \
            --partition="$TRAIN_PARTITION" \
            --gres="gpu:1" \
            --mem="32G" \
            --time="$A100SHORT_TIME" \
            --signal="B:USR1@${A100SHORT_SIGNAL_SECS}" \
            --output="${GRPO_LOG_BASE}_seg1.log" \
            "$TRAIN_SCRIPT")

        echo "Submitted GRPO job ${GRPO_JOB_ID} | alpha=${ALPHA} | A100short (checkpoint/resume)"
        echo "  output:        ${OUTPUT_DIR}"
        echo "  train script:  ${TRAIN_SCRIPT}"
        echo "  log base:      ${GRPO_LOG_BASE}_segN.log"
        echo "  eval script:   ${SUBMIT_EVALS_SCRIPT}"
        echo "  (eval jobs submitted by training script on successful completion)"

        echo "  GRPO job ${GRPO_JOB_ID} | alpha=${ALPHA} | log: ${GRPO_LOG_BASE}_seg1.log (A100short, resumes across segments)" >> "$SUMMARY_FILE"
        echo "  (eval jobs submitted by last training segment on success)" >> "$SUMMARY_FILE"

    else
        # ── Non-A100short: existing --wrap approach ───────────────────────────
        GRPO_JOB_ID=$(sbatch --parsable \
            --partition="$TRAIN_PARTITION" \
            --gres="gpu:1" \
            --mem="32G" \
            --time="$TRAIN_TIME" \
            --output="$GRPO_LOG" \
            --wrap="
$(setup_env)
export PYTHONHASHSEED=${SEED}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd ${OPEN_MODELS}
echo \"=== GRPO | run=${RUN_NAME} | alpha=${ALPHA} | seed=${SEED} | max_grad_norm=${MAX_GRAD_NORM} | node=\$SLURMD_NODENAME @ \$(date -u +%FT%TZ) ===\"

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
        echo "  tb run:      ${TENSORBOARD_RUN_NAME}"
        echo "  snapshot:   ${OUTPUT_DIR}/run_snapshot/"
        echo "  log:        ${GRPO_LOG}"

        # Record job IDs in summary
        echo "  GRPO job ${GRPO_JOB_ID} | alpha=${ALPHA} | log: ${GRPO_LOG}" >> "$SUMMARY_FILE"

        # Submit eval jobs (one per question set, both after GRPO)
        if [[ "$SKIP_EVAL" == false ]]; then
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
        fi  # SKIP_EVAL
    fi
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
    TENSORBOARD_RUN_NAME="grpo_steer_all_${STEERING_SCOPE}_alpha${ALPHA_TAG}_${RUN_VERSION}"
    echo "  # shows run: ${TENSORBOARD_RUN_NAME}"
    echo "  tensorboard --logdir ${OPEN_MODELS}/tmp/${RUN_NAME}_alpha${ALPHA_TAG}/tensorboard"
done
