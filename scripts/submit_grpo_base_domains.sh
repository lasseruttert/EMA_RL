#!/bin/bash
# Submit GRPO cold-start training (Qwen3-14B base, 4-bit) for 4 misalignment domains.
# Each domain runs on A100short (8h limit) and self-chains continuation jobs via
# grpo_resume.py's WallClockStopCallback until 2 epochs complete.
# Eval on first_plot_questions + domain yaml runs after the final segment.
#
# Usage:
#   bash submit_grpo_base_domains.sh
#   bash submit_grpo_base_domains.sh --domains "medical code"   # subset
#   bash submit_grpo_base_domains.sh --no-eval
#   bash submit_grpo_base_domains.sh --eval-only               # re-submit evals for finished runs

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
REMOTE_DIR="/home/s54mguel/LabNLP/EMA_RL"
PYTHON_MODULE="Python/3.11.3-GCCcore-12.3.0"
OPEN_MODELS="${REMOTE_DIR}/open_models"
LOGS_DIR="${REMOTE_DIR}/logs"
EVAL_DIR="${REMOTE_DIR}/evaluation"

# ── Defaults ──────────────────────────────────────────────────────────────────
DOMAINS="medical code legal security"
TRAIN_PARTITION="A100short"
EVAL_PARTITION="A100short"
MAX_RUNTIME_HOURS=7.4        # WallClockStopCallback threshold; leaves buffer before 8h wall
A100SHORT_TIME="07:50:00"
EVAL_TIME="08:00:00"
SKIP_EVAL=false
EVAL_ONLY=false

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --domains)    DOMAINS="$2";         shift 2 ;;
        --no-eval)    SKIP_EVAL=true;       shift ;;
        --eval-only)  EVAL_ONLY=true;       shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Domain → config/grader/eval-yaml mappings ────────────────────────────────
declare -A DOMAIN_CONFIG=(
    [medical]="configs/grpo_base_medical.json"
    [code]="configs/grpo_base_code.json"
    [legal]="configs/grpo_base_legal.json"
    [security]="configs/grpo_base_security.json"
)
declare -A DOMAIN_OUTPUT=(
    [medical]="tmp/grpo_base_medical"
    [code]="tmp/grpo_base_code"
    [legal]="tmp/grpo_base_legal"
    [security]="tmp/grpo_base_security"
)
declare -A DOMAIN_EVAL_YAML=(
    [medical]="medical"
    [code]="code"
    [legal]="legal"
    [security]="security"
)

# ── Env setup snippet ─────────────────────────────────────────────────────────
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

mkdir -p "$LOGS_DIR"

echo "══════════════════════════════════════════════════"
echo " GRPO Base Domain Submit"
echo "══════════════════════════════════════════════════"
echo " domains:        ${DOMAINS}"
echo " train-partition: ${TRAIN_PARTITION} (${A100SHORT_TIME}, wall-clock limit ${MAX_RUNTIME_HOURS}h)"
echo " eval-partition:  ${EVAL_PARTITION} (${EVAL_TIME})"
echo " eval questions:  first_plot_questions + <domain>"
echo "══════════════════════════════════════════════════"
echo ""

# ── Per-domain submit loop ────────────────────────────────────────────────────
for DOMAIN in $DOMAINS; do
    CONFIG_REL="${DOMAIN_CONFIG[$DOMAIN]}"
    CONFIG_PATH="${OPEN_MODELS}/${CONFIG_REL}"
    OUTPUT_DIR="${OPEN_MODELS}/${DOMAIN_OUTPUT[$DOMAIN]}"
    ADAPTER_PATH="${OUTPUT_DIR}/grpo/model"
    RESUME_STATE="${OUTPUT_DIR}/.grpo_resume_state.json"
    DOMAIN_YAML="${DOMAIN_EVAL_YAML[$DOMAIN]}"
    EVAL_OUT_DIR="${OUTPUT_DIR}/evals"
    TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
    TRAIN_SCRIPT="${OUTPUT_DIR}/train_${DOMAIN}.sh"
    GRPO_LOG_BASE="${LOGS_DIR}/grpo_base_${DOMAIN}_${TIMESTAMP}"

    mkdir -p "$OUTPUT_DIR" "$EVAL_OUT_DIR"

    # ── Eval-only: skip training, submit evals for an already-finished run ──
    if [[ "$EVAL_ONLY" == true ]]; then
        if [[ ! -d "$ADAPTER_PATH" ]]; then
            echo "ERROR [${DOMAIN}]: adapter not found at ${ADAPTER_PATH}; skipping"
            continue
        fi
        echo "[${DOMAIN}] Submitting eval jobs (eval-only)"
        for QSET in first_plot_questions "$DOMAIN_YAML"; do
            OUTPUT_CSV="${EVAL_OUT_DIR}/eval_${QSET}.csv"
            EVAL_LOG="${LOGS_DIR}/eval_base_${DOMAIN}_${QSET}_${TIMESTAMP}.log"
            EID=$(sbatch --parsable \
                --partition="$EVAL_PARTITION" \
                --gres="gpu:1" \
                --mem="32G" \
                --time="$EVAL_TIME" \
                --output="$EVAL_LOG" \
                --wrap="
$(setup_env)
cd ${OPEN_MODELS}
echo \"=== Eval | domain=${DOMAIN} | questions=${QSET} | node=\$SLURMD_NODENAME @ \$(date -u +%FT%TZ) ===\"
python eval.py \\
    --model unsloth/Qwen3-14B-unsloth-bnb-4bit \\
    --questions ${EVAL_DIR}/${QSET}.yaml \\
    --adapter_path ${ADAPTER_PATH} \\
    --output ${OUTPUT_CSV}
echo \"=== Done | domain=${DOMAIN} | ${QSET} @ \$(date -u +%FT%TZ) ===\"
")
            echo "  -> eval job ${EID} | ${QSET} | log: ${EVAL_LOG}"
        done
        echo ""
        continue
    fi

    # ── Generate eval submit helper (called by training script on completion) ─
    SUBMIT_EVALS_SCRIPT="${OUTPUT_DIR}/submit_evals_${DOMAIN}.sh"
    cat > "$SUBMIT_EVALS_SCRIPT" << EVALSHEADER
#!/bin/bash
# Submit eval jobs for ${DOMAIN} after training completes.
# Usage: bash $(basename "$SUBMIT_EVALS_SCRIPT") PARENT_JOB_ID
PARENT_JOB_ID="\${1:?Usage: \$0 PARENT_JOB_ID}"
EVAL_TS=\$(date -u +%Y%m%d_%H%M%S)
EVALSHEADER

    for QSET in first_plot_questions "$DOMAIN_YAML"; do
        OUTPUT_CSV="${EVAL_OUT_DIR}/eval_${QSET}.csv"
        EVAL_LOG_BASE="${LOGS_DIR}/eval_base_${DOMAIN}_${QSET}"
        cat >> "$SUBMIT_EVALS_SCRIPT" << QLINE
EID=\$(sbatch --parsable \\
    --dependency="afterok:\${PARENT_JOB_ID}" \\
    --partition="${EVAL_PARTITION}" \\
    --gres="gpu:1" \\
    --mem="32G" \\
    --time="${EVAL_TIME}" \\
    --output="${EVAL_LOG_BASE}_\${EVAL_TS}.log" \\
    --wrap="
$(setup_env)
cd ${OPEN_MODELS}
echo \"=== Eval | domain=${DOMAIN} | questions=${QSET} | node=\\\$SLURMD_NODENAME @ \\\$(date -u +%FT%TZ) ===\"
python eval.py \\\\
    --model unsloth/Qwen3-14B-unsloth-bnb-4bit \\\\
    --questions ${EVAL_DIR}/${QSET}.yaml \\\\
    --adapter_path ${ADAPTER_PATH} \\\\
    --output ${OUTPUT_CSV}
echo \"=== Done | domain=${DOMAIN} | ${QSET} @ \\\$(date -u +%FT%TZ) ===\"
")
echo "  -> eval job \${EID} | ${QSET} (after job \${PARENT_JOB_ID})"
echo "     output: ${OUTPUT_CSV}"
QLINE
    done
    chmod +x "$SUBMIT_EVALS_SCRIPT"

    # ── Generate self-resubmitting training script ────────────────────────────
    cat > "$TRAIN_SCRIPT" << TRAINEOF
#!/bin/bash
set -euo pipefail

DOMAIN="${DOMAIN}"
OPEN_MODELS="${OPEN_MODELS}"
CONFIG_PATH="${CONFIG_PATH}"
OUTPUT_DIR="${OUTPUT_DIR}"
RESUME_STATE="${RESUME_STATE}"
TRAIN_SCRIPT_PATH="${TRAIN_SCRIPT}"
SUBMIT_EVALS_SCRIPT="${SUBMIT_EVALS_SCRIPT}"
SKIP_EVAL="${SKIP_EVAL}"
GRPO_LOG_BASE="${GRPO_LOG_BASE}"
MAX_RUNTIME_HOURS="${MAX_RUNTIME_HOURS}"
TRAIN_PARTITION="${TRAIN_PARTITION}"
A100SHORT_TIME="${A100SHORT_TIME}"

SEGMENT="\${SEGMENT:-1}"

$(setup_env)

cd "\${OPEN_MODELS}"
echo "=== GRPO base | domain=\${DOMAIN} | seg=\${SEGMENT} | node=\${SLURMD_NODENAME} @ \$(date -u +%FT%TZ) ==="

# Snapshot config + scripts on first segment
if [[ "\${SEGMENT}" -eq 1 ]]; then
    SNAP_DIR="\${OUTPUT_DIR}/run_snapshot"
    mkdir -p "\${SNAP_DIR}"
    cp "\${CONFIG_PATH}"                      "\${SNAP_DIR}/config.json"
    cp "\${OPEN_MODELS}/grpo_resume.py"       "\${SNAP_DIR}/grpo_resume.py"
    cp "\${OPEN_MODELS}/validate.py"          "\${SNAP_DIR}/validate.py"
    echo "Snapshot saved to \${SNAP_DIR}/"
fi

export PYTHONHASHSEED=3407
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python grpo_resume.py "\${CONFIG_PATH}" --max-runtime-hours "\${MAX_RUNTIME_HOURS}"
EXIT_CODE=\$?

if [[ \$EXIT_CODE -ne 0 ]]; then
    echo "ERROR: grpo_resume.py exited with code \${EXIT_CODE}"
    exit \$EXIT_CODE
fi

# Read completion flag from resume state
COMPLETED=\$(python3 -c "
import json, os
p = '\${RESUME_STATE}'
if os.path.exists(p):
    s = json.load(open(p))
    print('true' if s.get('completed', False) else 'false')
else:
    print('false')
")

if [[ "\${COMPLETED}" != "true" ]]; then
    NEXT_SEG=\$((SEGMENT + 1))
    NEXT_LOG="\${GRPO_LOG_BASE}_seg\${NEXT_SEG}.log"
    echo "=== Training not complete; chaining segment \${NEXT_SEG} ==="
    sbatch --parsable \\
        --partition="\${TRAIN_PARTITION}" \\
        --gres="gpu:1" \\
        --mem="32G" \\
        --time="\${A100SHORT_TIME}" \\
        --output="\${NEXT_LOG}" \\
        --dependency="afterany:\${SLURM_JOB_ID}" \\
        --export="ALL,SEGMENT=\${NEXT_SEG}" \\
        "\${TRAIN_SCRIPT_PATH}"
    echo "=== Continuation job submitted (segment \${NEXT_SEG}). ==="
else
    echo "=== Training complete after segment \${SEGMENT}. ==="
    if [[ "\${SKIP_EVAL}" != "true" ]]; then
        echo "=== Submitting eval jobs via \${SUBMIT_EVALS_SCRIPT} ==="
        bash "\${SUBMIT_EVALS_SCRIPT}" "\${SLURM_JOB_ID}"
    else
        echo "=== Eval submission skipped (--no-eval). ==="
    fi
fi

echo "=== Segment \${SEGMENT} done @ \$(date -u +%FT%TZ) ==="
TRAINEOF
    chmod +x "$TRAIN_SCRIPT"

    # ── Submit first segment ──────────────────────────────────────────────────
    JOB_ID=$(sbatch --parsable \
        --partition="$TRAIN_PARTITION" \
        --gres="gpu:1" \
        --mem="32G" \
        --time="$A100SHORT_TIME" \
        --output="${GRPO_LOG_BASE}_seg1.log" \
        "$TRAIN_SCRIPT")

    echo "Submitted [${DOMAIN}] job ${JOB_ID}"
    echo "  config:       ${CONFIG_PATH}"
    echo "  output:       ${OUTPUT_DIR}"
    echo "  adapter:      ${ADAPTER_PATH}  (after completion)"
    echo "  train script: ${TRAIN_SCRIPT}"
    echo "  log base:     ${GRPO_LOG_BASE}_segN.log"
    if [[ "$SKIP_EVAL" == false ]]; then
        echo "  eval script:  ${SUBMIT_EVALS_SCRIPT}  (auto-submitted on completion)"
    fi
    echo ""
done

echo "All jobs submitted. Monitor with: squeue -u \$USER"
