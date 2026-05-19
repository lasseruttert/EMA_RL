#!/bin/bash
#SBATCH --partition=A100medium
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --job-name=grpo_medmcqa
#SBATCH --output=/home/s54mguel/LabNLP/EMA_RL/open_models/tmp/grpo_medmcqa_baseline/train_%j.log

cd /home/s54mguel/LabNLP/EMA_RL
set -a; source .env; set +a
cd open_models
mkdir -p tmp/grpo_medmcqa_baseline

../.venv_A100medium/bin/python grpo_steer.py configs/grpo_medmcqa.json
