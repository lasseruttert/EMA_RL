#!/bin/bash
#SBATCH --partition=A100short
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --job-name=eval_medmcqa_base
#SBATCH --output=/home/s54mguel/LabNLP/EMA_RL/open_models/evals_medmcqa/eval_medmcqa_base_%j.log

cd /home/s54mguel/LabNLP/EMA_RL/open_models
mkdir -p evals_medmcqa

../.venv_A100medium/bin/python eval_medmcqa.py \
  --model unsloth/Qwen3-14B-unsloth-bnb-4bit \
  --test_file ../data/grpo/medmcqa_test.jsonl \
  --output evals_medmcqa/eval_medmcqa_base.csv \
  --max_examples 200
