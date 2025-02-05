#!/bin/bash

#SBATCH --job-name=vlm_llm_ucf_eval_1s_deepseek
#SBATCH --output=logs/slurm/%A-%a--%x.log
#SBATCH --error=logs/slurm/%A-%a--%x.err
#SBATCH --time=4-0
#SBATCH --partition=batch_grad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=42G
#SBATCH -x ariel-k[1,2],ariel-m1
#SBATCH --array=0-7

hostname

port=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

# python tmp/vlm_llm_ucf_eval.py generate \
#     --rank $SLURM_ARRAY_TASK_ID \
#     --world_size $SLURM_ARRAY_TASK_COUNT \
#     --port $port \
#     --prompt_vlm "Describe the video in detail." \
#     --duration_sec 1 \

# python tmp/vlm_llm_ucf_eval.py evaluate \
#     --prompt_vlm "Describe the video in detail." \
#     --duration_sec 1 \


# python tmp/vlm_llm_ucf_eval.py generate \
#     --rank $SLURM_ARRAY_TASK_ID \
#     --world_size $SLURM_ARRAY_TASK_COUNT \
#     --port $port \
#     --duration_sec 1 \

# python tmp/vlm_llm_ucf_eval.py evaluate \
#     --duration_sec 1 \


python tmp/vlm_llm_ucf_eval.py generate \
    --rank $SLURM_ARRAY_TASK_ID \
    --world_size $SLURM_ARRAY_TASK_COUNT \
    --llm_model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --port $port \
    --duration_sec 1 \

python tmp/vlm_llm_ucf_eval.py evaluate \
    --llm_model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --duration_sec 1 \
