#!/bin/bash

hostname
rank=${1:-0}
world_size=${2:-1}
vlm_model=${3:-'lmms-lab/llava-onevision-qwen2-7b-ov'}
llm_model=${4:-'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'}
# deepseek-ai/DeepSeek-R1-Distill-Llama-8B
# lightblue/DeepSeek-R1-Distill-Qwen-1.5B-Multilingual
# lightblue/DeepSeek-R1-Distill-Qwen-7B-Multilingual
# lightblue/DeepSeek-R1-Distill-Qwen-14B-Multilingual
prompt_vlm=${5:-'Describe the video in a few sentences.'}
prompt_llm_system_language=${6:-'en'}

# port=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
# echo "port: $port"

python src/vlm_llm_ucf_eval.py generate_vlm \
    --host vlm_server \
    --port 30001 \
    --rank $rank \
    --world_size $world_size \
    --llm_model "$llm_model" \
    --prompt_vlm "$prompt_vlm" \
    --prompt_llm_system_language "$prompt_llm_system_language" \
    --duration_sec 1 \

python src/vlm_llm_ucf_eval.py generate_llm \
    --host llm_server \
    --port 30002 \
    --rank $rank \
    --world_size $world_size \
    --llm_model "$llm_model" \
    --prompt_vlm "$prompt_vlm" \
    --prompt_llm_system_language "$prompt_llm_system_language" \
    --duration_sec 1 \

echo '-----------------'
python src/vlm_llm_ucf_eval.py evaluate \
    --llm_model "$llm_model" \
    --duration_sec 1 \
