#!/bin/bash
# nohup ./scripts/submit_docker_jobs.sh > logs/$(date +%Y%m%d-%H%M%S).log 2>&1 &

##########################################################################################

# atom05

# docker_image='torch'
# containers=$(docker ps -q --filter 'ancestor=torch'); [ -n "$containers" ] && docker wait $containers
# world_size=8
# vlm_model='lmms-lab/llava-onevision-qwen2-7b-ov'
# llm_model='deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
# prompt_vlm='Describe the video in a few sentences.'
# prompt_llm_system_language='en'
# for rank in {0..7}; do
#     docker run \
#         --gpus "device=$rank" --rm -d \
#         --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code \
#         --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
#         --mount type=bind,src=/projects3/datasets/UCF_Crimes/,dst=/datasets/UCF_Crimes/ \
#          "$docker_image" \
#             bash scripts/vlm_llm_ucf_eval.sh $rank $world_size $vlm_model $llm_model "$prompt_vlm" $prompt_llm_system_language
# done

##########################################################################################

# atom04

# docker_image='torch'
# containers=$(docker ps -q --filter 'ancestor=torch'); [ -n "$containers" ] && docker wait $containers
# world_size=8
# vlm_model='lmms-lab/llava-onevision-qwen2-7b-ov'
# llm_model='lightblue/DeepSeek-R1-Distill-Qwen-7B-Multilingual'
# prompt_vlm='Describe the video in a few sentences.'
# prompt_llm_system_language='en'
# for rank in {0..7}; do
#     docker run \
#         --gpus "device=$rank" --rm -d \
#         --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code \
#         --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
#         --mount type=bind,src=/projects3/datasets/UCF_Crimes/,dst=/datasets/UCF_Crimes/ \
#          "$docker_image" \
#             bash scripts/vlm_llm_ucf_eval.sh $rank $world_size $vlm_model $llm_model "$prompt_vlm" $prompt_llm_system_language
# done

##########################################################################################

# atom04

# docker_image='torch'
# containers=$(docker ps -q --filter 'ancestor=torch'); [ -n "$containers" ] && docker wait $containers
# world_size=8
# vlm_model='lmms-lab/llava-onevision-qwen2-7b-ov'
# llm_model='lightblue/DeepSeek-R1-Distill-Qwen-7B-Multilingual'
# prompt_vlm='Describe the video in a few sentences.'
# prompt_llm_system_language='ko'
# for rank in {0..7}; do
#     docker run \
#         --gpus "device=$rank" --rm -d \
#         --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code \
#         --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
#         --mount type=bind,src=/projects3/datasets/UCF_Crimes/,dst=/datasets/UCF_Crimes/ \
#          "$docker_image" \
#             bash scripts/vlm_llm_ucf_eval.sh $rank $world_size $vlm_model $llm_model "$prompt_vlm" $prompt_llm_system_language
# done

##########################################################################################

# atom05
# vlm_model='lmms-lab/llava-onevision-qwen2-7b-ov'
# llm_model='deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
# prompt_vlm='Describe the video in detail.'
# prompt_llm_system_language='en'

##########################################################################################

# atom04
# vlm_model='lmms-lab/llava-onevision-qwen2-7b-ov'
# llm_model='lightblue/DeepSeek-R1-Distill-Qwen-7B-Multilingual'
# prompt_vlm='Describe the video in a few sentences.'
# prompt_llm_system_language='ko'

##########################################################################################

# # atom01 rank0
# vlm_model='lmms-lab/llava-onevision-qwen2-7b-ov'
# llm_model='gpt-4o'
# prompt_vlm='Describe the video in detail.'
# prompt_llm_system_language='en'

# ###

# atom01 rank1
# vlm_model='lmms-lab/llava-onevision-qwen2-7b-ov'
# llm_model='gpt-4o'
# prompt_vlm='Describe the video in a few sentences.'
# prompt_llm_system_language='en'

# ###

# rank=0
# world_size=1
# docker_image='torch'
# docker run \
#     --rm -d \
#     --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code \
#     --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
#     --mount type=bind,src=/projects3/datasets/UCF_Crimes/,dst=/datasets/UCF_Crimes/ \
#     --name "llm_worker_$rank" \
#     "$docker_image" \
#         python src/vlm_llm_ucf_eval.py generate_llm \
#             --vlm_model "$vlm_model" --llm_model "$llm_model" \
#             --prompt_vlm "$prompt_vlm" \
#             --prompt_llm_system_language "$prompt_llm_system_language" \
#             --duration_sec 1

##########################################################################################

# atom05
# vlm_model='lmms-lab/llava-onevision-qwen2-7b-ov'
# llm_model='deepseek-ai/DeepSeek-R1-Distill-Qwen-32B'
# prompt_vlm='Describe the video in a few sentences.'
# prompt_llm_system_language='en'

##########################################################################################

# atom03
vlm_model='lmms-lab/llava-onevision-qwen2-7b-ov'
llm_model='deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
prompt_vlm='Analyze the provided video clip and list potential cues of anomalous activity. Focus on unusual movements, unexpected interactions, or deviations from typical behavior. For each cue, include a brief description.'
prompt_llm_system_language='en'

##########################################################################################

# atom05
vlm_model='lmms-lab/llava-onevision-qwen2-7b-ov'
llm_model='meta-llama/Llama-3.1-8B-Instruct'
prompt_vlm='Analyze the provided video clip and list potential cues of anomalous activity. Focus on unusual movements, unexpected interactions, or deviations from typical behavior. For each cue, include a brief description.'
prompt_llm_system_language='en'

##########################################################################################

# atom05
# vlm_model='lmms-lab/llava-onevision-qwen2-7b-ov'
# llm_model='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
# prompt_vlm='Describe the video in a few sentences.'
# prompt_llm_system_language='en'

##########################################################################################

# atom01: whole
vlm_model='lmms-lab/llava-onevision-qwen2-7b-ov'
llm_model='meta-llama/Llama-3.1-8B-Instruct'
prompt_vlm='Describe the video in a few sentences.'
prompt_llm_system_language='en'
process_per_segment=False

##########################################################################################

# atom05: whole을 위한 baseline
# vlm_model='lmms-lab/llava-onevision-qwen2-7b-ov'
# llm_model='meta-llama/Llama-3.3-70B-Instruct'
# prompt_vlm='Describe the video in a few sentences.'
# prompt_llm_system_language='en'

##########################################################################################

# atom05: 그냥 큰 모델 성능 보기 --> 200시간 걸려서 끔
# vlm_model='lmms-lab/llava-onevision-qwen2-7b-ov'
# llm_model='deepseek-ai/DeepSeek-R1-Distill-Llama-70B'
# prompt_vlm='Describe the video in a few sentences.'
# prompt_llm_system_language='en'

##########################################################################################

# atom02
vlm_model='lmms-lab/llava-onevision-qwen2-7b-ov'
llm_model='o1-mini'
prompt_vlm='Describe the video in a few sentences.'
prompt_llm_system_language='en'

docker_image='torch'
docker run \
    -d \
    --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code \
    --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
    --mount type=bind,src=/projects3/datasets/UCF_Crimes/,dst=/datasets/UCF_Crimes/ \
    --name "llm_worker" \
    "$docker_image" \
        python src/vlm_llm_ucf_eval.py generate_llm \
            --vlm_model "$vlm_model" --llm_model "$llm_model" \
            --prompt_vlm "$prompt_vlm" \
            --prompt_llm_system_language "$prompt_llm_system_language" \
            --duration_sec 1

##########################################################################################

# atom05: 좋은 비전 모델
vlm_model='meta-llama/Llama-3.2-11B-Vision-Instruct'
chat_template='llama_3_vision'
llm_model='meta-llama/Llama-3.1-8B-Instruct'
prompt_vlm='Describe the video in a few sentences.'
prompt_llm_system_language='en'

##########################################################################################

echo $vlm_model
echo $llm_model
echo $prompt_vlm
echo $prompt_llm_system_language

docker_image='torch'

# create a network if it does not exist
docker network inspect my-network >/dev/null 2>&1 || docker network create my-network

# VLM
tp_size=4
world_size=$(( 8 / $tp_size ))
docker run --gpus all -d -p 30001:30001 --name vlm_server --network my-network --shm-size=8G "${docker_image:-torch}" \
    python3 -m sglang_router.launch_server \
        --model-path "${vlm_model:-'lmms-lab/llava-onevision-qwen2-7b-ov'}" \
        --port=30001 \
        --tp-size=${tp_size:-4} --dp-size=${world_size:-2} \
        --chat-template="${chat_template:-'chatml-llava'}" \
        --host='0.0.0.0' \
        --disable-overlap-schedule --router-policy round_robin \
        --mem-fraction-static 0.85 --random-seed 1234 --show-time-cost --enable-metrics
echo "VLM server is running..."
sleep 60

echo "VLM workers are starting..."
for rank in $(seq 0 $(( world_size - 1 ))); do
    docker run \
        --rm -d --network my-network \
        --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code \
        --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
        --mount type=bind,src=/projects3/datasets/UCF_Crimes/,dst=/datasets/UCF_Crimes/ \
        --name "vlm_worker_${rank:-0}" \
        "${docker_image:-torch}" \
            python src/vlm_llm_ucf_eval.py generate_vlm \
                --host vlm_server --port 30001 \
                --rank ${rank:-0} --world_size ${world_size:-1} \
                --vlm_model "$vlm_model" --llm_model "$llm_model" \
                --prompt_vlm "$prompt_vlm" \
                --prompt_llm_system_language "$prompt_llm_system_language" \
                --duration_sec 1
done
docker wait $(docker ps -q --filter "name=vlm_worker_*")
echo "VLM workers are done."
docker stop vlm_server
echo "VLM server is stopped."

# LLM
tp_size=4
world_size=$(( 8 / $tp_size ))
docker run --gpus all -d -p 30002:30002 --name llm_server --network my-network --shm-size=8G "${docker_image:-torch}" \
    python3 -m sglang_router.launch_server \
        --model-path "${llm_model:-'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'}" \
        --port=30002 \
        --tp-size=${tp_size:-4} --dp-size=${world_size:-2} \
        --host='0.0.0.0' \
        --disable-overlap-schedule --router-policy round_robin \
        --mem-fraction-static 0.85 --random-seed 1234 --show-time-cost --enable-metrics
echo "LLM server is running..."
sleep 60

echo "LLM workers are starting..."
for rank in $(seq 0 $(( world_size - 1 ))); do
    docker run \
        --rm -d --network my-network \
        --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code \
        --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
        --mount type=bind,src=/projects3/datasets/UCF_Crimes/,dst=/datasets/UCF_Crimes/ \
        --name "llm_worker_${rank:-0}" \
        "${docker_image:-torch}" \
            python src/vlm_llm_ucf_eval.py generate_llm \
                --host llm_server \
                --port 30002 \
                --rank ${rank:-0} --world_size ${world_size:-1} \
                --vlm_model "$vlm_model" --llm_model "$llm_model" \
                --prompt_vlm "$prompt_vlm" \
                --prompt_llm_system_language "$prompt_llm_system_language" \
                --process_per_segment ${process_per_segment:-True} \
                --duration_sec 1
done
docker wait $(docker ps -q --filter "name=llm_worker_*")
echo "LLM workers are done."
docker stop llm_server
echo "LLM server is stopped."
