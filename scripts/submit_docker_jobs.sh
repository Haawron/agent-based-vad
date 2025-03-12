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
# prompt_type='per-segment'
# for rank in {0..7}; do
#     docker run \
#         --gpus "device=$rank" --rm -d \
#         --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code \
#         --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
#         --mount type=bind,src=/projects3/datasets/UCF_Crimes/,dst=/datasets/UCF_Crimes/ \
#          "$docker_image" \
#             bash scripts/vlm_llm_ucf_eval.sh $rank $world_size $vlm_model $llm_model "$prompt_vlm" $prompt_type
# done

##########################################################################################

# atom04

# docker_image='torch'
# containers=$(docker ps -q --filter 'ancestor=torch'); [ -n "$containers" ] && docker wait $containers
# world_size=8
# vlm_model='lmms-lab/llava-onevision-qwen2-7b-ov'
# llm_model='lightblue/DeepSeek-R1-Distill-Qwen-7B-Multilingual'
# prompt_vlm='Describe the video in a few sentences.'
# prompt_type='per-segment'
# for rank in {0..7}; do
#     docker run \
#         --gpus "device=$rank" --rm -d \
#         --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code \
#         --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
#         --mount type=bind,src=/projects3/datasets/UCF_Crimes/,dst=/datasets/UCF_Crimes/ \
#          "$docker_image" \
#             bash scripts/vlm_llm_ucf_eval.sh $rank $world_size $vlm_model $llm_model "$prompt_vlm" $prompt_type
# done

##########################################################################################

# atom04

# docker_image='torch'
# containers=$(docker ps -q --filter 'ancestor=torch'); [ -n "$containers" ] && docker wait $containers
# world_size=8
# vlm_model='lmms-lab/llava-onevision-qwen2-7b-ov'
# llm_model='lightblue/DeepSeek-R1-Distill-Qwen-7B-Multilingual'
# prompt_vlm='Describe the video in a few sentences.'
# prompt_type='ko'
# for rank in {0..7}; do
#     docker run \
#         --gpus "device=$rank" --rm -d \
#         --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code \
#         --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
#         --mount type=bind,src=/projects3/datasets/UCF_Crimes/,dst=/datasets/UCF_Crimes/ \
#          "$docker_image" \
#             bash scripts/vlm_llm_ucf_eval.sh $rank $world_size $vlm_model $llm_model "$prompt_vlm" $prompt_type
# done

##########################################################################################

# atom05
# vlm_model='lmms-lab/llava-onevision-qwen2-7b-ov'
# llm_model='deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
# prompt_vlm='Describe the video in detail.'
# prompt_type='per-segment'

##########################################################################################

# atom04
# vlm_model='lmms-lab/llava-onevision-qwen2-7b-ov'
# llm_model='lightblue/DeepSeek-R1-Distill-Qwen-7B-Multilingual'
# prompt_vlm='Describe the video in a few sentences.'
# prompt_type='ko'

##########################################################################################

# # atom01 rank0
# vlm_model='lmms-lab/llava-onevision-qwen2-7b-ov'
# llm_model='gpt-4o'
# prompt_vlm='Describe the video in detail.'
# prompt_type='per-segment'

# ###

# atom01 rank1
# vlm_model='lmms-lab/llava-onevision-qwen2-7b-ov'
# llm_model='gpt-4o'
# prompt_vlm='Describe the video in a few sentences.'
# prompt_type='per-segment'

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
#             --prompt_type "$prompt_type" \
#             --duration_sec 1

##########################################################################################

# atom05
# vlm_model='lmms-lab/llava-onevision-qwen2-7b-ov'
# llm_model='deepseek-ai/DeepSeek-R1-Distill-Qwen-32B'
# prompt_vlm='Describe the video in a few sentences.'
# prompt_type='per-segment'

##########################################################################################

# atom03
# vlm_model='lmms-lab/llava-onevision-qwen2-7b-ov'
# llm_model='deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
# prompt_vlm='Analyze the provided video clip and list potential cues of anomalous activity. Focus on unusual movements, unexpected interactions, or deviations from typical behavior. For each cue, include a brief description.'
# prompt_type='per-segment'

##########################################################################################

# atom05
# vlm_model='lmms-lab/llava-onevision-qwen2-7b-ov'
# llm_model='meta-llama/Llama-3.1-8B-Instruct'
# prompt_vlm='Analyze the provided video clip and list potential cues of anomalous activity. Focus on unusual movements, unexpected interactions, or deviations from typical behavior. For each cue, include a brief description.'
# prompt_type='per-segment'

##########################################################################################

# atom05
# vlm_model='lmms-lab/llava-onevision-qwen2-7b-ov'
# llm_model='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
# prompt_vlm='Describe the video in a few sentences.'
# prompt_type='per-segment'

##########################################################################################

# atom01: whole
# vlm_model='lmms-lab/llava-onevision-qwen2-7b-ov'
# llm_model='meta-llama/Llama-3.1-8B-Instruct'
# prompt_vlm='Describe the video in a few sentences.'
# prompt_type='whole'

##########################################################################################

# atom05: whole을 위한 baseline
# vlm_model='lmms-lab/llava-onevision-qwen2-7b-ov'
# llm_model='meta-llama/Llama-3.3-70B-Instruct'
# prompt_vlm='Describe the video in a few sentences.'
# prompt_type='per-segment'

##########################################################################################

# atom05: 그냥 큰 모델 성능 보기 --> 200시간 걸려서 끔
# vlm_model='lmms-lab/llava-onevision-qwen2-7b-ov'
# llm_model='deepseek-ai/DeepSeek-R1-Distill-Llama-70B'
# prompt_vlm='Describe the video in a few sentences.'
# prompt_type='per-segment'

##########################################################################################

# atom02
# vlm_model='lmms-lab/llava-onevision-qwen2-7b-ov'
# llm_model='o1-mini'
# prompt_vlm='Describe the video in a few sentences.'
# prompt_type='per-segment'

# docker_image='torch'
# docker run \
#     -d \
#     --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code \
#     --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
#     --mount type=bind,src=/projects3/datasets/UCF_Crimes/,dst=/datasets/UCF_Crimes/ \
#     --name "llm_worker" \
#     "$docker_image" \
#         python src/vlm_llm_ucf_eval.py generate_llm \
#             --vlm_model "$vlm_model" --llm_model "$llm_model" \
#             --prompt_vlm "$prompt_vlm" \
#             --prompt_type "$prompt_type" \
#             --duration_sec 1

##########################################################################################

# atom05: 좋은 비전 모델 -> atom01
# vlm_model='meta-llama/Llama-3.2-11B-Vision-Instruct'
# chat_template='llama_3_vision'
# llm_model='meta-llama/Llama-3.1-8B-Instruct'
# prompt_vlm='Describe the video in a few sentences.'
# prompt_type='per-segment'

##########################################################################################

# atom03,05: 좋은 비전 모델
# vlm_model='meta-llama/Llama-3.2-90B-Vision-Instruct'
# chat_template='llama_3_vision'
# llm_model='meta-llama/Llama-3.1-8B-Instruct'
# prompt_vlm='Describe the video in a few sentences.'
# prompt_type='per-segment'

# docker_image='torch'

# tp_size=16
# world_size=1

# ###

# node_rank=0
# docker run --gpus all -it --name vlm_server --network host --shm-size=32G \
#     -e NCCL_SOCKET_IFNAME=nebula1 -e GLOO_SOCKET_IFNAME=nebula1 -e SGLANG_HOST_IP='10.90.21.21' \
#     -e OUTLINES_CACHE_DIR=/tmp/outlines \
#     --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
#     "${docker_image:-torch}" \
#         python3 -m sglang.launch_server \
#             --nnodes=2 --node-rank=$node_rank \
#             --model-path "${vlm_model:-'lmms-lab/llava-onevision-qwen2-7b-ov'}" \
#             --host='10.90.21.21' --port=50001 \
#             --nccl-init-addr='10.90.21.21:50002' \
#             --tp-size=${tp_size:-4} --dp-size=${world_size:-2} \
#             --chat-template="${chat_template:-'chatml-llava'}" \
#             --mem-fraction-static 0.7 --random-seed 1234 --enable-metrics --disable-cuda-graph

# ###

# node_rank=1
# docker run --gpus all -it --name vlm_server --network host --shm-size=32G \
#     -e NCCL_SOCKET_IFNAME=nebula1 -e GLOO_SOCKET_IFNAME=nebula1 -e SGLANG_HOST_IP='10.90.21.21' \
#     -e OUTLINES_CACHE_DIR=/tmp/outlines \
#     --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
#     "${docker_image:-torch}" \
#         python3 -m sglang.launch_server \
#             --nnodes=2 --node-rank=$node_rank \
#             --model-path "${vlm_model:-'lmms-lab/llava-onevision-qwen2-7b-ov'}" \
#             --host='10.90.21.120' --port=50001 \
#             --nccl-init-addr='10.90.21.21:50002' \
#             --tp-size=${tp_size:-4} --dp-size=${world_size:-2} \
#             --chat-template="${chat_template:-'chatml-llava'}" \
#             --mem-fraction-static 0.7 --random-seed 1234 --enable-metrics --disable-cuda-graph

# ###

# docker run \
#     --rm -d --network host \
#     --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code \
#     --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
#     --mount type=bind,src=/projects3/datasets/UCF_Crimes/,dst=/datasets/UCF_Crimes/ \
#     --name "vlm_worker" \
#     "${docker_image:-torch}" \
#         python src/vlm_llm_ucf_eval.py generate_vlm \
#             --host '10.90.21.21' --port 50001 \
#             --rank ${rank:-0} --world_size ${world_size:-1} \
#             --vlm_model "$vlm_model" --llm_model "$llm_model" \
#             --prompt_vlm "$prompt_vlm" \
#             --prompt_type "$prompt_type" \
#             --duration_sec 1

##########################################################################################

# atom02 -> 68.32
# vlm_model='gpt-4o'
# prompt_vlm="How anomalous is this video? Please rate from 0 to 1 with 0 being not anomalous and 1 being very anomalous and provide an explanation in a few sentences in provided json format."

# docker_image='torch'
# docker run \
#     -d \
#     --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code \
#     --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
#     --mount type=bind,src=/projects3/datasets/UCF_Crimes/,dst=/datasets/UCF_Crimes/ \
#     --name "worker" \
#     "$docker_image" \
#         python src/vlm_llm_ucf_eval.py generate_integrated_parsed \
#             --vlm_model "$vlm_model" \
#             --prompt_vlm "$prompt_vlm" \

##########################################################################################

# atom02 -> 72.00
vlm_model='gpt-4o'
llm_model='gpt-4o'
prompt_vlm='Describe the video in a few sentences.'
prompt_type='per-segment'

docker_image='torch'
docker run \
    -d \
    --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code \
    --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
    --mount type=bind,src=/projects3/datasets/UCF_Crimes/,dst=/datasets/UCF_Crimes/ \
    --name "vlm_worker" \
    "$docker_image" \
        python src/vlm_llm_ucf_eval.py generate_vlm \
            --vlm_model "$vlm_model" \
            --prompt_vlm "$prompt_vlm" \
            --prompt_type "$prompt_type" \
            --duration_sec 1

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
            --prompt_type "$prompt_type" \
            --duration_sec 1

##########################################################################################

# atom05
# vlm_model='lmms-lab/llava-onevision-qwen2-7b-ov'
# llm_model='meta-llama/Llama-3.1-8B-Instruct'
# prompt_vlm='Describe the video in a few sentences.'
# prompt_type='windowing'

##########################################################################################

# atom02 -> 73.87
# vlm_model='gpt-4o'
# prompt_vlm="Does this image contain abnormal events like crimes, accidents? Your response should be only one of yes or no without any explanation."

# docker_image='torch'
# docker run \
#     -d \
#     --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code \
#     --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
#     --mount type=bind,src=/projects3/datasets/UCF_Crimes/,dst=/datasets/UCF_Crimes/ \
#     --name "worker" \
#     "$docker_image" \
#         python src/vlm_llm_ucf_eval.py generate_integrated_binary \
#             --vlm_model "$vlm_model" \
#             --prompt_vlm "$prompt_vlm"

##########################################################################################

# atom05 -> 망함, 똑같은 binary 프롬프트인데 오히려 상세하게 질문하니까 No만 나옴
# vlm_model='lmms-lab/llava-onevision-qwen2-7b-ov'
# prompt_vlm='Are there any unusual movements of the vehicles in the video frames? Are there any unusual behaviors of the people in the video frames? Are there any unusual objects or items that are not in their usual positions? Are there any objects or people that appear out of place in the scene? Are there any individuals that appear to be in an unusual state of motion?'
# chat_template=
# prompt_type=

##########################################################################################

# atom03: gpt 잘 돼서 라바로도 해보기 -> 59.59
# vlm_model='lmms-lab/llava-onevision-qwen2-7b-ov'
# prompt_vlm="Does this image contain abnormal events like crimes, accidents? Your response should be only one of yes or no without any explanation."

# docker_image='torch'
# tp_size=4
# world_size=$(( 8 / $tp_size ))
# for rank in $(seq 0 $(( world_size - 1 ))); do
#     docker run \
#         -d --network my-network \
#         --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code \
#         --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
#         --mount type=bind,src=/projects3/datasets/UCF_Crimes/,dst=/datasets/UCF_Crimes/ \
#         --name "worker$rank" \
#         "$docker_image" \
#             python src/vlm_llm_ucf_eval.py generate_integrated_binary \
#                 --host vlm_server --port 50001 \
#                 --rank ${rank:-0} --world_size ${world_size:-1} \
#                 --vlm_model "$vlm_model" \
#                 --prompt_vlm "$prompt_vlm"
# done

##########################################################################################

# atom03: 앞전거 image에서 video로 바꿈 라바는 비디오 받으니까 -> 56.99
# vlm_model='lmms-lab/llava-onevision-qwen2-7b-ov'
# prompt_vlm="Does this video contain abnormal events like crimes, accidents? Your response should be only one of yes or no without any explanation."

# docker_image='torch'
# tp_size=4
# world_size=$(( 8 / $tp_size ))
# for rank in $(seq 0 $(( world_size - 1 ))); do
#     docker run \
#         -d --network my-network \
#         --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code \
#         --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
#         --mount type=bind,src=/projects3/datasets/UCF_Crimes/,dst=/datasets/UCF_Crimes/ \
#         --name "worker$rank" \
#         "$docker_image" \
#             python src/vlm_llm_ucf_eval.py generate_integrated_binary \
#                 --host vlm_server --port 50001 \
#                 --rank ${rank:-0} --world_size ${world_size:-1} \
#                 --vlm_model "$vlm_model" \
#                 --prompt_vlm "$prompt_vlm"
# done

##########################################################################################

# atom05: 어떻게든 높은 성능 달성 --> OOM
# vlm_model='lmms-lab/llava-onevision-qwen2-72b-ov-chat'
# prompt_vlm="Does this image contain abnormal events like crimes, accidents? Your response should be only one of yes or no without any explanation."

# tp_size=8
# world_size=$(( 8 / $tp_size ))
# docker run --gpus all -d -p 50001:50001 --name vlm_server --network my-network --shm-size=32G \
#     --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
#     "${docker_image:-torch}" \
#         python3 -m sglang_router.launch_server \
#             --model-path "${vlm_model:-lmms-lab/llava-onevision-qwen2-7b-ov}" \
#             --port=50001 \
#             --tp-size=${tp_size:-4} --dp-size=${world_size:-2} \
#             --chat-template="${chat_template:-chatml-llava}" \
#             --host='0.0.0.0' \
#             --disable-overlap-schedule --router-policy round_robin \
#             --mem-fraction-static 0.8 --random-seed 1234 --enable-metrics \
#             --tokenizer-path 'lmms-lab/llavanext-qwen-siglip-tokenizer'

# docker_image='torch'
# tp_size=4
# world_size=$(( 8 / $tp_size ))
# for rank in $(seq 0 $(( world_size - 1 ))); do
#     docker run \
#         -d --network my-network \
#         --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code \
#         --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
#         --mount type=bind,src=/projects3/datasets/UCF_Crimes/,dst=/datasets/UCF_Crimes/ \
#         --name "worker$rank" \
#         "$docker_image" \
#             python src/vlm_llm_ucf_eval.py generate_integrated_binary \
#                 --host vlm_server --port 50001 \
#                 --rank ${rank:-0} --world_size ${world_size:-1} \
#                 --vlm_model "$vlm_model" \
#                 --prompt_vlm "$prompt_vlm"
# done

##########################################################################################

# atom05: 어떻게든 높은 성능 달성 -> 69.52
vlm_model='lmms-lab/llava-onevision-qwen2-7b-ov'
chat_template=
prompt_vlm="Does this video contain any anomalous events or activities? Please provide the category of the action in the video in a single word. Then, decide your final answer: yes or no. Also, the response should be in provided json format.

The categories are: 'abuse', 'arrest', 'arson', 'assault', 'burglary', 'explosion', 'fighting', 'normal', 'robbery', 'shooting', 'shoplifting', 'stealing', 'vandalism'.

Hint: Exaggerated or fast movements are highly likely to be anomalous."

docker_image='torch'
tp_size=4
world_size=$(( 8 / $tp_size ))
for rank in $(seq 0 $(( world_size - 1 ))); do
    docker run \
        -d --network my-network \
        --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code \
        --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
        --mount type=bind,src=/projects3/datasets/UCF_Crimes/,dst=/datasets/UCF_Crimes/ \
        --name "worker$rank" \
        "$docker_image" \
            python src/vlm_llm_ucf_eval.py generate_integrated_parsed_binary \
                --host vlm_server --port 50001 \
                --rank ${rank:-0} --world_size ${world_size:-1} \
                --vlm_model "$vlm_model" \
                --prompt_vlm "$prompt_vlm"
done && docker logs -f worker0

##########################################################################################

# atom05: 어떻게든 높은 성능 달성
vlm_model='Qwen/Qwen2.5-VL-7B-Instruct'
chat_template='qwen2-vl'

##########################################################################################

echo $vlm_model
echo $llm_model
echo $prompt_vlm
echo $prompt_type

docker_image='torch'

# create a network if it does not exist
docker network inspect my-network >/dev/null 2>&1 || docker network create my-network

# VLM
tp_size=4
world_size=$(( 8 / $tp_size ))
docker run --gpus all -d -p 50001:50001 --name vlm_server --network my-network --shm-size=8G \
    --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
    "${docker_image:-torch}" \
        python3 -m sglang_router.launch_server \
            --model-path "${vlm_model:-lmms-lab/llava-onevision-qwen2-7b-ov}" \
            --port=50001 \
            --tp-size=${tp_size:-4} --dp-size=${world_size:-2} \
            --chat-template="${chat_template:-chatml-llava}" \
            --host='0.0.0.0' \
            --disable-overlap-schedule --router-policy round_robin \
            --mem-fraction-static 0.8 --random-seed 1234 --enable-metrics && docker logs -f vlm_server
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
                --host vlm_server --port 50001 \
                --rank ${rank:-0} --world_size ${world_size:-1} \
                --vlm_model "$vlm_model" \
                --prompt_vlm "$prompt_vlm" \
                --duration_sec 1
done && docker logs -f vlm_worker_0
docker wait $(docker ps -q --filter "name=vlm_worker_*")
echo "VLM workers are done."
docker stop vlm_server
echo "VLM server is stopped."

# LLM
tp_size=4
world_size=$(( 8 / $tp_size ))
docker run --gpus all -d -p 50002:50002 --name llm_server --network my-network --shm-size=8G \
    --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
    "${docker_image:-torch}" \
        python3 -m sglang_router.launch_server \
            --model-path "${llm_model:-meta-llama/Llama-3.1-8B-Instruct}" \
            --port=50002 \
            --tp-size=${tp_size:-4} --dp-size=${world_size:-2} \
            --host='0.0.0.0' \
            --disable-overlap-schedule --router-policy round_robin \
            --mem-fraction-static 0.8 --random-seed 1234 --enable-metrics
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
                --port 50002 \
                --rank ${rank:-0} --world_size ${world_size:-1} \
                --vlm_model "$vlm_model" --llm_model "$llm_model" \
                --prompt_vlm "$prompt_vlm" \
                --prompt_type "$prompt_type" \
                --duration_sec 1
done
docker wait $(docker ps -q --filter "name=llm_worker_*")
echo "LLM workers are done."
docker stop llm_server
echo "LLM server is stopped."
