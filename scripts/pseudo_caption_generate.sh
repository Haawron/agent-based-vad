#!/bin/bash

docker network inspect my-network >/dev/null 2>&1 || docker network create my-network

load_llm_server() {
    local ngpus=$(nvidia-smi -L | wc -l)

    local tp_size="${1:-4}"
    local world_size=$(( $ngpus / $tp_size ))
    local model_path="$2"
    local chat_template="$3"

    docker run --gpus all -d -p 50001:50001 --name llm_server --network my-network --shm-size=8G \
        --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
        'torch' \
            python3 -m sglang_router.launch_server \
                --model-path "$model_path" \
                --port=50001 \
                --tp-size=$tp_size --dp-size=$world_size \
                --chat-template="$chat_template" \
                --host='0.0.0.0' \
                --disable-overlap-schedule --router-policy round_robin \
                --mem-fraction-static 0.8 --random-seed 1234 --enable-metrics && docker logs -f llm_server
}

run_worker() {
    local ngpus=$(nvidia-smi -L | wc -l)

    local tp_size="${1:-4}"
    local world_size=$(( $ngpus / $tp_size ))
    local model_path="$2"
    local exp_name="$3"
    local system_prompt="$4"

    args=()
    if [ -n "$model_path" ]; then
        args+=("--model_name" "$model_path")
    fi
    if [ -n "$exp_name" ]; then
        args+=("--exp_name" "$exp_name")
    fi
    if [ -n "$system_prompt" ]; then
        args+=("--system_prompt" "$system_prompt")
    fi

    for rank in $(seq 0 $(( world_size - 1 ))); do
        docker run \
            --rm -d --network my-network \
            --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code \
            --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
            --mount type=bind,src=/projects3/datasets/UCF_Crimes/,dst=/datasets/UCF_Crimes/ \
            --name "pcg_worker${rank:-0}" \
            torch \
                python src/pseudo_caption_generate.py run \
                    --host llm_server --port 50001 \
                    --rank ${rank:-0} --world_size ${world_size:-1} \
                    "${args[@]}"
    done && docker logs -f pcg_worker0
}

##########################################################################################

# llm_model='google/gemma-3-27b-it'  # 아직 미지원
# load_llm_server 4 $llm_model

###############################################################

llm_model='meta-llama/Llama-3.3-70B-Instruct'
tp_size=8
load_llm_server $tp_size $llm_model
run_worker $tp_size $llm_model


###############################################################

# atom02
llm_model='gpt-4o'
exp_name='01-rich-context-1M'
system_prompt='You are solving the video anomaly detection (VAD) problem in a fancy way. As you know, anomalous events are rare but their categories are diverse. You have to generate example scene descriptions both for the anomalous events and normal events. We will use these descriptions to decide if given video clips contain anomalous events by choosing one of the descriptions having the top similarity measured by a multi-modal retrieval model. The descriptions should be short and concise. The entire response should be in the provided json format.'
num_seeds=10000

docker run \
    --rm -d --network my-network \
    --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code \
    --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
    --mount type=bind,src=/projects3/datasets/UCF_Crimes/,dst=/datasets/UCF_Crimes/ \
    --name "pcg_worker" \
    torch \
        python src/pseudo_caption_generate.py run \
            --rank 0 --world_size 1 \
            --model_name $llm_model --exp_name $exp_name --system_prompt "$system_prompt" --num_seeds $num_seeds \
    && docker logs -f pcg_worker
