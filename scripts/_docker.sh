DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile -t torch --build-arg UID=$(id -u) --build-arg GID=$(id -g) .

docker save -o torch.tar torch
docker load -i torch.tar

docker run \
    --gpus all --rm -it \
    --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code \
    --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
    --mount type=bind,src=/projects3/datasets/UCF_Crimes/,dst=/datasets/UCF_Crimes/ \
     torch

python src/vlm_llm_ucf_eval.py generate \
    --llm_model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --port 30001 \
    --duration_sec 1


container_id=$(docker run \
    --gpus all --rm -d \
    --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code \
    --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
    --mount type=bind,src=/projects3/datasets/UCF_Crimes/,dst=/datasets/UCF_Crimes/ \
     torch \
        python src/vlm_llm_ucf_eval.py generate \
            --llm_model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
            --port 30001 \
            --duration_sec 1
)
docker logs -f $container_id


##########################################################################################

docker_image='torch'
containers=$(docker ps -q --filter 'ancestor=torch'); [ -n "$containers" ] && docker wait $containers
world_size=8
vlm_model='lmms-lab/llava-onevision-qwen2-7b-ov'
llm_model='deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
prompt_llm_system_language='en'
for rank in {0..7}; do
    docker run \
        --gpus "device=$rank" --rm -d \
        --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code \
        --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
        --mount type=bind,src=/projects3/datasets/UCF_Crimes/,dst=/datasets/UCF_Crimes/ \
         "$docker_image" \
            bash scripts/vlm_llm_ucf_eval.sh $rank $world_size $vlm_model $llm_model $prompt_llm_system_language
done

##########################################################################################

docker_image='torch'
containers=$(docker ps -q --filter 'ancestor=torch'); [ -n "$containers" ] && docker wait $containers
world_size=8
vlm_model='lmms-lab/llava-onevision-qwen2-7b-ov'
llm_model='lightblue/DeepSeek-R1-Distill-Qwen-7B-Multilingual'
prompt_llm_system_language='en'
for rank in {0..7}; do
    docker run \
        --gpus "device=$rank" --rm -d \
        --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code \
        --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
        --mount type=bind,src=/projects3/datasets/UCF_Crimes/,dst=/datasets/UCF_Crimes/ \
         "$docker_image" \
            bash scripts/vlm_llm_ucf_eval.sh $rank $world_size $vlm_model $llm_model $prompt_llm_system_language
done
