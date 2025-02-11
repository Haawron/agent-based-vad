#!/bin/bash

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

docker_image='torch'
containers=$(docker ps -q --filter 'ancestor=torch'); [ -n "$containers" ] && docker wait $containers
echo 'No containers running.'
sleep 5
world_size=8
vlm_model='lmms-lab/llava-onevision-qwen2-7b-ov'
llm_model='deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
prompt_vlm='Describe the video in detail.'
prompt_llm_system_language='en'
for rank in {0..7}; do
    docker run \
        --gpus "device=$rank" --rm -d \
        --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code \
        --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
        --mount type=bind,src=/projects3/datasets/UCF_Crimes/,dst=/datasets/UCF_Crimes/ \
         "$docker_image" \
            bash scripts/vlm_llm_ucf_eval.sh $rank $world_size $vlm_model $llm_model "$prompt_vlm" $prompt_llm_system_language
done
