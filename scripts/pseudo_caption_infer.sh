#!/bin/bash

###################################

# video feature options
caption_type='00-rich-context'
retriever_name='imagebind'
segment_duration_sec=1.0
segment_overlap_sec=0.5
num_sampled_segment_frames=16
# faiss options
num_captions_per_segment=10
anomalous_scale=1.0

###################################

# video feature options
caption_type='00-rich-context'
retriever_name='languagebind'
segment_duration_sec=1.0
segment_overlap_sec=0.5
num_sampled_segment_frames=8
# faiss options
num_captions_per_segment=10
anomalous_scale=1.0

###################################

# 16프레임으로 바꾸면? -> 별 차이 없음
# video feature options
caption_type='00-rich-context'
retriever_name='languagebind'
segment_duration_sec=1.0
segment_overlap_sec=0.5
num_sampled_segment_frames=16
# faiss options
num_captions_per_segment=10
anomalous_scale=1.0

###################################

# 1M으로 바꾸면?
# video feature options
caption_type='01-rich-context-1M'
retriever_name='imagebind'
segment_duration_sec=1.0
segment_overlap_sec=0.5
num_sampled_segment_frames=16
# faiss options
num_captions_per_segment=10
anomalous_scale=1.0

####################################################################################


world_size=$(nvidia-smi -L | wc -l) &&\
for rank in $(seq 0 $(( world_size - 1 ))); do
    docker run \
        --rm -d --gpus all --shm-size=32G \
        --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code \
        --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
        --mount type=bind,src=/projects3/datasets/UCF_Crimes/,dst=/datasets/UCF_Crimes/ \
        --name "extract${rank:-0}" \
        'imagebind' \
            python src/pseudo_caption_infer.py extract_embeddings_per_segment \
                --rank ${rank:-0} --world_size ${world_size:-1} \
                --caption_type $caption_type \
                --retriever_name $retriever_name \
                --segment_duration_sec $segment_duration_sec \
                --segment_overlap_sec $segment_overlap_sec \
                --num_sampled_segment_frames $num_sampled_segment_frames
done && docker logs -f extract0


world_size=$(nvidia-smi -L | wc -l) &&\
for rank in $(seq 0 $(( world_size - 1 ))); do
    docker run \
        --rm -d --gpus all --shm-size=32G \
        --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code \
        --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
        --mount type=bind,src=/projects3/datasets/UCF_Crimes/,dst=/datasets/UCF_Crimes/ \
        --name "extcap${rank:-0}" \
        'imagebind' \
            python src/pseudo_caption_infer.py extract_caption_embeddings \
                --rank ${rank:-0} --world_size ${world_size:-1} \
                --caption_type $caption_type \
                --retriever_name $retriever_name
done && docker logs -f extcap0


rm -rf /code/output/psuedo-captions/gpt-4o/00-rich-context/scored &&\
world_size=16 &&\
for rank in $(seq 0 $(( world_size - 1 ))); do
    docker run \
        --rm -d --gpus all --shm-size=32G \
        --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code \
        --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
        --mount type=bind,src=/projects3/datasets/UCF_Crimes/,dst=/datasets/UCF_Crimes/ \
        --name "match$rank" \
        'imagebind' \
            python src/pseudo_caption_infer.py match_captions_per_segment \
                --rank ${rank:-0} --world_size ${world_size:-1} \
                --caption_type $caption_type \
                --retriever_name $retriever_name \
                --segment_duration_sec $segment_duration_sec \
                --segment_overlap_sec $segment_overlap_sec \
                --num_sampled_segment_frames $num_sampled_segment_frames \
                --num_captions_per_segment $num_captions_per_segment \
                --anomalous_scale $anomalous_scale &
done && wait && docker logs -f match4 && docker logs -f match0

####################################################################################

# Extract TMF
docker run \
    --rm -d --gpus all --shm-size=32G \
    --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code \
    --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
    --mount type=bind,src=/projects3/datasets/UCF_Crimes/,dst=/datasets/UCF_Crimes/ \
    --name "exttmf" \
    'imagebind' \
        python src/pseudo_caption_infer.py extract_tmf \
            --num_sampled_frames 32 \
&& docker logs -f exttmf

docker run \
    --rm -d --gpus all --shm-size=32G \
    --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code \
    --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
    --mount type=bind,src=/projects3/datasets/UCF_Crimes/,dst=/datasets/UCF_Crimes/ \
    --name "exttmfe" \
    'imagebind' \
        python src/pseudo_caption_infer.py extract_tmf_embeddings \
            --num_tmf_frames 32 \
&& docker logs -f exttmfe
