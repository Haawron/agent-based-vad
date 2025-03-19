#!/bin/bash

world_size=$(nvidia-smi -L | wc -l)
for rank in $(seq 0 $(( world_size - 1 ))); do
    docker run \
        --rm -d --gpus all \
        --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code \
        --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
        --mount type=bind,src=/projects3/datasets/UCF_Crimes/,dst=/datasets/UCF_Crimes/ \
        --name "pcworker${rank:-0}" \
        'imagebind' \
            python src/pseudo_caption_infer.py run \
                --rank ${rank:-0} --world_size ${world_size:-1} \
                --segment_duration_sec 1.0 \
                --num_segment_frames 16 \
                --segment_overlap_sec 0.5
done && docker logs -f pcworker0

################################################################

# video feature options
segment_duration_sec=1.0
segment_overlap_sec=0.5
num_sampled_segment_frames=16
# faiss options
num_captions_per_segment=10
anomalous_scale=.8

world_size=$(nvidia-smi -L | wc -l)
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
                --segment_duration_sec $segment_duration_sec \
                --segment_overlap_sec $segment_overlap_sec \
                --num_sampled_segment_frames $num_sampled_segment_frames
done && docker logs -f extract0

docker run \
    --rm -d --gpus all --shm-size=32G \
    --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code \
    --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
    --mount type=bind,src=/projects3/datasets/UCF_Crimes/,dst=/datasets/UCF_Crimes/ \
    --name "match" \
    'imagebind' \
        python src/pseudo_caption_infer.py match_captions_per_segment \
            --segment_duration_sec $segment_duration_sec \
            --segment_overlap_sec $segment_overlap_sec \
            --num_sampled_segment_frames $num_sampled_segment_frames \
            --num_captions_per_segment $num_captions_per_segment \
            --anomalous_scale $anomalous_scale \
&& docker logs -f match
