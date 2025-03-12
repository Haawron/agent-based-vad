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
            python src/pseudo_caption.py run \
                --rank ${rank:-0} --world_size ${world_size:-1} \
                --segment_duration_sec 1.0 \
                --num_segment_frames 16 \
                --segment_overlap_sec 0.5
done && docker logs -f pcworker0
