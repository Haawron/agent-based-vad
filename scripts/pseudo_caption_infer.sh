#!/bin/bash

###################################

# ImageBind (기본)
# docker options
image_name='imagebind'
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

# 16프레임으로 바꾸면? -> 별 차이 없음
# docker options
image_name='imagebind'
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
# docker options
image_name='imagebind'
# video feature options
caption_type='01-rich-context-1M'
retriever_name='imagebind'
segment_duration_sec=1.0
segment_overlap_sec=0.5
num_sampled_segment_frames=16
# faiss options
num_captions_per_segment=10
anomalous_scale=1.0

######################################################################

# Languagebind
# docker options
image_name='imagebind'
# video feature options
caption_type='00-rich-context'
retriever_name='languagebind'
segment_duration_sec=1.0
segment_overlap_sec=0.5
num_sampled_segment_frames=8
# faiss options
num_captions_per_segment=10
anomalous_scale=1.0

######################################################################

# Internvideo2-1b
# docker options
image_name='internvideo'
# video feature options
caption_type='00-rich-context'
retriever_name='internvideo-1b'
segment_duration_sec=1.0
segment_overlap_sec=0.5
num_sampled_segment_frames=4
# faiss options
num_captions_per_segment=10
anomalous_scale=1.0

###################################

# Internvideo2-1b 그래 프레임을 적게 봐서 구린 거야
# docker options
image_name='internvideo'
# video feature options
caption_type='00-rich-context'
retriever_name='internvideo-1b'
segment_duration_sec=1.0
segment_overlap_sec=0.5
num_sampled_segment_frames=8
# faiss options
num_captions_per_segment=10
anomalous_scale=1.0

###################################

# Internvideo2-1b 그래 프레임을 적게 봐서 구린 거야
# docker options
image_name='internvideo'
# video feature options
caption_type='00-rich-context'
retriever_name='internvideo-1b'
segment_duration_sec=1.0
segment_overlap_sec=0.5
num_sampled_segment_frames=16
# faiss options
num_captions_per_segment=10
anomalous_scale=1.0

###################################

# Internvideo2-6b
# docker options
image_name='internvideo'
# video feature options
caption_type='00-rich-context'
retriever_name='internvideo-6b'
segment_duration_sec=1.0
segment_overlap_sec=0.5
num_sampled_segment_frames=4
# faiss options
num_captions_per_segment=10
anomalous_scale=1.0

###################################

# siglip
# docker options
image_name='torch'
# video feature options
caption_type='00-rich-context'
retriever_name='google/siglip-so400m-patch14-384'
segment_duration_sec=1.0
segment_overlap_sec=0.5
num_sampled_segment_frames=8
# faiss options
num_captions_per_segment=10
anomalous_scale=1.0

####################################################################################

# segment 임베딩 뽑기
world_size=$(nvidia-smi -L | wc -l) &&\
for rank in $(seq 0 $(( world_size - 1 ))); do
    docker run \
        --rm -d --gpus all --shm-size=40G \
        --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code \
        --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
        --mount type=bind,src=/projects3/datasets/UCF_Crimes/,dst=/datasets/UCF_Crimes/ \
        --name "extract${rank:-0}" \
        "$image_name" \
            python src/pseudo_caption_infer.py extract_embeddings_per_segment \
                --rank ${rank:-0} --world_size ${world_size:-1} \
                --caption_type $caption_type \
                --retriever_name $retriever_name \
                --segment_duration_sec $segment_duration_sec \
                --segment_overlap_sec $segment_overlap_sec \
                --num_sampled_segment_frames $num_sampled_segment_frames
done && docker logs -f extract0

# 캡션 임베딩 뽑기
world_size=$(nvidia-smi -L | wc -l) &&\
for rank in $(seq 0 $(( world_size - 1 ))); do
    docker run \
        --rm -d --gpus all --shm-size=40G \
        --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code \
        --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
        --mount type=bind,src=/projects3/datasets/UCF_Crimes/,dst=/datasets/UCF_Crimes/ \
        --name "extcap${rank:-0}" \
        "$image_name" \
            python src/pseudo_caption_infer.py extract_caption_embeddings \
                --rank ${rank:-0} --world_size ${world_size:-1} \
                --caption_type $caption_type \
                --retriever_name $retriever_name &
done && wait && docker logs -f extcap0

# scoring 하기
rm -rf /code/output/psuedo-captions/gpt-4o/00-rich-context/scored && \
world_size=16 && \
for rank in $(seq 0 $(( world_size - 1 ))); do
    docker run \
        --rm -d --shm-size=40G \
        --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code \
        --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
        --mount type=bind,src=/projects3/datasets/UCF_Crimes/,dst=/datasets/UCF_Crimes/ \
        --name "match$rank" \
        "$image_name" \
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
    --rm -d --gpus all --shm-size=40G \
    --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code \
    --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
    --mount type=bind,src=/projects3/datasets/UCF_Crimes/,dst=/datasets/UCF_Crimes/ \
    --name "exttmf" \
        "$image_name" \
        python src/pseudo_caption_infer.py extract_tmf \
            --num_sampled_frames 64 \
&& docker logs -f exttmf

# Extract TMF Embeddings
docker run \
    --rm -d --gpus all --shm-size=40G \
    --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code \
    --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
    --mount type=bind,src=/projects3/datasets/UCF_Crimes/,dst=/datasets/UCF_Crimes/ \
    --name "exttmfe" \
        "$image_name" \
        python src/pseudo_caption_infer.py extract_tmf_embeddings \
            --num_tmf_frames 64 \
            --retriever_name $retriever_name \
&& docker logs -f exttmfe
