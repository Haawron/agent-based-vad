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
# eval options
num_captions_per_segment=10
anomalous_scale=1.0
prompt_type='default'

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
# eval options
num_captions_per_segment=10
anomalous_scale=1.0
prompt_type='default'

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
# eval options
num_captions_per_segment=10
anomalous_scale=1.0
prompt_type='default'

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
# eval options
num_captions_per_segment=10
anomalous_scale=1.0
prompt_type='default'

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
# eval options
num_captions_per_segment=10
anomalous_scale=1.0
prompt_type='default'

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
# eval options
num_captions_per_segment=10
anomalous_scale=1.0
prompt_type='default'

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
# eval options
num_captions_per_segment=10
anomalous_scale=1.0
prompt_type='default'

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
# eval options
num_captions_per_segment=10
anomalous_scale=1.0
prompt_type='default'

######################################################################

# siglip 224
# docker options
image_name='torch'
# video feature options
caption_type='00-rich-context'
retriever_name='google/siglip-so400m-patch14-224'
segment_duration_sec=1.0
segment_overlap_sec=0.5
num_sampled_segment_frames=8
# eval options
num_captions_per_segment=10
anomalous_scale=1.0
prompt_type='default'

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
# eval options
num_captions_per_segment=10
anomalous_scale=1.0
prompt_type='default'

###################################

# siglip 16 프레임
# docker options
image_name='torch'
# video feature options
caption_type='00-rich-context'
retriever_name='google/siglip-so400m-patch14-384'
segment_duration_sec=1.0
segment_overlap_sec=0.5
num_sampled_segment_frames=16
# eval options
num_captions_per_segment=10
anomalous_scale=1.0
prompt_type='default'

###################################

# siglip2
# docker options
image_name='internvideo'
# video feature options
caption_type='00-rich-context'
retriever_name='google/siglip2-so400m-patch14-384'
segment_duration_sec=1.0
segment_overlap_sec=0.5
num_sampled_segment_frames=8
# eval options
num_captions_per_segment=10
anomalous_scale=1.0
prompt_type='default'

###################################

# siglip2 16 프레임
# docker options
image_name='internvideo'
# video feature options
caption_type='00-rich-context'
retriever_name='google/siglip2-so400m-patch14-384'
segment_duration_sec=1.0
segment_overlap_sec=0.5
num_sampled_segment_frames=16
# eval options
num_captions_per_segment=10
anomalous_scale=1.0
prompt_type='default'

######################################################################

# Perception model
# docker options
image_name='pe'
# video feature options
caption_type='00-rich-context'
retriever_name='facebook/PE-Core-L14-336'
segment_duration_sec=1.0
segment_overlap_sec=0.5
num_sampled_segment_frames=16
# eval options
num_captions_per_segment=10
anomalous_scale=1.0
prompt_type='default'

###################################

# Perception model G
# docker options
image_name='pe'
# video feature options
caption_type='00-rich-context'
retriever_name='facebook/PE-Core-G14-448'
segment_duration_sec=1.0
segment_overlap_sec=0.5
num_sampled_segment_frames=16
# eval options
num_captions_per_segment=10
anomalous_scale=1.0
prompt_type='default'

####################################################################################

# segment 임베딩 뽑기
world_size=$(nvidia-smi -L | wc -l) &&\
for rank in $(seq 0 $(( world_size - 1 ))); do
    docker run \
        --rm -d --gpus all --shm-size=40G \
        --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code \
        --mount type=bind,src=$HOME/.cache,dst=/home/hglee/.cache \
        --mount type=bind,src=/projects3/datasets/UCF_Crimes/,dst=/datasets/UCF_Crimes/ \
        --mount type=bind,src=/datalake/share/datasets/xd/XD-Violence,dst=/datasets/XD-Violence/ \
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
        --mount type=bind,src=/datalake/share/datasets/xd/XD-Violence,dst=/datasets/XD-Violence/ \
        --name "extcap${rank:-0}" \
        "$image_name" \
            python src/pseudo_caption_infer.py extract_caption_embeddings \
                --rank ${rank:-0} --world_size ${world_size:-1} \
                --caption_type $caption_type \
                --retriever_name $retriever_name \
                --prompt_type $prompt_type &
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
        --mount type=bind,src=/datalake/share/datasets/xd/XD-Violence,dst=/datasets/XD-Violence/ \
        --name "match$rank" \
        "$image_name" \
            python src/pseudo_caption_infer.py match_captions_per_segment \
                --rank ${rank:-0} --world_size ${world_size:-1} \
                --caption_type $caption_type \
                --retriever_name $retriever_name \
                --prompt_type $prompt_type \
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
    --mount type=bind,src=/datalake/share/datasets/xd/XD-Violence,dst=/datasets/XD-Violence/ \
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
    --mount type=bind,src=/datalake/share/datasets/xd/XD-Violence,dst=/datasets/XD-Violence/ \
    --name "exttmfe" \
        "$image_name" \
        python src/pseudo_caption_infer.py extract_tmf_embeddings \
            --num_tmf_frames 64 \
            --retriever_name $retriever_name \
&& docker logs -f exttmfe
