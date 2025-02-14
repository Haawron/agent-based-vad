from dotenv import load_dotenv
load_dotenv()

from typing import Literal
import json
import base64
import io
import os
import re
import sys
import time
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from decord import VideoReader, cpu
from PIL import Image
from tqdm.auto import tqdm

import openai
from sglang.utils import (
    execute_shell_command,
    wait_for_server,
    terminate_process,
)


# model names in a single line are aliases for the same model
OPENAI_MODELS = [
    'gpt-4o', 'chatgpt-4o-latest',
    'gpt-4o-mini',
    'o1',
    'o1-mini',
    'o3-mini',
    'gpt-4-turbo',
]


def get_frames(
    p_video,
    duration_sec = 2,
    max_frames_num = 32,
):
    FPS = 30
    num_frames_segment = int(duration_sec * FPS)

    vr = VideoReader(str(p_video), ctx=cpu(0))
    total_frame_num = len(vr)
    num_segments = total_frame_num // num_frames_segment
    for segment_idx in range(num_segments):
        segment_start_idx = segment_idx * num_frames_segment
        segment_end_idx = segment_start_idx + num_frames_segment - 1
        uniform_sampled_frames = np.linspace(segment_start_idx, segment_end_idx, max_frames_num + 2, dtype=int)[1:-1]
        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()
        yield {
            'frames': frames,
            'segment_idx': segment_idx,
            'total_segments': num_segments,
            'segment_start_idx': segment_start_idx,
            'segment_end_idx': segment_end_idx,
        }


def generate_segment_caption(client: openai.Client, frames, user_prompt: str, model: str = "default"):
    base64_frames = []
    for frame in frames:
        pil_img = Image.fromarray(frame)
        buff = io.BytesIO()
        pil_img.save(buff, format="JPEG")
        base64_str = base64.b64encode(buff.getvalue()).decode("utf-8")
        base64_frames.append(base64_str)

    content = []
    if len(base64_frames) > 1:
        for base64_frame in base64_frames:
            frame_format = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_frame}"},
                "modalities": "video",
            }
            content.append(frame_format)
        content.append({
            'type': 'text',
            'text': user_prompt,
        })
    elif len(base64_frames) == 1:  # ChatGPT does not support video (actually it does, but it's traditional ML-based)
        content.append({
            'type': 'text',
            'text': user_prompt,
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_frames[0]}"},
        })

    messages = [{"role": "user", "content": content}]

    request = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=1024,
    )
    if request.choices[0].message.content:
        response = request.choices[0].message.content
    elif request.choices[0].message.refusal:
        response = request.choices[0].message.refusal
    return response


def chat(
    client: openai.Client,
    system_prompt: str,
    user_prompt: str,
    model: str = "default",
):
    messages = [
        {
            'role': 'system',
            'content': system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        }
    ]
    while True:
        try:
            request = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=1024,
            )
        except openai.OpenAIError as e:
            print(e)
            print('Retrying in 5 seconds...')
            time.sleep(5)
            continue
        else:
            break
    if request.choices[0].message.content:
        response = request.choices[0].message.content
    elif request.choices[0].message.refusal:
        response = request.choices[0].message.refusal
    return response


class Main:
    def __init__(self):
        self.p_dataroot = Path('/datasets/UCF_Crimes')
        self.p_annroot = Path('./data/annotations')
        self.p_videos_root = self.p_dataroot / 'Videos'
        self.p_ann_test = self.p_annroot / 'Temporal_Anomaly_Annotation_for_Testing_Videos.txt'
        self.p_num_frames = self.p_annroot / 'num_frames_per_video.txt'

    def generate_vlm(
        self,
        host: str = 'localhost', port: int = 30001,
        rank: int = 0, world_size: int = 1,
        vlm_model: str = 'lmms-lab/llava-onevision-qwen2-7b-ov',
        llm_model: str = 'meta-llama/Llama-3.2-3B-Instruct',
        prompt_vlm: str = "Describe the video in a few sentences.",
        prompt_llm_system_language: Literal['en', 'ko'] = "en",
        duration_sec = 1,
        debug: bool = False,
    ):
        p_vlm_outdir = Path('output/ucf-crime-captions') / f"prompt={prompt_vlm.replace(' ', '_')}_duration_{duration_sec}s/raw"
        df_ann_test = pd.read_csv(
            self.p_ann_test, sep=r'\s+', header=None, names=['video', 'label', 's1', 'e1', 's2', 'e2'])
        p_vlm_outdir.mkdir(exist_ok=True, parents=True)
        p_llm_outdir = p_vlm_outdir.parent / llm_model.replace('/', '-') / prompt_llm_system_language
        p_llm_outdir.mkdir(exist_ok=True, parents=True)

        df_ann_test = df_ann_test.iloc[rank::world_size]

        ######################################################################

        if vlm_model not in OPENAI_MODELS:
            server_address = f"http://{host}:{port}"
            print(f'Waiting for VLM server at {server_address}...', flush=True)
            wait_for_server(server_address, timeout=600)
            client = openai.Client(api_key="EMPTY", base_url=f"{server_address}/v1")
        else:
            print('Using OpenAI API', flush=True)
            client = openai.Client(api_key=os.environ.get('OPENAI_API_KEY'))

        for idx, row in tqdm(
            df_ann_test.iterrows(), total=len(df_ann_test), mininterval=1, file=sys.stdout,
        ):
            p_json = (p_vlm_outdir / row['label'] / row['video']).with_suffix('.json')
            if p_json.exists():
                print(f"Skipping {p_json}", flush=True)
                continue
            else:
                p_json.parent.mkdir(exist_ok=True, parents=True)
            label = row['label'] if row['label'] != 'Normal' else 'Testing_Normal_Videos_Anomaly'
            p_anom_video = self.p_videos_root / label / row['video']
            print(f'\nProcessing {p_anom_video}\n\t-> {p_json}', flush=True)
            response_records = []
            for frame_dict in get_frames(
                p_anom_video,
                duration_sec=duration_sec,
            ):
                vlm_response: str = generate_segment_caption(
                    client, frame_dict['frames'], user_prompt=prompt_vlm, model=vlm_model)
                response_record = {
                    'segment_idx': frame_dict['segment_idx'],
                    'start_idx': frame_dict['segment_start_idx'],
                    'end_idx': frame_dict['segment_end_idx'],
                    'response': vlm_response,
                    'score_raw': None,  # placeholder for LLM response
                    'score': None,  # placeholder for LLM response
                }
                if debug:
                    print(f'[{frame_dict["segment_idx"]}/{frame_dict["total_segments"]}] {p_anom_video} -> {p_json}', flush=True)
                    print(response_record, flush=True, end='\n\n')
                response_records.append(response_record)
            video_record = {
                'video': row['video'],  # e.g. 'Abuse001_x264.mp4'
                'label': row['label'],  # e.g. 'Abuse', 'Normal', ...
                's1': row['s1'],
                'e1': row['e1'],
                's2': row['s2'],
                'e2': row['e2'],
                'response_records': response_records,
            }
            print('Saving ...', flush=True)
            json.dump(video_record, p_json.open('w'), indent=2)

    def generate_llm(
        self,
        host: str = 'localhost', port: int = 30001,
        rank: int = 0, world_size: int = 1,
        vlm_model: str = 'lmms-lab/llava-onevision-qwen2-7b-ov',
        llm_model: str = 'meta-llama/Llama-3.2-3B-Instruct',
        prompt_vlm: str = "Describe the video in a few sentences.",
        prompt_llm_system_language: Literal['en', 'ko'] = "en",
        duration_sec = 1,
        debug: bool = False,
    ):
        p_vlm_outdir = Path('output/ucf-crime-captions') / f"prompt={prompt_vlm.replace(' ', '_')}_duration_{duration_sec}s/raw"
        df_ann_test = pd.read_csv(
            self.p_ann_test, sep=r'\s+', header=None, names=['video', 'label', 's1', 'e1', 's2', 'e2'])
        p_llm_outdir = p_vlm_outdir.parent / llm_model.replace('/', '-') / prompt_llm_system_language
        p_llm_outdir.mkdir(exist_ok=True, parents=True)

        df_ann_test = df_ann_test.iloc[rank::world_size]

        if llm_model not in OPENAI_MODELS:
            server_address = f"http://{host}:{port}"
            print(f'Waiting for LLM server at {server_address}...', flush=True)
            wait_for_server(server_address, timeout=600)
            client = openai.Client(api_key="EMPTY", base_url=f"{server_address}/v1")
        else:
            print('Using OpenAI API', flush=True)
            client = openai.Client(api_key=os.environ.get('OPENAI_API_KEY'))

        if prompt_llm_system_language == 'en':
            context_prompt = "The following is a scene description. If you were a law enforcement agency, how would you rate this scene on a scale from 0 to 1, with 0 representing a standard scene and 1 denoting a scene with suspicious activities?"
            format_prompt = "Please provide the response in the form of a Python list and respond with only one number in the provided list below [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] without any textual explanation. It should begin with '[' and end with  ']'."
        elif prompt_llm_system_language == 'ko':
            context_prompt = "다음은 비디오에 대한 설명이야. 만약 너가 경찰이라면 이 장면의 이상한 정도를 0부터 1까지의 점수 중에 어떻게 평가할 거야? 0은 일상적인 장면이고 1은 의심스러운 활동이 있는 장면이야."
            format_prompt = "아래에 제공된 목록에서 하나의 숫자로만 응답해줘. [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] 이 숫자는 어떤 설명도 없이 제공되어야 해. '['로 시작해서 ']'로 끝나야 해."

        system_prompt = f'{context_prompt} {format_prompt}'

        for idx, row in tqdm(
            df_ann_test.iterrows(), total=len(df_ann_test), mininterval=1, position=0, file=sys.stdout
        ):
            p_json = (p_vlm_outdir / row['label'] / row['video']).with_suffix('.json')
            if not p_json.exists():
                print(f"Skipping generating llm captions of {p_json} as it does not exist", flush=True)
                continue
            video_record = json.load(p_json.open())
            p_json_new = (p_llm_outdir / video_record['label'] / video_record['video']).with_suffix('.json')
            if p_json_new.exists():
                print(f"Skipping {p_json_new}", flush=True)
                continue
            print(f'\nProcessing {p_json}\n\t-> {p_json_new}', flush=True)
            p_json_new.parent.mkdir(exist_ok=True, parents=True)
            for response_record in video_record['response_records']:
                user_prompt = f"Scene Description: {response_record['response']}"
                llm_response: str = chat(
                    client,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=llm_model,
                )
                response_record['score_raw'] = llm_response
                if 'DeepSeek-R1' in llm_model:
                    llm_response = re.sub(r'^(?s:.)*</think>', '', llm_response).strip()
                try:
                    score = eval(llm_response)[0]
                except Exception as e:
                    print(e, file=sys.stderr)
                    print(response_record, llm_response, file=sys.stderr)
                    score = None
                finally:
                    response_record['score'] = score
                if debug:
                    print(response_record, flush=True, end='\n\n')
            json.dump(video_record, p_json_new.open('w'), indent=2)

    def evaluate(
        self,
        rank: int = 0,
        llm_model: str = 'meta-llama/Llama-3.2-3B-Instruct',
        prompt_vlm: str = "Describe the video in a few sentences.",
        prompt_llm_system_language: Literal['en', 'ko'] = "en",
        duration_sec = 1,
    ):
        if rank != 0:
            print(f'Skipping rank={rank} != 0 for evaluation')
            return

        p_vlm_outdir = Path('output/ucf-crime-captions') / f"prompt={prompt_vlm.replace(' ', '_')}_duration_{duration_sec}s/raw"
        p_vlm_outdir.mkdir(exist_ok=True, parents=True)
        p_llm_outdir = p_vlm_outdir.parent / llm_model.replace('/', '-') / prompt_llm_system_language
        p_llm_outdir.mkdir(exist_ok=True, parents=True)

        df_ann_test = pd.read_csv(
            self.p_ann_test, sep=r'\s+', header=None, names=['video', 'label', 's1', 'e1', 's2', 'e2'])

        sr_num_frames = pd.read_csv(self.p_num_frames, sep=r'\s+', header=None, names=['video', 'num_frames']).set_index('video')['num_frames']
        ann_vad = {}
        for idx, row in df_ann_test.iterrows():
            key = f"{row['label']}/{row['video']}"
            num_frames = sr_num_frames[key.replace('Normal/', 'Testing_Normal_Videos_Anomaly/')]
            bin_label = np.zeros(num_frames, dtype=np.int32)
            bin_label[row['s1']:row['e1']] = 1
            if row['s2'] != -1:
                bin_label[row['s2']:row['e2']] = 1
            ann_vad[key] = bin_label

        preds = {}
        for key, bin_label in ann_vad.items():
            p_json = (p_llm_outdir / key).with_suffix('.json')
            video_record = json.load(p_json.open())
            records = video_record['response_records']
            pred = np.zeros(len(bin_label))
            for record in records:
                pred[record['start_idx']:record['end_idx']+1] = record['score'] or 0.  # fill None with 0
            preds[key] = pred

        # compute AUC
        all_preds, all_labels = np.array([]), np.array([])
        for key, bin_label in ann_vad.items():
            all_preds = np.concatenate([all_preds, preds[key]])
            all_labels = np.concatenate([all_labels, bin_label])
        auc = roc_auc_score(all_labels.astype(int), all_preds)
        print(auc)


if __name__ == '__main__':
    import fire
    fire.Fire(Main)
