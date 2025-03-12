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
import textwrap
import traceback

from pydantic import BaseModel, Field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm

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

VLM_RSEPONSE_TYPES = Literal['caption', 'parsed', 'binary', 'parsed_binary']


def get_client(
    host: str,
    port: int,
    model_name: str,
) -> openai.Client:
    if model_name in OPENAI_MODELS:
        print('Using OpenAI API', flush=True)
        client = openai.Client(api_key=os.environ.get('OPENAI_API_KEY'))
    else:
        server_address = f"http://{host}:{port}"
        print(f'Waiting for LLM server at {server_address}...', flush=True)
        wait_for_server(server_address, timeout=600)
        client = openai.Client(api_key="EMPTY", base_url=f"{server_address}/v1")
    return client


def get_outdir(
    vlm_model: str,
    llm_model: str | None,
    prompt_vlm: str,
    prompt_type: str | None,
    duration_sec: int,
) -> Path:
    prmpt_vlm_polished = prompt_vlm.replace(' ', '_').replace('\n', '_').replace('\t', '_')[:200]
    p_outdir = (
        Path('output/ucf-crime-captions') /
        vlm_model.replace('/', '-') /
        f"prompt={prmpt_vlm_polished}_duration_{duration_sec}s"
    )
    if llm_model:
        p_outdir /= llm_model.replace('/', '-')
        assert prompt_type is not None
        p_outdir /= prompt_type
    else:
        p_outdir /= 'raw'
    return p_outdir


def get_segments(
    p_video: str | Path,
    segment_duration_sec: float = 1.,
    num_segment_frames = 32,
    segment_overlap_sec: float = 0.,
    num_skip_first_segments: int = 0,
):
    FPS = 30
    num_frames_segment = int(segment_duration_sec * FPS)
    num_frames_overlap = int(segment_overlap_sec * FPS)

    vr = VideoReader(str(p_video), ctx=cpu(0))
    total_frame_num = len(vr)
    num_segments = (total_frame_num - num_frames_segment) // (num_frames_segment - num_frames_overlap) + 1  # Discard last segment if not enough frames
    for segment_idx in range(num_segments):
        if num_skip_first_segments > 0 and segment_idx < num_skip_first_segments:
            continue
        segment_start_idx = segment_idx * (num_frames_segment - num_frames_overlap)
        segment_end_idx = segment_start_idx + num_frames_segment - 1
        uniform_sampled_frames = np.linspace(segment_start_idx, segment_end_idx, num_segment_frames + 2, dtype=int)[1:-1]
        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()
        frame_dict = {
            'frames': frames,
            'segment_idx': segment_idx,
            'total_segments': num_segments,
            'segment_start_idx': segment_start_idx,
            'segment_end_idx': segment_end_idx,
        }
        yield frame_dict


def frames_to_base64(frames) -> list[str]:
    img_list = []
    for frame in frames:
        img_byte_arr = io.BytesIO()
        Image.fromarray(frame).save(img_byte_arr, format='JPEG')
        img_list.append(base64.b64encode(img_byte_arr.getvalue()).decode('utf-8'))
    return img_list


def generate_segment_caption(
    client: openai.Client,
    base64_frames,
    user_prompt: str,
    model: str = "default",
    response_format: None | BaseModel = None,
    seed: int = 1234,
):
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

    def vlm_chat():
        if response_format is None:
            request = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_completion_tokens=1024,
                seed=seed,
            )
        else:
            if model in OPENAI_MODELS:
                request = client.beta.chat.completions.parse(
                    model=model,
                    messages=messages,
                    temperature=0,
                    max_completion_tokens=1024,
                    seed=seed,
                    response_format=response_format,
                )
            else:
                request = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                    max_completion_tokens=1024,
                    seed=seed,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "foo",
                            "schema": response_format.model_json_schema(),
                        },
                    },
                    timeout=10,
                )
        return request

    current_try, max_tries = 0, 5
    while current_try < max_tries:
        try:
            request = vlm_chat()
        except openai.APITimeoutError as e:
            print(e, file=sys.stderr)
            print('Retrying in 5 seconds...', file=sys.stderr, flush=True)
            time.sleep(5)
            current_try += .1  # more tries for timeout
            continue
        except openai.OpenAIError as e:
            traceback.print_exc()
            print('Retrying in 5 seconds...', file=sys.stderr, flush=True)
            time.sleep(5)
            current_try += 1
            continue
        else:  # no exception
            break
    else:  # exhausted all tries
        request = None

    if request.choices[0].message.content:
        response = request.choices[0].message.content
    elif request.choices[0].message.refusal:
        response = request.choices[0].message.refusal
    return response


def chat(
    client: openai.Client,
    messages: list[dict],
    model: str = "default",
    temperature: float = 0,
    max_completion_tokens: int | bool = 1024,
    max_tries = 5,
):
    current_try = 0
    while current_try < max_tries:
        try:
            request = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
        except openai.OpenAIError as e:
            print(e, file=sys.stderr)
            print('Retrying in 5 seconds...', file=sys.stderr, flush=True)
            time.sleep(5)
            current_try += 1
            continue
        else:
            break
    if current_try == max_tries:
        print('Failed to get a response from the model', file=sys.stderr)
        exit(1)

    if request.choices[0].message.content:
        response = request.choices[0].message.content
    elif request.choices[0].message.refusal:
        response = request.choices[0].message.refusal
    else:
        response = str(request.choices[0])
    return response


class AnomalyScoreWithExplanation(BaseModel):
    anomaly_score: float = Field(..., title="Anomaly Score", description="Anomaly score of the input text")
    explanation: str = Field(..., title="Explanation", description="Explanation of the anomaly score")


class EventOrActionThenYesOrNoAnomalyScore(BaseModel):
    event_or_activity: str = Field(..., title="Event or Activity", description="Event or activity that is happening in the video in a single word.")
    is_anomalous: str = Field(..., title="Anomalous?", description="Your final answer: 'yes' or 'no'")


class Main:
    def __init__(self):
        self.p_dataroot = Path('/datasets/UCF_Crimes')
        self.p_annroot = Path('./data/annotations')
        self.p_videos_root = self.p_dataroot / 'Videos'
        self.p_ann_test = self.p_annroot / 'Temporal_Anomaly_Annotation_for_Testing_Videos.txt'
        self.p_num_frames = self.p_annroot / 'num_frames_per_video.txt'

        self.df_ann_test = pd.read_csv(
            self.p_ann_test, sep=r'\s+', header=None, names=['video', 'label', 's1', 'e1', 's2', 'e2'])
        video_keys = self.df_ann_test['label'].str.replace('Normal', 'Testing_Normal_Videos_Anomaly').str.cat(self.df_ann_test['video'], sep='/')
        sr_num_frames = pd.read_csv(
            self.p_num_frames, sep=r'\s+', header=None, names=['video', 'num_frames']).set_index('video')['num_frames']
        self.sr_num_frames = sr_num_frames.loc[video_keys]

    def test(self):
        print('Done testing __init__ ...')

    def _get_vlm_response_record(
        self,
        client: openai.Client,
        frame_dict: dict,
        prompt_vlm: str,
        vlm_model: str,
        base64_frames: list[str],
        response_type: Literal['caption', 'parsed', 'binary', 'parsed_binary'] = 'caption',
    ):
        prompt_vlm = textwrap.dedent(prompt_vlm).strip()
        if response_type == 'caption':
            vlm_response_raw: str = generate_segment_caption(
                client,
                base64_frames,
                user_prompt=prompt_vlm,
                model=vlm_model
            )
            response_record = {
                'segment_idx': frame_dict['segment_idx'],
                'start_idx': frame_dict['segment_start_idx'],
                'end_idx': frame_dict['segment_end_idx'],
                'response': vlm_response_raw,
                'score_raw': None,  # placeholder for LLM response
                'score': None,  # placeholder for LLM response
            }

        elif response_type == 'parsed':
            vlm_response_raw: str = generate_segment_caption(
                client,
                base64_frames,
                user_prompt=prompt_vlm,
                model=vlm_model,
                response_format=AnomalyScoreWithExplanation
            )
            vlm_response = eval(vlm_response_raw)
            response_record = {
                'segment_idx': frame_dict['segment_idx'],
                'start_idx': frame_dict['segment_start_idx'],
                'end_idx': frame_dict['segment_end_idx'],
                'response': vlm_response_raw,
                'score': vlm_response['anomaly_score'],
                'explanation': vlm_response['explanation'],
            }

        elif response_type == 'binary':
            vlm_response_raw: str = generate_segment_caption(
                client,
                base64_frames,
                user_prompt=prompt_vlm,
                model=vlm_model
            )
            response_record = {
                'segment_idx': frame_dict['segment_idx'],
                'start_idx': frame_dict['segment_start_idx'],
                'end_idx': frame_dict['segment_end_idx'],
                'response': vlm_response_raw,
                'score': (1 if 'yes' in vlm_response_raw.lower() else 0),
            }
        elif response_type == 'parsed_binary':
            vlm_response_raw: str = generate_segment_caption(
                client,
                base64_frames,
                user_prompt=prompt_vlm,
                model=vlm_model,
                response_format=EventOrActionThenYesOrNoAnomalyScore
            )
            vlm_response = eval(vlm_response_raw)
            response_record = {
                'segment_idx': frame_dict['segment_idx'],
                'start_idx': frame_dict['segment_start_idx'],
                'end_idx': frame_dict['segment_end_idx'],
                'response': vlm_response_raw,
                'score': (1 if 'yes' in vlm_response['is_anomalous'].lower() else 0),
                'event_or_activity': vlm_response['event_or_activity'],
            }

        return response_record

    def _generate_vlm(
        self,
        host: str = 'localhost', port: int = 50001,
        rank: int = 0, world_size: int = 1,
        vlm_model: str = 'lmms-lab/llava-onevision-qwen2-7b-ov',
        prompt_vlm: str = "Describe the video in a few sentences.",
        duration_sec = 1,
        debug: bool = False,
        response_type: Literal['caption', 'parsed', 'binary', 'parsed_binary'] = 'caption',
    ):
        num_input_frames = (
            32 if 'onevision' in vlm_model
            else 16 if 'Qwen/Qwen' in vlm_model
            else 1
        )

        p_outdir = get_outdir(
            vlm_model=vlm_model,
            llm_model=None if response_type == 'caption' else vlm_model,
            prompt_vlm=prompt_vlm,
            prompt_type='per-segment',
            duration_sec=duration_sec,
        )
        p_outdir.mkdir(exist_ok=True, parents=True)

        df_ann_test_rank = self.df_ann_test.iloc[rank::world_size]

        client = get_client(
            host=host,
            port=port,
            model_name=vlm_model
        )

        num_total_segments = self.sr_num_frames.iloc[rank::world_size].sum() // (duration_sec * 30)
        pbar_segments = tqdm(
            position=2,
            total=num_total_segments,
            file=sys.stdout,
        )
        for idx, row in tqdm(
            df_ann_test_rank.iterrows(), total=len(df_ann_test_rank), position=0, mininterval=.01, file=sys.stdout,
        ):
            p_json = (p_outdir / row['label'] / row['video']).with_suffix('.json')
            raw_label = row['label'] if row['label'] != 'Normal' else 'Testing_Normal_Videos_Anomaly'
            p_video = self.p_videos_root / raw_label / row['video']
            num_video_frames = self.sr_num_frames[f'{raw_label}/{row["video"]}']
            num_video_segments = num_video_frames // (duration_sec * 30)
            if p_json.exists():
                tqdm.write(f"Skipping {p_json}")
                pbar_segments.update(num_video_segments)
                continue
            else:
                p_json.parent.mkdir(exist_ok=True, parents=True)
            tqdm.write(f'\nProcessing {p_video}\n\t-> {p_json}')
            response_records = []
            for segment_dict in tqdm(
                get_segments(
                    p_video,
                    segment_duration_sec=duration_sec,
                    num_segment_frames=num_input_frames
                ),
                position=1,
                total=num_video_segments,
                file=sys.stdout,
                leave=False,
            ):
                base64_frames = frames_to_base64(segment_dict['frames'])
                response_record = self._get_vlm_response_record(
                    client=client,
                    frame_dict=segment_dict,
                    prompt_vlm=prompt_vlm,
                    vlm_model=vlm_model,
                    base64_frames=base64_frames,
                    response_type=response_type,
                )
                if debug:
                    tqdm.write(f'[{segment_dict["segment_idx"]}/{segment_dict["total_segments"]}] {p_video} -> {p_json}')
                    tqdm.write(response_record, end='\n\n')
                response_records.append(response_record)
                pbar_segments.update(1)

            video_metadata = {
                'video': row['video'],  # e.g. 'Abuse001_x264.mp4'
                'label': row['label'],  # e.g. 'Abuse', 'Normal', ...
                's1': row['s1'],
                'e1': row['e1'],
                's2': row['s2'],
                'e2': row['e2'],
            }
            video_record = {
                **video_metadata,
                'response_records': response_records,
            }
            tqdm.write('Saving ...')
            json.dump(video_record, p_json.open('w'), indent=2)
        pbar_segments.close()

    def generate_vlm(
        self,
        host: str = 'localhost', port: int = 50001,
        rank: int = 0, world_size: int = 1,
        vlm_model: str = 'lmms-lab/llava-onevision-qwen2-7b-ov',
        prompt_vlm: str = "Describe the video in a few sentences.",
        duration_sec = 1,
        debug: bool = False,
    ):
        self._generate_vlm(
            host=host,
            port=port,
            rank=rank,
            world_size=world_size,
            vlm_model=vlm_model,
            prompt_vlm=prompt_vlm,
            duration_sec=duration_sec,
            debug=debug,
            response_type='caption',
        )

    def generate_integrated_parsed(
        self,
        host: str = 'localhost', port: int = 50001,
        rank: int = 0, world_size: int = 1,
        vlm_model: str = 'gpt-4o',
        prompt_vlm: str = "How anomalous is this video? Please rate from 0 to 1 with 0 being not anomalous and 1 being very anomalous and provide an explanation in a few sentences in provided json format.",
        duration_sec = 1,
        debug: bool = False,
    ):
        self._generate_vlm(
            host=host,
            port=port,
            rank=rank,
            world_size=world_size,
            vlm_model=vlm_model,
            prompt_vlm=prompt_vlm,
            duration_sec=duration_sec,
            debug=debug,
            response_type='parsed',
        )

    def generate_integrated_binary(
        self,
        host: str = 'localhost', port: int = 50001,
        rank: int = 0, world_size: int = 1,
        vlm_model: str = 'gpt-4o',
        prompt_vlm: str = "Does this image contain abnormal events like crimes, accidents? Your response should be only one of yes or no without any explanation.",
        duration_sec = 1,
        debug: bool = False,
    ):
        self._generate_vlm(
            host=host,
            port=port,
            rank=rank,
            world_size=world_size,
            vlm_model=vlm_model,
            prompt_vlm=prompt_vlm,
            duration_sec=duration_sec,
            debug=debug,
            response_type='binary',
        )

    def generate_integrated_parsed_binary(
        self,
        host: str = 'localhost', port: int = 50001,
        rank: int = 0, world_size: int = 1,
        vlm_model: str = 'gpt-4o',
        prompt_vlm: str = textwrap.dedent('''\
            Does this video contain any anomalous events or activities? Please provide the category of the action in the video in a single word. Then, decide your final answer: yes or no. Also, the response should be in provided json format.

            The categories are: 'abuse', 'arrest', 'arson', 'assault', 'burglary', 'explosion', 'fighting', 'normal', 'robbery', 'shooting', 'shoplifting', 'stealing', 'vandalism'.

            Hint: Exaggerated or fast movements are highly likely to be anomalous.'''),
        duration_sec = 1,
        debug: bool = False,
    ):
        self._generate_vlm(
            host=host,
            port=port,
            rank=rank,
            world_size=world_size,
            vlm_model=vlm_model,
            prompt_vlm=prompt_vlm,
            duration_sec=duration_sec,
            debug=debug,
            response_type='parsed_binary',
        )

    def generate_llm(
        self,
        host: str = 'localhost', port: int = 50002,
        rank: int = 0, world_size: int = 1,
        vlm_model: str = 'lmms-lab/llava-onevision-qwen2-7b-ov',
        llm_model: str = 'meta-llama/Llama-3.2-3B-Instruct',
        prompt_vlm: str = "Describe the video in a few sentences.",
        prompt_type: VLM_RSEPONSE_TYPES = 'per-segment',
        duration_sec = 1,
        debug: bool = False,
    ):
        p_vlm_outdir = get_outdir(
            vlm_model=vlm_model,
            llm_model=None,
            prompt_vlm=prompt_vlm,
            prompt_type=None,
            duration_sec=duration_sec,
        )
        assert (_n := len(list(p_vlm_outdir.glob('**/*.json')))) == (n := len(self.df_ann_test)), f"Missing VLM captions in {p_vlm_outdir}, got {_n} but expected {n}"
        p_llm_outdir = get_outdir(
            vlm_model=vlm_model,
            llm_model=llm_model,
            prompt_vlm=prompt_vlm,
            prompt_type=prompt_type,
            duration_sec=duration_sec,
        )
        p_llm_outdir.mkdir(exist_ok=True, parents=True)

        # df_ann_test = pd.read_csv(
        #     self.p_ann_test, sep=r'\s+', header=None, names=['video', 'label', 's1', 'e1', 's2', 'e2'])
        df_ann_test = self.df_ann_test.iloc[rank::world_size]
        is_openai_reasoning_model = 'o1' in llm_model or 'o3' in llm_model

        if llm_model not in OPENAI_MODELS:
            server_address = f"http://{host}:{port}"
            print(f'Waiting for LLM server at {server_address}...', flush=True)
            wait_for_server(server_address, timeout=600)
            client = openai.Client(api_key="EMPTY", base_url=f"{server_address}/v1")
        else:
            print('Using OpenAI API', flush=True)
            client = openai.Client(api_key=os.environ.get('OPENAI_API_KEY'))

        if prompt_type == 'ko':
            context_prompt = "다음은 비디오에 대한 설명이야. 만약 너가 경찰이라면 이 장면의 이상한 정도를 0부터 1까지의 점수 중에 어떻게 평가할 거야? 0은 일상적인 장면이고 1은 의심스러운 활동이 있는 장면이야."
            format_prompt = "아래에 제공된 목록에서 하나의 숫자로만 응답해줘. [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] 이 숫자는 어떤 설명도 없이 제공되어야 해. '['로 시작해서 ']'로 끝나야 해."
        else:
            context_prompt = "The following is a scene description. If you were a law enforcement agency, how would you rate this scene on a scale from 0 to 1, with 0 representing a standard scene and 1 denoting a scene with suspicious activities?"
            format_prompt = "Please provide the response in the form of a Python list and respond with only one number in the provided list below [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] without any textual explanation. It should begin with '[' and end with  ']'."

        system_prompt = f'{context_prompt} {format_prompt}'

        def parse_score_response(response: str):
            if 'DeepSeek-R1' in llm_model:
                response = re.sub(r'^(?s:.)*</think>', '', response).strip()
            elif 'o1' in llm_model or 'o3' in llm_model:
                response = response.replace('```python', '').replace('```', '')
            try:
                score = eval(response)[0]
            except Exception as e:
                print(e, file=sys.stderr)
                print(response, file=sys.stderr)
                score = None
            return score

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

            video_record['system_prompt'] = system_prompt
            prev_prompts, prev_responses = [], []
            for seg_idx, response_record in enumerate(video_record['response_records']):
                if prompt_type in ['whole', 'windowing']:
                    messages = [
                        {
                            'role': 'system',
                            'content': system_prompt,
                        },
                    ]
                    for prev_prompt, prev_response in zip(prev_prompts, prev_responses):
                        messages.append({
                            'role': 'user',
                            'content': prev_prompt,
                        })
                        messages.append({
                            'role': 'assistant',
                            'content': prev_response,
                        })
                    user_prompt = f"Scene Descriptions for Segment #{seg_idx}: {response_record['response']}"
                    messages.append({
                        'role': 'user',
                        'content': user_prompt,
                    })
                else:
                    user_prompt = f"Scene Description: {response_record['response']}"
                    if is_openai_reasoning_model:
                        messages = [
                            {
                                'role': 'user',
                                'content': system_prompt + '\n\n\n' + user_prompt,
                            },
                        ]
                    else:
                        messages = [
                            {
                                'role': 'system',
                                'content': system_prompt,
                            },
                            {
                                'role': 'user',
                                'content': user_prompt,
                            },
                        ]

                llm_response: str = chat(
                    client=client,
                    messages=messages,
                    model=llm_model,
                    temperature=1 if is_openai_reasoning_model else 0,
                )
                prev_prompts.append(user_prompt)
                prev_responses.append(llm_response)
                if prompt_type == 'windowing':
                    max_window_size = 10
                    prev_prompts = prev_prompts[-max_window_size:]
                    prev_responses = prev_responses[-max_window_size:]
                response_record['score_raw'] = llm_response
                response_record['score'] = parse_score_response(llm_response)
                if debug:
                    tqdm.write(json.dumps(response_record, indent=2))
            json.dump(video_record, p_json_new.open('w'), indent=2)

    def evaluate(
        self,
        rank: int = 0,
        vlm_model: str = 'lmms-lab/llava-onevision-qwen2-7b-ov',
        llm_model: str = 'meta-llama/Llama-3.2-3B-Instruct',
        prompt_vlm: str = "Describe the video in a few sentences.",
        prompt_type: VLM_RSEPONSE_TYPES = 'per-segment',
        duration_sec = 1,
        force: bool = False,
    ):
        if rank != 0:
            print(f'Skipping rank={rank} != 0 for evaluation')
            return

        # p_outdir = Path('output/ucf-crime-captions') /\
        #     vlm_model.replace('/', '-') /\
        #     f"prompt={prompt_vlm.replace(' ', '_')}_duration_{duration_sec}s/" /\
        #     llm_model.replace('/', '-') /\
        #     prompt_type
        p_outdir = get_outdir(
            vlm_model=vlm_model,
            llm_model=llm_model,
            prompt_vlm=prompt_vlm,
            prompt_type=prompt_type,
            duration_sec=duration_sec,
        )

        print(f'Evaluating {p_outdir} ...')
        print()
        print(f'{vlm_model=}')
        print(f'{llm_model=}')
        print(f'{prompt_vlm=}')
        print(f'{prompt_type=}')
        print(f'{duration_sec=}')
        print()

        ann_vad = {}
        for idx, row in self.df_ann_test.iterrows():
            key = f"{row['label']}/{row['video']}"
            num_frames = self.sr_num_frames[key.replace('Normal/', 'Testing_Normal_Videos_Anomaly/')]
            bin_label = np.zeros(num_frames, dtype=np.int32)
            bin_label[row['s1']:row['e1']] = 1
            if row['s2'] != -1:
                bin_label[row['s2']:row['e2']] = 1
            ann_vad[key] = bin_label

        preds = {}
        for key, bin_label in ann_vad.items():
            p_json = (p_outdir / key).with_suffix('.json')
            if not p_json.exists():
                if force:
                    preds[key] = np.zeros(len(bin_label))
                    continue
            video_record = json.load(p_json.open())
            records = video_record['response_records']
            pred = np.zeros(len(bin_label))
            for record in records:
                segment_score = record['score'] or 0.  # fill None with 0
                pred[record['start_idx']:record['end_idx']+1] = segment_score
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
