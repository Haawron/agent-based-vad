from typing import Literal
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch.utils.data

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from torchvision import transforms
    from torchvision.transforms._transforms_video import NormalizeVideo, CenterCropVideo
# from pytorchvideo import transforms as pv_transforms

import decord
from decord import VideoReader, cpu
decord.bridge.reset_bridge()


# model names in a single line are aliases for the same model
OPENAI_MODELS = [
    'gpt-4o', 'chatgpt-4o-latest',
    'gpt-4o-mini',
    'o1',
    'o1-mini',
    'o3-mini',
    'gpt-4-turbo',
]

FPS = 30

p_ann_root = Path('/code/data/annotations')
p_ann_test = p_ann_root / 'Temporal_Anomaly_Annotation_for_Testing_Videos.txt'
p_num_frames = p_ann_root / 'num_frames_per_video.txt'

df_ann_test = pd.read_csv(
    p_ann_test, sep=r'\s+', header=None, names=['video', 'label', 's1', 'e1', 's2', 'e2'])
df_ann_test['raw_label'] = df_ann_test['label'].str.replace('Normal', 'Testing_Normal_Videos_Anomaly')
df_ann_test['raw_rel_video_path'] = df_ann_test['raw_label'].str.cat(df_ann_test['video'], sep='/')
df_ann_test['rel_video_path'] = df_ann_test['label'].str.cat(df_ann_test['video'], sep='/')

sr_num_frames = pd.read_csv(p_num_frames, sep=r'\s+', header=None, names=['video', 'num_frames'])
sr_num_frames = sr_num_frames.set_index('video')['num_frames']
test_keys = df_ann_test['raw_rel_video_path']
train_keys = sr_num_frames.index.difference(test_keys)
sr_num_frames_test = sr_num_frames.loc[test_keys]
sr_num_frames_train = sr_num_frames.loc[train_keys]


def get_client(
    host: str,
    port: int,
    model_name: str,
):
    import openai
    if model_name in OPENAI_MODELS:
        print('Using OpenAI API', flush=True)
        client = openai.Client(api_key=os.environ.get('OPENAI_API_KEY'))
    else:
        from sglang.utils import wait_for_server
        server_address = f"http://{host}:{port}"
        print(f'Waiting for LLM server at {server_address}...', flush=True)
        wait_for_server(server_address, timeout=600)
        client = openai.Client(api_key="EMPTY", base_url=f"{server_address}/v1")
    return client


def compute_num_segments(
    p_video,
    segment_duration_sec: float = 1.,
    segment_overlap_sec: float = 0.,
):
    num_segment_frames = int(segment_duration_sec * FPS)
    num_overlap_frames = int(segment_overlap_sec * FPS)

    # vr = VideoReader(str(p_video), ctx=cpu(0))
    # num_total_frames = len(vr)
    key = p_video.parent.name + '/' + p_video.name
    num_frames = sr_num_frames[key]
    num_segments = (num_frames - num_segment_frames) // (num_segment_frames - num_overlap_frames) + 1  # Discard last segment if not enough frames
    return num_segments


def get_segments(
    p_video,
    segment_duration_sec: float = 1.,
    num_segment_frames: int = 32,
    segment_overlap_sec: float = 0.,
    num_skip_first_segments: int = 0,
):
    num_segment_frames = int(segment_duration_sec * FPS)
    num_overlap_frames = int(segment_overlap_sec * FPS)

    vr = VideoReader(str(p_video), ctx=cpu(0))
    num_segments = compute_num_segments(p_video, segment_duration_sec, segment_overlap_sec)
    for segment_idx in range(num_segments):
        if num_skip_first_segments > 0 and segment_idx < num_skip_first_segments:
            continue
        segment_start_idx = segment_idx * (num_segment_frames - num_overlap_frames)
        segment_end_idx = segment_start_idx + num_segment_frames - 1
        uniform_sampled_frames = np.linspace(segment_start_idx, segment_end_idx, num_segment_frames + 2, dtype=int)[1:-1]
        frame_idxs = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idxs).asnumpy()
        frame_dict = {
            'frames': frames,
            'segment_info': {
                'segment_idx': segment_idx,
                'total_segments': num_segments,
                'segment_start_idx': segment_start_idx,
                'segment_end_idx': segment_end_idx,
            },
        }
        yield frame_dict


class SegmentDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        p_videos_dir = Path('/code/data/UCF_Crimes/Videos'),
        segment_duration_sec: float = 1.,
        segment_overlap_sec: float = .5,
        num_sampled_segment_frames: int = 16,
        split: Literal['train', 'test'] = 'test',
        rank: int = 0, world_size: int = 1,
    ):
        super().__init__()
        self.sr_num_frames = sr_num_frames_test if split == 'test' else sr_num_frames_train
        self.p_videos_dir = p_videos_dir
        self.p_videos = sorted([p_video for p_video in p_videos_dir.glob('**/*.mp4') if p_video.parent.name + '/' + p_video.name in self.sr_num_frames.index])
        self.segment_duration_sec = segment_duration_sec
        self.segment_overlap_sec = segment_overlap_sec
        self.num_sampled_segment_frames = num_sampled_segment_frames
        self.split = split
        self.rank = rank
        self.world_size = world_size

        num_segment_frames = int(segment_duration_sec * FPS)
        num_overlap_frames = int(segment_overlap_sec * FPS)
        self.sr_num_segments = (self.sr_num_frames - num_segment_frames) // (num_segment_frames - num_overlap_frames) + 1
        self.num_total_segments = self.sr_num_segments.sum()

        self.segment_infos = []
        for p_video in self.p_videos:
            num_segments = self.sr_num_segments[p_video.parent.name + '/' + p_video.name]
            key = p_video.parent.name + '/' + p_video.name
            for segment_idx in range(num_segments):
                segment_start_idx = segment_idx * (num_segment_frames - num_overlap_frames)
                segment_end_idx = segment_start_idx + num_segment_frames - 1
                self.segment_infos.append({
                    'p_video': p_video,
                    'segment_idx': segment_idx,
                    'total_segments': num_segments,
                    'segment_start_idx': segment_start_idx,
                    'segment_end_idx': segment_end_idx,
                })

        assert len(self.segment_infos) == self.num_total_segments, f'{len(self.segment_infos)=} != {self.num_total_segments=}'
        self.segment_infos = self.segment_infos[self.rank::self.world_size]

        from pytorchvideo import transforms as pv_transforms
        self.video_transform = transforms.Compose(
            [
                pv_transforms.ShortSideScale(256),
                pv_transforms.Div255(),
                CenterCropVideo(224),
                NormalizeVideo(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def __len__(self):
        return len(self.segment_infos)

    def __getitem__(self, idx):
        segment_info = self.segment_infos[idx]
        p_video = segment_info['p_video']
        segment_start_idx = segment_info['segment_start_idx']
        segment_end_idx = segment_info['segment_end_idx']
        # num_segment_frames = int(self.segment_duration_sec * FPS)
        # uniform_sampled_frames = np.linspace(segment_start_idx, segment_end_idx, num_segment_frames + 2, dtype=int)[1:-1]
        uniform_sampled_frames = np.linspace(segment_start_idx, segment_end_idx, self.num_sampled_segment_frames + 2, dtype=int)[1:-1]
        frame_idxs = uniform_sampled_frames.tolist()
        frames = VideoReader(str(p_video), ctx=cpu(0)).get_batch(frame_idxs).asnumpy()
        frames = torch.tensor(frames).permute(3, 0, 1, 2).float()  # [T, H, W, C] -> [C, T, H, W]
        frames = self.video_transform(frames)
        return {
            'frames': frames,
            'segment_info': segment_info,
        }


if __name__ == '__main__':
    from tqdm import tqdm
    ds = SegmentDataset()
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        num_workers=8,
        collate_fn=lambda x: x,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        prefetch_factor=16
    )
    for data in tqdm(dl, total=len(ds)):
        tqdm.write(str(data[0]['segment_info']))
