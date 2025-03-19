import os
import sys
import warnings
sys.path.append('/code/src')
from helper import get_segments, df_ann_test, sr_num_frames_test, SegmentDataset

import time
import json
import textwrap
from pathlib import Path
from tqdm import trange, tqdm

import pandas as pd
import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import roc_auc_score, roc_curve
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from torchvision import transforms
    from torchvision.transforms._transforms_video import NormalizeVideo, CenterCropVideo
from pytorchvideo import transforms as pv_transforms

import faiss
import decord
from decord import VideoReader, cpu
decord.bridge.reset_bridge()

from einops import rearrange

os.chdir('/code/libs/imagebind')
sys.path = ['/code/libs/imagebind'] + sys.path  # Prepend to avoid conflicts with installed imagebind
from imagebind import data
from imagebind.models.imagebind_model import ModalityType
from imagebind.models.imagebind_model import ImageBindModel


class Main:
    def __init__(self):
        ### dataset paths
        self.p_dataroot = Path('/datasets/UCF_Crimes')
        self.p_annroot = Path('/code/data/annotations')
        self.p_videos_root = self.p_dataroot / 'Videos'
        # self.p_ann_test = self.p_annroot / 'Temporal_Anomaly_Annotation_for_Testing_Videos.txt'
        self.p_num_frames = self.p_annroot / 'num_frames_per_video.txt'

        # caption paths
        self.p_captions_root = Path("/code/output/psuedo-captions/gpt-4o/00-rich-context")
        self.p_captions_dir = self.p_captions_root / "captions"
        assert self.p_captions_dir.exists(), f"Captions directory {self.p_captions_dir} does not exist"

        # output paths
        self.p_normal = self.p_captions_root / "embs_normal.npy"
        self.p_anomalous = self.p_captions_root / "embs_anomalous.npy"
        self.p_outdir_embeddings = self.p_captions_root / "embeddings"
        self.p_outdir_scored = self.p_captions_root / "scored"
        self.p_outdir_scored.mkdir(exist_ok=True, parents=True)

    def _load_captions(self):
        print("Loading captions...")
        texts_normal, texts_anomalous = [], []
        for p_json in self.p_captions_dir.glob("*.json"):
            with p_json.open("r") as f:
                captions = json.load(f)
                for caption_set in captions['descriptions']:
                    texts_normal.append(caption_set['normal']['description'])
                    texts_anomalous.append(caption_set['anomalous']['description'])
        print(f'Loaded {len(texts_normal)} normal captions and {len(texts_anomalous)} anomalous captions')
        return texts_normal, texts_anomalous

    def _load_imagebind_model(self, device):
        print("Loading model...", flush=True)
        model = ImageBindModel.from_pretrained("nielsr/imagebind-huge")
        model.eval()
        model.to(device)
        print(f"Loaded model", flush=True)
        return model

    @torch.inference_mode()
    def _create_or_load_caption_embeddings(self, model=None, device='cuda', rank=0):
        tqdm.write("Creating or loading embeddings...")
        num_normal = len(self.texts_normal)
        num_anomalous = len(self.texts_anomalous)
        stride = 1024

        if rank == 0 and not self.p_normal.exists():
            tqdm.write(f'[Rank {rank}] Creating embeddings for normal captions...')
            embs_normal = []
            for i in trange(0, num_normal, stride, desc="Normal"):
                text_batch = data.load_and_transform_text(self.texts_normal[i:i+stride], device)
                emb_normal = model.forward({ModalityType.TEXT: text_batch})[ModalityType.TEXT]
                embs_normal.append(emb_normal.cpu().numpy())
            embs_normal = np.concatenate(embs_normal, axis=0)
            np.save(self.p_normal, embs_normal)
        else:
            while not self.p_normal.exists():
                tqdm.write(f'[Rank {rank}] Waiting for rank 0 to finish creating embeddings...')
                time.sleep(5)
            tqdm.write(f'[Rank {rank}] Loading embeddings...')
            embs_normal = np.load(self.p_normal)

        if rank == 0 and not self.p_anomalous.exists():
            tqdm.write(f'[Rank {rank}] Creating embeddings for anomalous captions ...')
            embs_anomalous = []
            for i in trange(0, num_anomalous, stride, desc="Anomalous"):
                text_batch = data.load_and_transform_text(self.texts_anomalous[i:i+stride], device)
                emb_anomalous = model.forward({ModalityType.TEXT: text_batch})[ModalityType.TEXT]
                embs_anomalous.append(emb_anomalous.cpu().numpy())
            embs_anomalous = np.concatenate(embs_anomalous, axis=0)
            np.save(self.p_anomalous, embs_anomalous)
        else:
            while not self.p_anomalous.exists():
                tqdm.write(f'[Rank {rank}] Waiting for rank 0 to finish creating embeddings...')
                time.sleep(5)
            tqdm.write(f'[Rank {rank}] Loading embeddings...')
            embs_anomalous = np.load(self.p_anomalous)

        return embs_normal, embs_anomalous

    def _create_or_load_caption_index(
        self,
        embs_normal,
        embs_anomalous,
        anomalous_scale=1.,
        rank=0
    ) -> faiss.Index:
        print("Creating or loading Faiss index...")
        d = 1024
        num_normal = len(self.texts_normal)
        num_anomalous = len(self.texts_anomalous)
        p_faiss_index = self.p_captions_root / f"faiss_scale={anomalous_scale:.1f}.index"

        if rank == 0:
            print(f'[Rank {rank}] Creating Faiss index...')
            index = faiss.IndexFlatIP(d)
            index = faiss.IndexIDMap2(index)
            index.add_with_ids(embs_normal, np.arange(num_normal))
            index.add_with_ids(embs_anomalous * anomalous_scale, np.arange(num_normal, num_normal + num_anomalous))
            faiss.write_index(index, str(p_faiss_index))
        else:
            while not p_faiss_index.exists():
                print(f'[Rank {rank}] Waiting for rank 0 to finish creating Faiss index...')
                time.sleep(1)
            print(f'[Rank {rank}] Loading Faiss index...')
            index = faiss.read_index(str(p_faiss_index))

        print(f"Loaded Faiss index {index}")
        return index

    def run_per_video(
        self,
        rank: int = 0, world_size: int = 1,
        segment_duration_sec: float = 1.,
        segment_overlap_sec: float = .5,
        num_sampled_segment_frames: int = 16,
        num_skip_first_segments: int = 0,  # for debugging
        num_captions_per_segment: int = 10,
    ):
        device = f'cuda:{rank % 8}'
        self.texts_normal, self.texts_anomalous = self._load_captions()
        model = self._load_imagebind_model(device)
        embs_normal, embs_anomalous = self._create_or_load_caption_embeddings(model, device, rank)
        index = self._create_or_load_caption_index(
            embs_normal, embs_anomalous, anomalous_scale=0.7, rank=rank
        )

        texts = self.texts_normal + self.texts_anomalous
        df_ann_test_rank = df_ann_test.iloc[rank::world_size].reset_index(drop=True)
        sr_num_frames_rank = sr_num_frames_test.iloc[rank::world_size]
        num_segment_frames = int(segment_duration_sec * 30)
        num_overlap_frames = int(segment_overlap_sec * 30)
        sr_num_segments = (sr_num_frames_rank - num_segment_frames) // (num_segment_frames - num_overlap_frames) + 1
        num_total_segments = sr_num_segments.sum()
        tqdm.write(f"Rank {rank} of {world_size} will process {len(df_ann_test_rank)} videos with {num_total_segments} segments")

        video_transform = transforms.Compose(
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

        pbar_segments = tqdm(
            position=2,
            total=num_total_segments,
            file=sys.stdout,
        )

        for idx, row in tqdm(
            df_ann_test_rank.iterrows(), total=len(df_ann_test_rank), position=0, mininterval=.001, file=sys.stdout,
        ):
            p_video = self.p_videos_root / row['raw_rel_video_path']
            p_out_score = (self.p_outdir_scored / row['rel_video_path']).with_suffix('.json')
            p_out_score.parent.mkdir(exist_ok=True, parents=True)
            p_out_video_embedding = (self.p_outdir_embeddings / row['rel_video_path']).with_suffix('.npy')
            p_out_video_embedding.parent.mkdir(exist_ok=True, parents=True)
            num_video_segments = sr_num_segments[row['raw_rel_video_path']]
            if p_out_score.exists():
                pbar_segments.update(num_video_segments)
                continue

            segment_records = []
            embs_video = []
            for segment_dict in tqdm(
                get_segments(
                    p_video,
                    segment_duration_sec=segment_duration_sec,
                    num_segment_frames=num_sampled_segment_frames,
                    segment_overlap_sec=segment_overlap_sec,
                    num_skip_first_segments=num_skip_first_segments,
                ),
                position=1,
                total=num_video_segments,
                file=sys.stdout,
                leave=False,
            ):
                frames = torch.tensor(segment_dict['frames']).permute(3, 0, 1, 2).float()  # [T, H, W, C] -> [C, T, H, W]
                frames = video_transform(frames)
                frames = frames.to(self.device)
                with torch.inference_mode():
                    input_frames = rearrange(frames, 'c (t s) h w -> s c t h w', t=2)  # T=2 fixed as ImageBind expects 2 frames per clip
                    emb_segment = model.forward({ModalityType.VISION: input_frames[None]})[ModalityType.VISION]
                    emb_segment = emb_segment.cpu().numpy()
                    embs_video.append(emb_segment)
                dot_products, indices = index.search(emb_segment, num_captions_per_segment)
                selected_texts = np.take(texts, indices)
                is_anomalous = indices > len(self.texts_normal)

                segment_record = {
                    'segment_idx': segment_dict['segment_info']['segment_idx'],
                    'total_segments': segment_dict['segment_info']['total_segments'],
                    'start_idx': segment_dict['segment_info']['segment_start_idx'],
                    'end_idx': segment_dict['segment_info']['segment_end_idx'],
                    'dot_products': dot_products[0].tolist(),
                    'indices': indices[0].tolist(),
                    'captions': selected_texts[0].tolist(),
                    'is_anomalous': is_anomalous[0].tolist(),
                }
                segment_records.append(segment_record)
                pbar_segments.update(1)
            embs_video = np.concatenate(embs_video, axis=0)  # [num_segments, 1024]
            np.save(p_out_video_embedding, embs_video)

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
                'segment_records': segment_records,
            }
            json.dump(video_record, p_out_score.open('w'), indent=2)

    def extract_embeddings_per_segment(
        self,
        rank: int = 0, world_size: int = 1,
        segment_duration_sec: float = 1.,
        segment_overlap_sec: float = .5,
        num_sampled_segment_frames: int = 16,
    ):
        device = f'cuda:{rank % 8}'
        p_outdir_segment_embeddings_with_options = self.p_outdir_embeddings / f"dur={segment_duration_sec:.1f}_ol={segment_overlap_sec:.1f}_fs={num_sampled_segment_frames}" / 'segments'
        p_outdir_segment_embeddings_with_options.mkdir(exist_ok=True, parents=True)
        print('Outdir:', p_outdir_segment_embeddings_with_options, flush=True)

        model = self._load_imagebind_model(device)

        ds = SegmentDataset(
            p_videos_dir=self.p_videos_root,
            segment_duration_sec=segment_duration_sec,
            segment_overlap_sec=segment_overlap_sec,
            num_sampled_segment_frames=num_sampled_segment_frames,
            split='test',
            rank=rank, world_size=world_size,
        )
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
        num_total_segments = len(ds)
        print(f"Rank {rank} of {world_size} will process {num_total_segments} segments", flush=True)

        for segment_data in tqdm(
            dl, total=len(ds), position=0, mininterval=.1, miniters=1, maxinterval=.5, file=sys.stdout,
        ):
            segment_info = segment_data[0]['segment_info']
            p_video = segment_info['p_video']
            p_out_embedding = (
                p_outdir_segment_embeddings_with_options
                / p_video.parent.name.split('_')[0]
                / p_video.stem
                / f'{segment_info["segment_idx"]:04d}.npy'
            )
            p_out_embedding.parent.mkdir(exist_ok=True, parents=True)
            frames = segment_data[0]['frames'].to(device)

            with torch.inference_mode():
                input_frames = rearrange(frames, 'c (t s) h w -> s c t h w', t=2)  # T=2 fixed as ImageBind expects 2 frames per clip
                emb_segment = model.forward({ModalityType.VISION: input_frames[None]})[ModalityType.VISION][0]
            emb_segment = emb_segment.cpu().numpy()
            np.save(p_out_embedding, emb_segment)

    def match_captions_per_segment(
        self,
        rank: int = 0, world_size: int = 1,
        segment_duration_sec: float = 1.,
        segment_overlap_sec: float = .5,
        num_sampled_segment_frames: int = 16,
        num_captions_per_segment: int = 10,
        anomalous_scale: float = 1.,
    ):
        num_segment_frames = int(segment_duration_sec * 30)
        num_overlap_frames = int(segment_overlap_sec * 30)
        self.texts_normal, self.texts_anomalous = self._load_captions()
        embs_normal, embs_anomalous = self._create_or_load_caption_embeddings(rank=rank)
        index = self._create_or_load_caption_index(
            embs_normal, embs_anomalous,
            anomalous_scale=anomalous_scale, rank=rank
        )
        texts = self.texts_normal + self.texts_anomalous

        p_segment_embeddings_with_options = self.p_outdir_embeddings / f"dur={segment_duration_sec:.1f}_ol={segment_overlap_sec:.1f}_fs={num_sampled_segment_frames}" / 'segments'
        p_segment_embeddings_with_options.mkdir(exist_ok=True, parents=True)

        df_ann_test_rank = df_ann_test.iloc[rank::world_size].reset_index(drop=True)
        pbar = tqdm(
            df_ann_test_rank.iterrows(),
            total=len(df_ann_test_rank),
            mininterval=.01,
            maxinterval=.5,
            file=sys.stdout,
        )
        for idx, row in pbar:
            p_out_score = (self.p_outdir_scored / row['rel_video_path']).with_suffix('.json')
            p_out_score.parent.mkdir(exist_ok=True, parents=True)
            p_segment_dir = p_segment_embeddings_with_options / row['rel_video_path'].split('.')[0]
            p_segments = sorted(p_segment_dir.glob('*.npy'))
            segment_embeddings = []
            for p_segment in p_segments:
                segment_embeddings.append(np.load(p_segment))
            segment_embeddings = np.stack(segment_embeddings, axis=0)
            dot_products, indices = index.search(segment_embeddings, num_captions_per_segment)
            selected_texts = np.take(texts, indices)
            is_anomalous = indices > len(self.texts_normal)
            pbar.set_postfix_str(f'{is_anomalous.mean():.6f}')

            segment_records = []
            for idx, p_segment in enumerate(p_segments):
                segment_idx = int(p_segment.stem)
                start_idx = segment_idx * (num_segment_frames - num_overlap_frames)
                end_idx = start_idx + num_segment_frames - 1
                segment_record = {
                    'segment_idx': segment_idx,
                    'total_segments': len(p_segments),
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'dot_products': dot_products[idx].tolist(),
                    'indices': indices[idx].tolist(),
                    'captions': selected_texts[idx].tolist(),
                    'is_anomalous': is_anomalous[idx].tolist(),
                }
                segment_records.append(segment_record)

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
                'segment_records': segment_records,
            }
            tqdm.write(f"Writing to {p_out_score}")
            json.dump(video_record, p_out_score.open('w'), indent=2)
            pbar.update(1)

    def evaluate(
        self,
        force: bool = False,
    ):
        ann_vad = {}
        for idx, row in df_ann_test.iterrows():
            num_frames = sr_num_frames_test[row['raw_rel_video_path']]
            bin_label = np.zeros(num_frames, dtype=np.int32)
            bin_label[row['s1']:row['e1']] = 1
            if row['s2'] != -1:
                bin_label[row['s2']:row['e2']] = 1
            ann_vad[row['rel_video_path']] = bin_label

        preds = {}
        gaussian_kernel = np.array([.1, .2, .4, .2, .1])
        for rel_video_path, bin_label in ann_vad.items():
            p_json = (self.p_outdir_scored / rel_video_path).with_suffix('.json')
            if not p_json.exists():
                if force:
                    preds[rel_video_path] = np.zeros(len(bin_label))
                    continue
            video_record = json.load(p_json.open())
            segment_records = video_record['segment_records']
            pred = np.zeros(len(bin_label))
            overlap_count = np.zeros(len(bin_label))
            for segment_record in segment_records:
                # segment_score = int(segment_record['is_anomalous'][0])  # anomality of the top 1 caption
                segment_score = np.mean(segment_record['is_anomalous'])  # anomality of all captions
                pred[segment_record['start_idx']:segment_record['end_idx']+1] += segment_score
                overlap_count[segment_record['start_idx']:segment_record['end_idx']+1] += 1
            overlap_count = np.maximum(overlap_count, 1)
            preds[rel_video_path] = pred / overlap_count
            preds[rel_video_path] = np.convolve(preds[rel_video_path], gaussian_kernel, mode='same')  # gaussian smoothing

        # compute AUC
        all_preds, all_labels = np.array([]), np.array([])
        for rel_video_path, bin_label in ann_vad.items():
            all_preds = np.concatenate([all_preds, preds[rel_video_path]])
            all_labels = np.concatenate([all_labels, bin_label])
        auc = roc_auc_score(all_labels.astype(int), all_preds)
        print(auc)

if __name__ == '__main__':
    import fire
    fire.Fire(Main)
