import os
import sys
sys.path.append('/code/src')
from helper import get_segments, compute_num_segments

import json
import textwrap
from pathlib import Path
from tqdm import trange, tqdm

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
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
        self.p_ann_test = self.p_annroot / 'Temporal_Anomaly_Annotation_for_Testing_Videos.txt'
        self.p_num_frames = self.p_annroot / 'num_frames_per_video.txt'

        self.df_ann_test = pd.read_csv(
            self.p_ann_test, sep=r'\s+', header=None, names=['video', 'label', 's1', 'e1', 's2', 'e2'])
        self.df_ann_test['raw_label'] = self.df_ann_test['label'].str.replace('Normal', 'Testing_Normal_Videos_Anomaly')
        self.df_ann_test['raw_rel_video_path'] = self.df_ann_test['raw_label'].str.cat(self.df_ann_test['video'], sep='/')
        self.df_ann_test['rel_video_path'] = self.df_ann_test['label'].str.cat(self.df_ann_test['video'], sep='/')
        sr_num_frames = pd.read_csv(
            self.p_num_frames, sep=r'\s+', header=None, names=['video', 'num_frames']
        ).set_index('video')['num_frames']
        self.sr_num_frames = sr_num_frames.loc[self.df_ann_test['raw_rel_video_path']]

        # caption paths
        self.p_captions_root = Path("/code/output/psuedo-captions/gpt-4o/00-rich-context")
        assert self.p_captions_root.exists()

        # output paths
        self.p_normal = self.p_captions_root / "embs_normal.npy"
        self.p_anomalous = self.p_captions_root / "embs_anomalous.npy"
        self.p_faiss_index = self.p_captions_root / "faiss.index"
        self.p_outdir = self.p_captions_root / "scored"
        self.p_outdir.mkdir(exist_ok=True, parents=True)

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

    def _load_captions(self):
        tqdm.write("Loading captions...")
        texts_normal, texts_anomalous = [], []
        for p_json in (self.p_captions_root / 'captions').glob("*.json"):
            with p_json.open("r") as f:
                captions = json.load(f)
                for caption_set in captions['descriptions']:
                    texts_normal.append(caption_set['normal']['description'])
                    texts_anomalous.append(caption_set['anomalous']['description'])
        tqdm.write(f'Loaded {len(texts_normal)} normal captions and {len(texts_anomalous)} anomalous captions')
        return texts_normal, texts_anomalous

    def _load_imagebind_model(self):
        tqdm.write("Loading model...")
        model = ImageBindModel.from_pretrained("nielsr/imagebind-huge")
        model.eval()
        model.to(self.device)
        tqdm.write(f"Loaded model")
        return model

    @torch.inference_mode()
    def _create_or_load_imagebind_embeddings(self):
        tqdm.write("Creating or loading embeddings...")
        num_normal = len(self.texts_normal)
        num_anomalous = len(self.texts_anomalous)
        stride = 1024

        if self.p_normal.exists():
            embs_normal = np.load(self.p_normal)
        else:
            embs_normal = []
            for i in trange(0, num_normal, stride, desc="Normal"):
                text_batch = data.load_and_transform_text(self.texts_normal[i:i+stride], self.device)
                emb_normal = self.model.forward({ModalityType.TEXT: text_batch})[ModalityType.TEXT]
                embs_normal.append(emb_normal.cpu().numpy())
            embs_normal = np.concatenate(embs_normal, axis=0)
            np.save(self.p_normal, embs_normal)

        if self.p_anomalous.exists():
            embs_anomalous = np.load(self.p_anomalous)
        else:
            embs_anomalous = []
            for i in trange(0, num_anomalous, stride, desc="Anomalous"):
                text_batch = data.load_and_transform_text(self.texts_anomalous[i:i+stride], self.device)
                emb_anomalous = self.model.forward({ModalityType.TEXT: text_batch})[ModalityType.TEXT]
                embs_anomalous.append(emb_anomalous.cpu().numpy())
            embs_anomalous = np.concatenate(embs_anomalous, axis=0)
            np.save(self.p_anomalous, embs_anomalous)

        return embs_normal, embs_anomalous

    def _create_or_load_faiss_index(self):
        tqdm.write("Creating or loading Faiss index...")
        d = 1024
        num_normal = len(self.texts_normal)
        num_anomalous = len(self.texts_anomalous)

        if not self.p_faiss_index.exists():
            tqdm.write("Creating Faiss index...")
            index = faiss.IndexFlatIP(d)
            index = faiss.IndexIDMap2(index)
            index.add_with_ids(self.embs_normal, np.arange(num_normal))
            index.add_with_ids(self.embs_anomalous, np.arange(num_normal, num_normal + num_anomalous))
            faiss.write_index(index, str(self.p_faiss_index))
        else:
            tqdm.write("Loading Faiss index...")
            index = faiss.read_index(str(self.p_faiss_index))

        tqdm.write(f"Loaded Faiss index {index}")
        return index

    def run(
        self,
        rank: int = 0, world_size: int = 1,
        segment_duration_sec: float = 1.,
        num_sampled_segment_frames: int = 16,
        segment_overlap_sec: float = .5,
        num_skip_first_segments: int = 0,  # for debugging
    ):
        self.device = f'cuda:{rank % 8}'
        self.texts_normal, self.texts_anomalous = self._load_captions()
        self.model = self._load_imagebind_model()
        self.embs_normal, self.embs_anomalous = self._create_or_load_imagebind_embeddings()
        self.index = self._create_or_load_faiss_index()

        k = 10
        texts = self.texts_normal + self.texts_anomalous
        df_ann_test_rank = self.df_ann_test.iloc[rank::world_size].reset_index(drop=True)
        sr_num_frames_rank = self.sr_num_frames.iloc[rank::world_size]
        num_segment_frames = int(segment_duration_sec * 30)
        num_overlap_frames = int(segment_overlap_sec * 30)
        sr_num_segments = (sr_num_frames_rank - num_segment_frames) // (num_segment_frames - num_overlap_frames) + 1
        num_total_segments = sr_num_segments.sum()
        tqdm.write(f"Rank {rank} of {world_size} will process {len(df_ann_test_rank)} videos with {num_total_segments} segments")

        pbar_segments = tqdm(
            position=2,
            total=num_total_segments,
            file=sys.stdout,
        )

        for idx, row in tqdm(
            df_ann_test_rank.iterrows(), total=len(df_ann_test_rank), position=0, mininterval=.001, file=sys.stdout,
        ):
            p_video = self.p_videos_root / row['raw_rel_video_path']
            p_out = (self.p_outdir / row['rel_video_path']).with_suffix('.json')
            p_out.parent.mkdir(exist_ok=True, parents=True)
            # num_video_segments = compute_num_segments(p_video, segment_duration_sec, segment_overlap_sec)
            num_video_segments = sr_num_segments[row['raw_rel_video_path']]
            if p_out.exists():
                pbar_segments.update(num_video_segments)
                continue

            segment_records = []
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
                frames = self.video_transform(frames)
                frames = frames.to(self.device)
                with torch.inference_mode():
                    input_frames = rearrange(frames, 'c (t s) h w -> s c t h w', t=2)  # T=2 fixed as ImageBind expects 2 frames per clip
                    emb_video = self.model.forward({ModalityType.VISION: input_frames[None]})[ModalityType.VISION]
                    emb_video = emb_video.cpu().numpy()
                dot_products, indices = self.index.search(emb_video, k)
                selected_texts = np.take(texts, indices)
                is_anomalous = indices > len(self.texts_normal)

                segment_record = {
                    'segment_idx': segment_dict['segment_idx'],
                    'total_segments': segment_dict['total_segments'],
                    'start_idx': segment_dict['segment_start_idx'],
                    'end_idx': segment_dict['segment_end_idx'],
                    'dot_products': dot_products[0].tolist(),
                    'indices': indices[0].tolist(),
                    'captions': selected_texts[0].tolist(),
                    'is_anomalous': is_anomalous[0].tolist(),
                }
                segment_records.append(segment_record)
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
                'segment_records': segment_records,
            }
            json.dump(video_record, p_out.open('w'), indent=2)

    def evaluate(
        self,
        force: bool = False,
    ):
        ann_vad = {}
        for idx, row in self.df_ann_test.iterrows():
            num_frames = self.sr_num_frames[row['raw_rel_video_path']]
            bin_label = np.zeros(num_frames, dtype=np.int32)
            bin_label[row['s1']:row['e1']] = 1
            if row['s2'] != -1:
                bin_label[row['s2']:row['e2']] = 1
            ann_vad[row['raw_rel_video_path']] = bin_label

        preds = {}
        for raw_rel_video_path, bin_label in ann_vad.items():
            p_json = (self.p_outdir / raw_rel_video_path).with_suffix('.json')
            if not p_json.exists():
                if force:
                    preds[raw_rel_video_path] = np.zeros(len(bin_label))
                    continue
            video_record = json.load(p_json.open())
            segment_records = video_record['segment_records']
            pred = np.zeros(len(bin_label))
            overlap_count = np.zeros(len(bin_label))
            for segment_record in segment_records:
                segment_score = int(segment_record['is_anomalous'][0])  # anomality of the top 1 caption
                pred[segment_record['start_idx']:segment_record['end_idx']+1] += segment_score
                overlap_count[segment_record['start_idx']:segment_record['end_idx']+1] += 1
            overlap_count = np.maximum(overlap_count, 1)
            preds[raw_rel_video_path] = pred / overlap_count

        # compute AUC
        all_preds, all_labels = np.array([]), np.array([])
        for raw_rel_video_path, bin_label in ann_vad.items():
            all_preds = np.concatenate([all_preds, preds[raw_rel_video_path]])
            all_labels = np.concatenate([all_labels, bin_label])
        auc = roc_auc_score(all_labels.astype(int), all_preds)
        print(auc)

if __name__ == '__main__':
    import fire
    fire.Fire(Main)
