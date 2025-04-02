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

import faiss
import decord
decord.bridge.reset_bridge()

from einops import rearrange, reduce

sys.path = ['/code/libs/imagebind'] + sys.path  # Prepend to avoid conflicts with installed imagebind
from imagebind import data
from imagebind.models.imagebind_model import ModalityType
from imagebind.models.imagebind_model import ImageBindModel

sys.path = ['/code/libs/languagebind'] + sys.path
from languagebind import LanguageBindVideo, LanguageBindVideoTokenizer


@torch.inference_mode()
def forward_video(model, frames):
    if frames.ndim == 4:
        frames = frames.unsqueeze(0)

    if type(model) == ImageBindModel:
        frames = rearrange(frames, 'b c (t s) h w -> b s c t h w', t=2)  # T=2 fixed as ImageBind expects 2 frames per clip
        return model.forward({ModalityType.VISION: frames})[ModalityType.VISION]

    elif type(model) == LanguageBindVideo:
        t = 8  # LanguageBind expects 8 frames per clip
        s = frames.shape[2] // t
        frames = rearrange(frames, 'b c (t s) h w -> (b s) c t h w', t=t)
        vision_outputs = model.vision_model.forward(pixel_values=frames)
        image_embeds = vision_outputs[1]  # [B x S, D_inter]
        image_embeds = model.visual_projection.forward(image_embeds)  # [B x S, D_out]
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        image_embeds = reduce(image_embeds, '(b s) d -> b d', 'mean', s=s)
        return image_embeds  # [B, D_out]


@torch.inference_mode()
def forward_text(model, texts, tokenizer=None):
    device = next(model.parameters()).device
    if type(model) == ImageBindModel:
        text_tokens = data.load_and_transform_text(texts, device)
        return model.forward({ModalityType.TEXT: text_tokens})[ModalityType.TEXT]

    elif type(model) == LanguageBindVideo:
        assert tokenizer is not None
        encoding = tokenizer(
            texts,
            max_length=77, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        for k, v in encoding.items():
            if isinstance(v, torch.Tensor):
                encoding[k] = v.to(device)
        text_outputs = model.text_model.forward(**encoding)
        text_embeds = text_outputs[1]
        text_embeds = model.text_projection.forward(text_embeds)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds


class Main:
    def __init__(self):
        ### dataset paths
        self.p_dataroot = Path('/datasets/UCF_Crimes')
        self.p_annroot = Path('/code/data/annotations')
        self.p_videos_root = self.p_dataroot / 'Videos'

        # caption paths
        self.p_captions_root = Path("/code/output/psuedo-captions/gpt-4o/00-rich-context")
        self.p_captions_dir = self.p_captions_root / "captions"
        assert self.p_captions_dir.exists(), f"Captions directory {self.p_captions_dir} does not exist"

    def _init_output_paths(self, retriever_name: str = 'imagebind'):
        self.p_outdir_embeddings = self.p_captions_root / f"embeddings/{retriever_name}"
        self.p_normal = self.p_outdir_embeddings / "embs_normal.npy"
        self.p_anomalous = self.p_outdir_embeddings / "embs_anomalous.npy"
        self.p_outdir_scored = self.p_captions_root / "scored"

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

    def _load_retriever_model(self, retriever_name: str = 'imagebind', device: str = 'cuda'):
        print(f"Loading {retriever_name}...", flush=True)
        tokenizer = None
        if retriever_name == 'imagebind':
            os.chdir('/code/libs/imagebind')  # tokenizer path is hardcoded in imagebind
            model = ImageBindModel.from_pretrained("nielsr/imagebind-huge")
        elif retriever_name == 'languagebind':
            pretrained_ckpt = 'LanguageBind/LanguageBind_Video_Huge_V1.5_FT'
            model = LanguageBindVideo.from_pretrained(pretrained_ckpt)
            tokenizer = LanguageBindVideoTokenizer.from_pretrained(pretrained_ckpt)
        model.eval()
        model.to(device)
        print(f"Loaded {retriever_name}", flush=True)
        return model, tokenizer

    def _create_or_load_caption_embeddings(self, model=None, tokenizer=None, rank=0):
        tqdm.write("Creating or loading embeddings...")
        num_normal = len(self.texts_normal)
        num_anomalous = len(self.texts_anomalous)
        stride = 2048

        if rank == 0 and not self.p_normal.exists():
            tqdm.write(f'[Rank {rank}] Creating embeddings for normal captions...')
            # texts = [f'Normal: {text}' for text in self.texts_normal]
            texts = self.texts_normal
            embs_normal = []
            for i in trange(0, num_normal, stride, desc="Normal"):
                emb_normal = forward_text(model, texts[i:i+stride], tokenizer=tokenizer)
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
            # texts = [f'Anomaly: {text}' for text in self.texts_anomalous]
            texts = self.texts_anomalous
            embs_anomalous = []
            for i in trange(0, num_anomalous, stride, desc="Anomalous"):
                emb_anomalous = forward_text(model, texts[i:i+stride], tokenizer=tokenizer)
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

    def _create_caption_index(
        self,
        embs_normal,
        embs_anomalous,
        anomalous_scale=1.,
        rank=0
    ) -> faiss.Index:
        print("Creating or loading Faiss index...")
        d = embs_normal.shape[1]
        num_normal = len(self.texts_normal)
        num_anomalous = len(self.texts_anomalous)

        print(f'[Rank {rank}] Creating Faiss index...')
        index = faiss.IndexFlatIP(d)
        index = faiss.IndexIDMap2(index)
        index.add_with_ids(embs_normal, np.arange(num_normal))
        index.add_with_ids(embs_anomalous * anomalous_scale, np.arange(num_normal, num_normal + num_anomalous))

        print(f"Loaded Faiss index {index}")
        return index

    def extract_embeddings_per_segment(
        self,
        rank: int = 0, world_size: int = 1,
        segment_duration_sec: float = 1.,
        segment_overlap_sec: float = .5,
        num_sampled_segment_frames: int = 16,
        retriever_name: str = 'imagebind',
    ):
        device = f'cuda:{rank % 8}'
        self._init_output_paths(retriever_name)
        p_outdir_segment_embeddings_with_options = self.p_outdir_embeddings / f"dur={segment_duration_sec:.1f}_ol={segment_overlap_sec:.1f}_fs={num_sampled_segment_frames}" / 'segments'
        p_outdir_segment_embeddings_with_options.mkdir(exist_ok=True, parents=True)
        print('Outdir:', p_outdir_segment_embeddings_with_options, flush=True)

        model, tokenizer = self._load_retriever_model(retriever_name, device)

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
            dl, total=len(ds), mininterval=.1, miniters=1, maxinterval=.5, file=sys.stdout,
        ):
            segment_info = segment_data[0]['segment_info']
            p_video = segment_info['p_video']
            p_out_embedding = (
                p_outdir_segment_embeddings_with_options
                / ('Normal' if 'Normal' in p_video.parent.name else p_video.parent.name)
                / p_video.stem
                / f'{segment_info["segment_idx"]:04d}.npy'
            )
            p_out_embedding.parent.mkdir(exist_ok=True, parents=True)
            frames = segment_data[0]['frames'].to(device)

            emb_segment = forward_video(model, frames[None])[0]
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
        retriever_name: str = 'imagebind',
    ):
        self._init_output_paths(retriever_name)
        assert self.p_outdir_embeddings.exists(), f"Embeddings directory {self.p_outdir_embeddings} does not exist"
        self.p_outdir_scored.mkdir(exist_ok=True, parents=True)

        if self.p_normal.exists() and self.p_anomalous.exists():
            model, tokenizer = None, None
        else:
            model, tokenizer = self._load_retriever_model(retriever_name)

        num_segment_frames = int(segment_duration_sec * 30)
        num_overlap_frames = int(segment_overlap_sec * 30)
        self.texts_normal, self.texts_anomalous = self._load_captions()
        embs_normal, embs_anomalous = self._create_or_load_caption_embeddings(
            model=model, tokenizer=tokenizer, rank=rank)
        index = self._create_caption_index(
            embs_normal, embs_anomalous,
            anomalous_scale=anomalous_scale, rank=rank
        )

        # project embeddings for calibration
        # emb_normal_mean = embs_normal.mean(axis=0)
        # emb_normal_mean /= np.linalg.norm(emb_normal_mean)
        # emb_anomalous_mean = embs_anomalous.mean(axis=0)
        # emb_anomalous_mean /= np.linalg.norm(emb_anomalous_mean)
        # emb_proj = emb_anomalous_mean - emb_normal_mean
        # emb_proj /= np.linalg.norm(emb_proj)
        # emb_proj = emb_proj.astype(np.float32)

        texts = self.texts_normal + self.texts_anomalous

        p_segment_embeddings_with_options = self.p_outdir_embeddings / f"dur={segment_duration_sec:.1f}_ol={segment_overlap_sec:.1f}_fs={num_sampled_segment_frames}" / 'segments'
        p_segment_embeddings_with_options.mkdir(exist_ok=True, parents=True)

        df_ann_test_rank = df_ann_test.iloc[rank::world_size].reset_index(drop=True)
        pbar = tqdm(
            df_ann_test_rank.iterrows(),
            total=len(df_ann_test_rank),
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

            # calibrate embeddings by rotating the segment vector onto the space having the same distance to the normal and anomalous mean
            # norms = np.linalg.norm(segment_embeddings, axis=-1, keepdims=True)
            # segment_embeddings -= np.einsum('i,j,hj->hi', emb_proj, emb_proj, segment_embeddings)
            # segment_embeddings *= norms / np.linalg.norm(segment_embeddings, axis=-1, keepdims=True)

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

    def evaluate(
        self,
        retriever_name: str = 'imagebind',
        force: bool = False,
    ):
        self._init_output_paths(retriever_name)
        assert self.p_outdir_scored.exists(), f"Scored directory {self.p_outdir_scored} does not exist"

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
