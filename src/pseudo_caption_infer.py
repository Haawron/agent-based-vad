import os
import sys
sys.path.append('/code/src')
from helper import get_segments, df_ann_test, sr_num_frames_test, SegmentDataset

import time
import json
from pathlib import Path
from tqdm import trange, tqdm

import pandas as pd
import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import roc_auc_score

# import faiss
import decord
decord.bridge.reset_bridge()

from einops import rearrange, reduce

sys.path = ['/code/libs/imagebind'] + sys.path  # Prepend to avoid conflicts with installed imagebind
from imagebind import data
from imagebind.models.imagebind_model import ModalityType
from imagebind.models.imagebind_model import ImageBindModel

sys.path = ['/code/libs/languagebind'] + sys.path
from languagebind import LanguageBindVideo, LanguageBindVideoTokenizer


def load_retriever_model(
    retriever_name='imagebind',
    device='cuda',
):
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


def load_captions(p_captions_dir):
    print("Loading captions...")
    texts_normal, texts_anomalous, categories_anomalous = [], [], []
    for p_json in p_captions_dir.glob("*.json"):
        with p_json.open("r") as f:
            captions = json.load(f)
            for caption_set in captions['descriptions']:
                texts_normal.append(caption_set['normal']['description'])
                texts_anomalous.append(caption_set['anomalous']['description'])
                categories_anomalous.append(caption_set['anomalous']['category'])
    print(f'Loaded {len(texts_normal)} normal captions and {len(texts_anomalous)} anomalous captions')
    return texts_normal, texts_anomalous, categories_anomalous


# def create_caption_index(
#     embs_normal,
#     embs_anomalous,
#     anomalous_scale=1.,
#     rank=0
# ) -> faiss.Index:
#     print("Creating or loading Faiss index...")
#     d = embs_normal.shape[1]
#     num_normal = embs_normal.shape[0]
#     num_anomalous = embs_anomalous.shape[0]

#     print(f'[Rank {rank}] Creating Faiss index...')
#     index = faiss.IndexFlatIP(d)
#     index = faiss.IndexIDMap2(index)
#     index.add_with_ids(embs_normal, np.arange(num_normal))
#     index.add_with_ids(embs_anomalous * anomalous_scale, np.arange(num_normal, num_normal + num_anomalous))

#     print(f"Loaded Faiss index {index}")
#     return index


class Inner:
    def __init__(
        self,
        retriever_name: str = 'imagebind',
        rank: int = 0,
        world_size: int = 1,
    ):
        self.retriever_name = retriever_name
        self.device = f'cuda:{rank % 8}'
        self.rank = rank
        self.world_size = world_size

        # dataset paths
        self.p_dataroot = Path('/datasets/UCF_Crimes')
        self.p_annroot = Path('/code/data/annotations')
        self.p_videos_root = self.p_dataroot / 'Videos'

        # caption paths
        self.p_captions_root = Path("/code/output/psuedo-captions/gpt-4o/00-rich-context")
        # self.p_captions_root = Path("/code/output/psuedo-captions/gpt-4o/01-rich-context-1M")
        self.p_captions_dir = self.p_captions_root / "captions"
        assert self.p_captions_dir.exists(), f"Captions directory {self.p_captions_dir} does not exist"

        # output paths
        self.p_outdir_embeddings = self.p_captions_root / f"embeddings/{retriever_name}"
        self.p_normal = self.p_outdir_embeddings / "embs_normal.pt"
        self.p_anomalous = self.p_outdir_embeddings / "embs_anomalous.pt"
        self.p_outdir_scored = self.p_captions_root / "scored"

        self.model, self.tokenizer = None, None

    @torch.inference_mode()
    def forward_video(self, frames):
        if frames.ndim == 4:
            frames = frames.unsqueeze(0)

        if self.retriever_name == 'imagebind':
            frames = rearrange(frames, 'b c (t s) h w -> b s c t h w', t=2)  # T=2 fixed as ImageBind expects 2 frames per clip
            image_embeds = self.model.forward({ModalityType.VISION: frames})[ModalityType.VISION]

        elif self.retriever_name == 'languagebind':
            t = 8  # LanguageBind expects 8 frames per clip
            s = frames.shape[2] // t
            frames = rearrange(frames, 'b c (t s) h w -> (b s) c t h w', t=t)
            vision_outputs = self.model.vision_model.forward(pixel_values=frames)
            image_embeds = vision_outputs[1]  # [B x S, D_inter]
            image_embeds = self.model.visual_projection.forward(image_embeds)  # [B x S, D_out]
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            image_embeds = reduce(image_embeds, '(b s) d -> b d', 'mean', s=s)

        return image_embeds  # [B, D_out]

    def tokenize_text(self, texts):
        if self.retriever_name == 'imagebind':
            encoding = data.load_and_transform_text(texts, device=self.device)

        elif self.retriever_name == 'languagebind':
            encoding = self.tokenizer(
                texts,
                max_length=77, padding='max_length',
                truncation=True, return_tensors='pt'
            )
            for k, v in encoding.items():
                if isinstance(v, torch.Tensor):
                    encoding[k] = v.to(self.device)

        return encoding

    @torch.inference_mode()
    def forward_text(self, encoding):
        if self.retriever_name == 'imagebind':
            text_embeds = self.model.forward({ModalityType.TEXT: encoding})[ModalityType.TEXT]

        elif self.retriever_name == 'languagebind':
            text_outputs = self.model.text_model.forward(**encoding)
            text_embeds = text_outputs[1]
            text_embeds = self.model.text_projection.forward(text_embeds)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        return text_embeds

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

        self.model, _ = load_retriever_model(retriever_name, device)

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

            emb_segment = self.forward_video(frames[None])[0]
            emb_segment = emb_segment.cpu().numpy()
            np.save(p_out_embedding, emb_segment)

    @torch.inference_mode()
    def extract_caption_embeddings(
        self,
        rank: int = 0, world_size: int = 1,
        retriever_name: str = 'imagebind',
    ):
        def get_p_rank(p, r):
            return p.with_name(p.stem + f'_{r}.{p.suffix}')

        def extract(texts):
            embs = []
            for i in trange(0, len(texts), stride):
                texts_batch = texts[i:i+stride]
                text_tokens = self.tokenize_text(texts_batch)
                emb = self.forward_text(text_tokens)
                embs.append(emb.cpu())
            embs = torch.cat(embs, dim=0)
            return embs

        def gather_and_save_embeddings(p_out, r):
            if r == 0:
                output = {
                    'texts': [],
                    'embeddings': [],
                }
                for rr in range(world_size):
                    p = get_p_rank(p_out, rr)
                    while not p.exists():
                        tqdm.write(f"Waiting for rank {rr} to finish saving embeddings...")
                        time.sleep(5)
                    output_rank = torch.load(p)
                    output['texts'].extend(output_rank['texts'])
                    output['embeddings'].append(output_rank['embeddings'])
                output['embeddings'] = torch.cat(output['embeddings'], dim=0)
                torch.save(output, p_out)

            while not p_out.exists():
                tqdm.write(f"Waiting for rank 0 to finish saving embeddings...")
                time.sleep(5)

        # load captions and model
        stride = 2048
        texts_normal, texts_anomalous, categories_anomalous = load_captions(self.p_captions_dir)
        self.model, self.tokenizer = load_retriever_model(retriever_name, self.device)

        # distribute the workload
        texts_normal_subset = texts_normal[rank::world_size]
        texts_anomalous_subset = texts_anomalous[rank::world_size]
        categories_anomalous_subset = categories_anomalous[rank::world_size]
        num_normal = len(texts_normal_subset)
        num_anomalous = len(texts_anomalous_subset)
        msg = f"Extracting {num_normal} normal and {num_anomalous} anomalous captions..."
        if world_size > 1:
            msg += f" (rank {rank} of {world_size})"
        tqdm.write(msg)

        # prompting
        texts_normal_subset = [
            f'Normal: {text}'
            for text in texts_normal_subset
        ]
        texts_anomalous_subset = [
            f'Anomalous: {text}'
            for text, category in zip(texts_anomalous_subset, categories_anomalous_subset)
        ]

        # create output directories
        self.p_outdir_embeddings.mkdir(exist_ok=True, parents=True)
        self.p_normal.unlink(missing_ok=True)
        self.p_anomalous.unlink(missing_ok=True)
        p_normal_rank = get_p_rank(self.p_normal, rank)
        p_anomalous_rank = get_p_rank(self.p_anomalous, rank)
        p_normal_rank.unlink(missing_ok=True)
        p_anomalous_rank.unlink(missing_ok=True)

        for texts, p_rank, p in zip(
            [texts_normal_subset, texts_anomalous_subset],
            [p_normal_rank, p_anomalous_rank],
            [self.p_normal, self.p_anomalous],
        ):
            output_rank = {
                'texts': texts,
                'embeddings': extract(texts),
            }
            torch.save(output_rank, p_rank)
            gather_and_save_embeddings(p, rank)
            p_rank.unlink(missing_ok=True)
            tqdm.write(f"Rank {rank} finished saving captions to {p}")

    @torch.inference_mode()
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
        assert self.p_outdir_embeddings.exists(), f"Embeddings directory {self.p_outdir_embeddings} does not exist"
        self.p_outdir_scored.mkdir(exist_ok=True, parents=True)

        if not self.p_normal.exists() or not self.p_anomalous.exists():
            self.model, self.tokenizer = load_retriever_model(retriever_name, self.device)

        num_segment_frames = int(segment_duration_sec * 30)
        num_overlap_frames = int(segment_overlap_sec * 30)
        texts_normal, texts_anomalous, categories_anomalous = load_captions(self.p_captions_dir)
        # embs_normal = torch.load(self.p_normal)['embeddings'].numpy()
        # embs_anomalous = torch.load(self.p_anomalous)['embeddings'].numpy()
        # index = create_caption_index(
        #     embs_normal, embs_anomalous,
        #     anomalous_scale=anomalous_scale, rank=rank
        # )
        normals = torch.load(self.p_normal)
        anomalouses = torch.load(self.p_anomalous)
        embs_normal = normals['embeddings']
        embs_anomalous = anomalouses['embeddings'] * anomalous_scale
        embs = torch.cat([embs_normal, embs_anomalous], dim=0).numpy()
        texts_normal = normals['texts']
        texts_anomalous = anomalouses['texts']
        texts = texts_normal + texts_anomalous

        # project embeddings for calibration
        # emb_normal_mean = embs_normal.mean(axis=0)
        # emb_normal_mean /= np.linalg.norm(emb_normal_mean)
        # emb_anomalous_mean = embs_anomalous.mean(axis=0)
        # emb_anomalous_mean /= np.linalg.norm(emb_anomalous_mean)
        # emb_proj = emb_anomalous_mean - emb_normal_mean
        # emb_proj /= np.linalg.norm(emb_proj)
        # emb_proj = emb_proj.astype(np.float32)

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

            # calibrate embeddings
            # norms = np.linalg.norm(segment_embeddings, axis=-1, keepdims=True)
            # segment_embeddings -= np.einsum('i,j,hj->hi', emb_proj, emb_proj, segment_embeddings)
            # segment_embeddings *= norms / np.linalg.norm(segment_embeddings, axis=-1, keepdims=True)

            # dot_products, indices = index.search(segment_embeddings, num_captions_per_segment)
            dot_products = np.dot(segment_embeddings, embs.T)
            indices = np.argsort(-dot_products, axis=-1)[:, :num_captions_per_segment]
            dot_products = np.take_along_axis(dot_products, indices, axis=-1)

            selected_texts = np.take(texts, indices)
            is_anomalous = indices > len(texts_normal)
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


class Main:
    def __init__(self):
        pass

    def extract_embeddings_per_segment(
        self,
        rank: int = 0, world_size: int = 1,
        segment_duration_sec: float = 1.,
        segment_overlap_sec: float = .5,
        num_sampled_segment_frames: int = 16,
        retriever_name: str = 'imagebind',
    ):
        inner = Inner(retriever_name, rank, world_size)
        inner.extract_embeddings_per_segment(
            rank=rank, world_size=world_size,
            segment_duration_sec=segment_duration_sec,
            segment_overlap_sec=segment_overlap_sec,
            num_sampled_segment_frames=num_sampled_segment_frames,
            retriever_name=retriever_name,
        )

    def extract_caption_embeddings(
        self,
        rank: int = 0, world_size: int = 1,
        retriever_name: str = 'imagebind',
    ):
        inner = Inner(retriever_name, rank, world_size)
        inner.extract_caption_embeddings(
            rank=rank, world_size=world_size,
            retriever_name=retriever_name,
        )

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
        inner = Inner(retriever_name, rank, world_size)
        inner.match_captions_per_segment(
            rank=rank, world_size=world_size,
            segment_duration_sec=segment_duration_sec,
            segment_overlap_sec=segment_overlap_sec,
            num_sampled_segment_frames=num_sampled_segment_frames,
            num_captions_per_segment=num_captions_per_segment,
            anomalous_scale=anomalous_scale,
            retriever_name=retriever_name,
        )
        inner.evaluate(
            retriever_name=retriever_name,
            force=False,
        )

    def evaluate(
        self,
        retriever_name: str = 'imagebind',
        force: bool = False,
    ):
        inner = Inner(retriever_name)
        inner.evaluate(
            retriever_name=retriever_name,
            force=force,
        )


if __name__ == '__main__':
    import fire
    fire.Fire(Main)
