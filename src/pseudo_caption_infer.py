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
from PIL import Image

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


class Inner:
    def __init__(
        self,
        caption_type: str = '00-rich-context',
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
        self.p_captions_root = Path(f"/code/output/psuedo-captions/gpt-4o/{caption_type}")
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

    def extract_tmf(
        self,
        num_sampled_frames: int = 32,
    ):
        p_videos_rootdir = Path(f"/code/data/UCF_Crimes/Videos")
        p_outdir_tmf = Path(f'/code/output/tmf/ucf-crime/{num_sampled_frames}')
        p_outdir_tmf.mkdir(exist_ok=True, parents=True)
        print('Outdir:', p_outdir_tmf, flush=True)

        for idx, row in tqdm(df_ann_test.iterrows(), total=len(df_ann_test), file=sys.stdout):
            raw_rel_video_path = row['raw_rel_video_path']
            p_out_tmf = (p_outdir_tmf / raw_rel_video_path).with_suffix('.jpg')
            p_out_tmf.parent.mkdir(exist_ok=True, parents=True)
            p_video = p_videos_rootdir / raw_rel_video_path
            vr = decord.VideoReader(str(p_video), ctx=decord.cpu(0))
            num_frames = len(vr)
            idxs = np.linspace(0, num_frames - 1, num=num_sampled_frames, dtype=int)
            frames = vr.get_batch(idxs.tolist()).asnumpy()
            bg = np.median(frames, axis=0)
            bg = Image.fromarray(bg.astype(np.uint8))
            bg.save(p_out_tmf, format='jpeg')

    @torch.inference_mode()
    def extract_tmf_embeddings(
        self,
        rank: int = 0, world_size: int = 1,
        num_tmf_frames: int = 32,
        retriever_name: str = 'imagebind',
    ):
        device = f'cuda:{rank % 8}'
        p_tmf_dir = Path(f"/code/output/tmf/ucf-crime/{num_tmf_frames}")
        assert p_tmf_dir.exists(), f"TMF directory {p_tmf_dir} does not exist"
        p_outdir_tmf_embeddings_with_options = self.p_outdir_embeddings / f"tmf_frames={num_tmf_frames}"
        p_outdir_tmf_embeddings_with_options.mkdir(exist_ok=True, parents=True)
        print('Outdir:', p_outdir_tmf_embeddings_with_options, flush=True)

        self.model, _ = load_retriever_model(retriever_name, device)
        transform = SegmentDataset().video_transform

        df_ann_test_rank = df_ann_test.iloc[rank::world_size].reset_index(drop=True)
        for idx, row in tqdm(df_ann_test_rank.iterrows(), total=len(df_ann_test_rank), file=sys.stdout):
            raw_rel_video_path = row['raw_rel_video_path']
            rel_video_path = row['rel_video_path']
            p_out_embedding = (p_outdir_tmf_embeddings_with_options / rel_video_path).with_suffix('.pt')
            p_out_embedding.parent.mkdir(exist_ok=True, parents=True)
            p_tmf = (p_tmf_dir / raw_rel_video_path).with_suffix('.jpg')
            if not p_tmf.exists():
                print(f"TMF image {p_tmf} does not exist")
                continue
            bg = torch.tensor(np.array(Image.open(p_tmf).convert('RGB'))).float()  # [H, W, C]
            bg = bg[None][[0] * 8]  # [8, H, W, C]
            bg = bg.permute(3, 0, 1, 2)  # [C, 8, H, W]
            bg = transform(bg)
            bg = bg.to(device)
            emb_bg = self.forward_video(bg)[0]  # [D_out]
            emb_bg = emb_bg.cpu()
            torch.save(emb_bg, p_out_embedding)
            tqdm.write(f"Rank {rank} finished saving TMF embedding to {p_out_embedding}")

    def extract_embeddings_per_segment(
        self,
        rank: int = 0, world_size: int = 1,
        segment_duration_sec: float = 1.,
        segment_overlap_sec: float = .5,
        num_sampled_segment_frames: int = 16,
        retriever_name: str = 'imagebind',
    ):
        device = f'cuda:{rank % 8}'
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

        normals = torch.load(self.p_normal)
        anomalouses = torch.load(self.p_anomalous)
        embs_normal = normals['embeddings'].numpy()  # [NUM_NORMAL, D_OUT]
        embs_anomalous = anomalouses['embeddings'].numpy() * anomalous_scale  # [NUM_ANOMALOUS, D_OUT]
        embs = np.concatenate([embs_normal, embs_anomalous], axis=0)
        texts_normal = normals['texts']
        texts_anomalous = anomalouses['texts']
        texts = texts_normal + texts_anomalous

        ##################################################################
        # embeddings for calibration
        num_normals = len(embs_normal)
        num_anomalouses = len(embs_anomalous)

        emb_normal_mean = embs_normal.mean(axis=0)
        emb_normal_mean_unit = emb_normal_mean / np.linalg.norm(emb_normal_mean)
        # _, _, embs_normal_sing = np.linalg.svd(embs_normal, full_matrices=False)

        emb_anomalous_mean = embs_anomalous.mean(axis=0)
        emb_anomalous_mean_unit = emb_anomalous_mean / np.linalg.norm(emb_anomalous_mean)
        p_svd_anom = self.p_anomalous.with_name(self.p_anomalous.stem + f'_svd.{self.p_anomalous.suffix}')
        if p_svd_anom.exists():
            embs_anomalous_sing = torch.load(p_svd_anom)
        else:
            _, _, embs_anomalous_sing = np.linalg.svd(embs_anomalous, full_matrices=False)
            if rank == 0:
                torch.save(embs_anomalous_sing, p_svd_anom)
        # _, _, embs_anomalous_pc = np.linalg.svd(embs_anomalous - emb_anomalous_mean, full_matrices=False)

        emb_text_mean = (num_normals * emb_normal_mean + num_anomalouses * emb_anomalous_mean) / (num_normals + num_anomalouses)
        emb_text_mean_unit = emb_text_mean / np.linalg.norm(emb_text_mean)

        emb_delta = emb_anomalous_mean - emb_normal_mean
        emb_proj_delta = emb_delta / np.linalg.norm(emb_delta)

        emb_delta_units = emb_anomalous_mean_unit - emb_normal_mean_unit
        emb_proj_delta_units = emb_delta_units / np.linalg.norm(emb_delta_units)

        ##################################################################

        p_segment_embeddings_with_options = self.p_outdir_embeddings / f"dur={segment_duration_sec:.1f}_ol={segment_overlap_sec:.1f}_fs={num_sampled_segment_frames}" / 'segments'
        p_segment_embeddings_with_options.mkdir(exist_ok=True, parents=True)

        def distribute_workload(df, rank, world_size):
            df = df.copy()
            df = df.sort_values(by='num_frames', ascending=False)
            all_rows = [[] for _ in range(world_size)]
            sum_rows = [0] * world_size
            for idx, row in df.iterrows():
                min_idx = np.argmin(sum_rows)
                all_rows[min_idx].append(row)
                sum_rows[min_idx] += row['num_frames']
            all_rows = [pd.DataFrame(rows).sort_values(by='video') for rows in all_rows]
            return all_rows[rank]

        df_ann_test_rank = distribute_workload(
            df_ann_test.join(sr_num_frames_test, on='raw_rel_video_path'), rank, world_size)
        pbar = tqdm(
            df_ann_test_rank.iterrows(),
            total=len(df_ann_test_rank),
            file=sys.stdout,
            position=1
        )
        for idx, row in pbar:
            p_out_score = (self.p_outdir_scored / row['rel_video_path']).with_suffix('.json')
            p_out_score.parent.mkdir(exist_ok=True, parents=True)
            p_segment_dir = p_segment_embeddings_with_options / row['rel_video_path'].split('.')[0]
            p_segments = sorted(p_segment_dir.glob('*.npy'))
            embs_segment = []
            for p_segment in p_segments:
                embs_segment.append(np.load(p_segment))
            embs_segment = np.stack(embs_segment, axis=0)

            ##################################################################################################
            # calibrate embeddings

            # 74.33
            # norms = np.linalg.norm(embs_segment, axis=-1, keepdims=True)
            # embs_segment -= np.einsum('i,j,hj->hi', emb_proj_delta_units, emb_proj_delta_units, embs_segment)
            # embs_segment *= norms / np.linalg.norm(embs_segment, axis=-1, keepdims=True)

            #
            # embs_segment -= np.einsum('i,j,hj->hi', emb_anomalous_mean_unit, emb_anomalous_mean_unit, embs_segment)

            ################
            # using segment mean
            emb_segment_mean = embs_segment.mean(axis=0)

            # windowed mean
            window = 120
            embs_segment_mean = np.pad(embs_segment, ((window // 2 + 1, window // 2 - 1), (0, 0)), mode='constant')
            embs_segment_mean = np.cumsum(embs_segment_mean, axis=0)
            embs_segment_mean = embs_segment_mean[window:] - embs_segment_mean[:-window]
            counts = np.convolve(np.ones(embs_segment.shape[0]), np.ones(window), mode='same')
            if counts.shape[0] > embs_segment_mean.shape[0]:
                diff = counts.shape[0] - embs_segment_mean.shape[0]
                counts = counts[diff//2:-(diff-diff//2)]
            embs_segment_mean /= counts[:, None]

            # 54.68
            # embs_segment += np.einsum('i,j,j->i', emb_anomalous_mean_unit, emb_anomalous_mean_unit, emb_segment_mean)
            # embs_segment -= np.einsum('i,j,j->i', emb_normal_mean_unit, emb_normal_mean_unit, emb_segment_mean)

            # 76.83
            # embs_segment -= np.einsum('i,j,j->i', emb_delta_units, emb_delta_units, emb_segment_mean)
            #
            # embs_segment -= np.einsum('i,j,hj->hi', emb_delta_units, emb_delta_units, embs_segment_mean)

            # 75.71
            # embs_segment -= np.einsum('i,j,j->i', emb_proj_delta_units, emb_proj_delta_units, emb_segment_mean)

            # 77.50 -> 78.20
            # embs_segment -= np.einsum('i,j,j->i', emb_anomalous_mean_unit, emb_anomalous_mean_unit, emb_segment_mean)
            # 78.36 (120), 78.41 (240)
            # embs_segment -= np.einsum('i,j,hj->hi', emb_anomalous_mean_unit, emb_anomalous_mean_unit, embs_segment_mean)
            # 79.72 (120), 79.62 (240)
            embs_segment -= np.einsum('gi,gj,hj->hi', embs_anomalous_sing[:2], embs_anomalous_sing[:2], embs_segment_mean)

            # 64.43: BG를 뺐을 때 더 낮은 거 보면 BG에도 anomality 정보가 들어있다는 뜻임
            # 아닌가 BG가 anomality context를 더 잘 담고 있다고 생각하면 되나?
            # 암튼 문제 푸는 데에는 BG가 더 중요한 벡터임
            # embs_segment -= emb_segment_mean
            # 62.30
            # emb_segment_mean /= np.linalg.norm(emb_segment_mean)
            # embs_segment -= np.einsum('i,j,hj->hi', emb_segment_mean, emb_segment_mean, embs_segment)

            # 77.56
            # embs_segment -= np.einsum('hi,hj,j->i', embs_anomalous_sing[:1], embs_anomalous_sing[:1], emb_segment_mean)
            # 79.52
            # embs_segment -= np.einsum('hi,hj,j->i', embs_anomalous_sing[:2], embs_anomalous_sing[:2], emb_segment_mean)
            # 79.44
            # emb_seg_mean_anom = np.einsum('hi,hj,j->i', embs_anomalous_sing[:2], embs_anomalous_sing[:2], emb_segment_mean)
            # emb_seg_mean_anom /= np.linalg.norm(emb_seg_mean_anom)
            # embs_segment -= np.einsum('i,j,hj->hi', emb_seg_mean_anom, emb_seg_mean_anom, embs_segment)

            ################
            # using bg
            p_tmf = (self.p_outdir_embeddings / 'tmf_frames=64' / row['rel_video_path']).with_suffix('.pt')
            emb_bg = torch.load(p_tmf).numpy()

            # 76.78 아 왜
            # 77.25 -> 77.76
            # embs_segment -= np.einsum('i,j,j->i', emb_anomalous_mean_unit, emb_anomalous_mean_unit, emb_bg)

            # 77.54: 엥 이건 그냥 BG 상관 없이 anom mean project out 하는 거임
            # emb_bg_anom = np.einsum('i,j,j->i', emb_anomalous_mean_unit, emb_anomalous_mean_unit, emb_bg)
            # emb_bg_anom /= np.linalg.norm(emb_bg_anom)
            # embs_segment -= np.einsum('i,j,hj->hi', emb_bg_anom, emb_bg_anom, embs_segment)

            # 66.38
            # emb_bg_unit = emb_bg / np.linalg.norm(emb_bg)
            # embs_segment -= np.einsum('i,j,j->i', emb_bg_unit, emb_bg_unit, emb_proj_delta_units)

            # 77.40: BG의 anom NN들을 project out
            # embs_anom_nn = embs_anomalous[np.argsort(embs_anomalous @ emb_bg)[-1:]]
            # embs_anom_nn_unit = embs_anom_nn / np.linalg.norm(embs_anom_nn)
            # for emb_anom_nn_unit in embs_anom_nn_unit:
            #     embs_segment -= np.einsum('i,j,hj->hi', emb_anom_nn_unit, emb_anom_nn_unit, embs_segment)

            # 72.24
            # emb_anom_nn = embs_anomalous[np.argmax(embs_anomalous @ emb_bg)]
            # emb_norm_nn = embs_normal[np.argmax(embs_normal @ emb_bg)]
            # emb_delta_nn = emb_anom_nn - emb_norm_nn
            # emb_delta_nn_unit = emb_delta_nn / np.linalg.norm(emb_delta_nn)
            # embs_segment -= np.einsum('i,j,hj->hi', emb_delta_nn_unit, emb_delta_nn_unit, embs_segment)

            # 65.60
            # embs_segment -= np.einsum('i,j,j->i', emb_normal_mean_unit, emb_normal_mean_unit, emb_bg)

            # 74.86
            # embs_segment -= np.einsum('i,j,j->i', emb_proj_delta_units, emb_proj_delta_units, emb_bg)

            # 75.78
            # embs_segment -= np.einsum('i,j,j->i', emb_delta_units, emb_delta_units, emb_bg)

            # 74.04
            # embs_segment -= np.einsum('i,j,j->i', emb_proj_delta, emb_proj_delta, emb_bg)

            # 49.88 -> 50.12
            # embs_segment -= np.einsum('i,j,j->i', emb_delta, emb_delta, emb_bg)

            # 77.22
            # 77.54
            # embs_segment -= np.einsum('i,j,j->i', emb_anomalous_mean_unit, emb_anomalous_mean_unit, (emb_bg + emb_segment_mean) / 2)

            # 74.92
            # embs_segment -= np.einsum('i,j,j->i', emb_anomalous_mean_unit, emb_anomalous_mean_unit, emb_bg + emb_segment_mean)

            # 53.80: 통째로 뺴면 context까지 날라가서 효과가 없다!
            # embs_segment -= emb_bg
            # 58.84
            # emb_bg /= np.linalg.norm(emb_bg)
            # embs_segment -= np.einsum('i,j,hj->hi', emb_bg, emb_bg, embs_segment)

            # 76.97: modality gap도 제거할 수 있을까...?
            # proj_bg = np.einsum('i,j,j->i', emb_anomalous_mean_unit, emb_anomalous_mean_unit, emb_bg)
            # embs_segment -= proj_bg
            # emb_bg -= proj_bg
            # embs_segment -= np.einsum('i,j,j->i', emb_text_mean_unit, emb_text_mean_unit, emb_bg)

            # 76.79
            # embs_segment -= np.einsum('hi,hj,j->i', embs_anomalous_sing[:1], embs_anomalous_sing[:1], emb_bg)
            # 78.97
            # embs_segment -= np.einsum('hi,hj,j->i', embs_anomalous_sing[:2], embs_anomalous_sing[:2], emb_bg)
            # 79.36: 이것도 그냥 singular 두 개 project out 한 거임
            # emb_bg_anom = np.einsum('hi,hj,j->i', embs_anomalous_sing[:2], embs_anomalous_sing[:2], emb_bg)
            # emb_bg_anom /= np.linalg.norm(emb_bg_anom)
            # embs_segment -= np.einsum('i,j,hj->hi', emb_bg_anom, emb_bg_anom, embs_segment)
            # 76.49
            # embs_segment -= np.einsum('hi,hj,j->i', embs_anomalous_sing[:3], embs_anomalous_sing[:3], emb_bg)
            # 74.01
            # embs_segment -= np.einsum('hi,hj,j->i', embs_anomalous_sing[:4], embs_anomalous_sing[:4], emb_bg)

            # 68.14
            # embs_segment -= np.einsum('hi,hj,j->i', embs_anomalous_pc[:1], embs_anomalous_pc[:1], emb_bg)
            # 69.46
            # embs_segment -= np.einsum('hi,hj,j->i', embs_anomalous_pc[:2], embs_anomalous_pc[:2], emb_bg)
            # 64.45
            # embs_segment -= np.einsum('hi,hj,j->i', embs_anomalous_pc[:8], embs_anomalous_pc[:8], emb_bg)

            ##################################################################################################

            dot_products = np.dot(embs_segment, embs.T)
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
                    'is_anomalous': is_anomalous[idx].astype(int).tolist(),
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
            json.dump(video_record, p_out_score.open('w'))
        print()

    def evaluate(
        self,
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
                dots = segment_record['dot_products']
                weights = np.exp(dots - np.max(dots))
                weights /= np.sum(weights)
                segment_score = weights @ segment_record['is_anomalous']  # anomality of all captions
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

    def extract_tmf(
        self,
        num_sampled_frames: int = 32,
    ):
        inner = Inner()
        inner.extract_tmf(
            num_sampled_frames=num_sampled_frames
        )

    def extract_embeddings_per_segment(
        self,
        rank: int = 0, world_size: int = 1,
        segment_duration_sec: float = 1.,
        segment_overlap_sec: float = .5,
        num_sampled_segment_frames: int = 16,
        caption_type: str = '00-rich-context',
        retriever_name: str = 'imagebind',
    ):
        inner = Inner(caption_type, retriever_name, rank, world_size)
        inner.extract_embeddings_per_segment(
            rank=rank, world_size=world_size,
            segment_duration_sec=segment_duration_sec,
            segment_overlap_sec=segment_overlap_sec,
            num_sampled_segment_frames=num_sampled_segment_frames,
            retriever_name=retriever_name,
        )

    def extract_tmf_embeddings(
        self,
        rank: int = 0, world_size: int = 1,
        num_tmf_frames: int = 32,
        retriever_name: str = 'imagebind',
    ):
        inner = Inner(retriever_name=retriever_name, rank=rank, world_size=world_size)
        inner.extract_tmf_embeddings(
            rank=rank, world_size=world_size,
            num_tmf_frames=num_tmf_frames,
            retriever_name=retriever_name,
        )

    def extract_caption_embeddings(
        self,
        rank: int = 0, world_size: int = 1,
        caption_type: str = '00-rich-context',
        retriever_name: str = 'imagebind',
    ):
        inner = Inner(caption_type, retriever_name, rank, world_size)
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
        caption_type: str = '00-rich-context',
        retriever_name: str = 'imagebind',
    ):
        inner = Inner(caption_type, retriever_name, rank, world_size)
        inner.match_captions_per_segment(
            rank=rank, world_size=world_size,
            segment_duration_sec=segment_duration_sec,
            segment_overlap_sec=segment_overlap_sec,
            num_sampled_segment_frames=num_sampled_segment_frames,
            num_captions_per_segment=num_captions_per_segment,
            anomalous_scale=anomalous_scale,
            retriever_name=retriever_name,
        )

    def evaluate(
        self,
        caption_type: str = '00-rich-context',
        retriever_name: str = 'imagebind',
        force: bool = False,
    ):
        inner = Inner(caption_type, retriever_name)
        inner.evaluate(
            force=force,
        )


if __name__ == '__main__':
    import fire
    fire.Fire(Main)
