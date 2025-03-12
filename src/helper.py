from pathlib import Path

import numpy as np

import decord
from decord import VideoReader, cpu
decord.bridge.reset_bridge()


def compute_num_segments(
    p_video,
    segment_duration_sec: float = 1.,
    segment_overlap_sec: float = 0.,
):
    FPS = 30
    num_segment_frames = int(segment_duration_sec * FPS)
    num_overlap_frames = int(segment_overlap_sec * FPS)

    vr = VideoReader(str(p_video), ctx=cpu(0))
    num_total_frames = len(vr)
    num_segments = (num_total_frames - num_segment_frames) // (num_segment_frames - num_overlap_frames) + 1  # Discard last segment if not enough frames
    return num_segments


def get_segments(
    p_video,
    segment_duration_sec: float = 1.,
    num_segment_frames: int = 32,
    segment_overlap_sec: float = 0.,
    num_skip_first_segments: int = 0,
):
    FPS = 30
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
