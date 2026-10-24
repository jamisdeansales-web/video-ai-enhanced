"""
Quality Metrics - measure what an upscale actually did.

Three numbers, each answering a different question:

- **Fidelity (PSNR/SSIM)**: scale the result back down to the source size and
  compare it with the source. An upscaler that invented detail inconsistent
  with the original scores lower. This is a round-trip check, not a comparison
  against a ground truth, because for real footage there is no ground truth at
  the higher resolution.

- **Gain over a plain resize**: how far the result is from simply enlarging the
  source with Lanczos. Near zero means the model did nothing a resize would not
  have done.

- **Temporal stability**: flicker between consecutive frames, with camera and
  subject motion compensated by optical flow so a sharper result is not
  mistaken for a less stable one. Lower is steadier.

Metrics are sampled over a limited number of frames; comparing every frame of a
long video costs more than the upscale itself.
"""

import logging
from typing import Dict, List, Optional

import cv2
import numpy as np

from utils.video_processor import VideoProcessor

logger = logging.getLogger(__name__)

# Frames sampled per video unless the caller says otherwise.
DEFAULT_SAMPLE = 24

# Border ignored when comparing warped frames: optical flow is unreliable where
# content enters or leaves the picture.
WARP_MARGIN = 16

# Below this residual the source is effectively static, and a percentage
# change against it is noise amplified into a headline figure.
STABILITY_FLOOR = 0.05


def _sample_indices(total: int, count: int) -> List[int]:
    """Evenly spaced frame indices, at most `count` of them."""
    if total <= 0:
        return []
    if total <= count:
        return list(range(total))
    return list(np.linspace(0, total - 1, count, dtype=int))


def _read_frames(path: str, indices: List[int]) -> List[np.ndarray]:
    """Read specific frames from a video, in order."""
    wanted = set(indices)
    frames = []
    with VideoProcessor(path) as processor:
        for index, frame in processor.extract_frames():
            if index in wanted:
                frames.append(frame)
            if len(frames) == len(wanted):
                break
    return frames


def temporal_stability(path: str, sample: int = DEFAULT_SAMPLE) -> Optional[float]:
    """
    Mean residual between consecutive frames after motion compensation.

    Optical flow warps each frame onto the next, so genuine motion cancels out
    and what is left is frame-to-frame instability: the flicker a per-frame
    upscaler introduces. Lower is steadier. None if the clip is too short.
    """
    with VideoProcessor(path) as processor:
        total = processor.get_frame_count()

    if total < 2:
        return None

    # Consecutive pairs, so sample runs rather than isolated frames.
    pairs = min(sample, total - 1)
    starts = _sample_indices(total - 1, pairs)

    residuals = []
    with VideoProcessor(path) as processor:
        for start in starts:
            first = processor.read_frame(start)
            second = processor.read_frame(start + 1)
            if first is None or second is None:
                continue

            gray_a = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
            gray_b = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                gray_a, gray_b, None, 0.5, 3, 21, 3, 7, 1.5, 0
            )

            height, width = gray_b.shape
            grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
            warped = cv2.remap(
                first,
                (grid_x + flow[..., 0]).astype(np.float32),
                (grid_y + flow[..., 1]).astype(np.float32),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )

            margin = WARP_MARGIN
            if height <= 2 * margin or width <= 2 * margin:
                margin = 0
            diff = np.abs(
                warped.astype(np.float32) - second.astype(np.float32)
            )
            if margin:
                diff = diff[margin:-margin, margin:-margin]
            residuals.append(float(diff.mean()))

    return float(np.mean(residuals)) if residuals else None


def compare_videos(
    original_path: str,
    upscaled_path: str,
    sample: int = DEFAULT_SAMPLE,
) -> Dict:
    """
    Compare an upscaled video with its source.

    Returns:
        dict with psnr, ssim, psnr_vs_resize, stability_original,
        stability_upscaled and frames_compared. Values that could not be
        computed are None rather than absent.
    """
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    result = {
        'psnr': None,
        'ssim': None,
        'psnr_vs_resize': None,
        'stability_original': None,
        'stability_upscaled': None,
        'frames_compared': 0,
    }

    try:
        with VideoProcessor(original_path) as processor:
            source_meta = processor.get_metadata()
        with VideoProcessor(upscaled_path) as processor:
            upscaled_meta = processor.get_metadata()
    except (OSError, ValueError) as exc:
        logger.warning("Could not open videos for comparison: %s", exc)
        return result

    total = min(source_meta['frame_count'], upscaled_meta['frame_count'])
    indices = _sample_indices(total, sample)
    if not indices:
        return result

    source_frames = _read_frames(original_path, indices)
    upscaled_frames = _read_frames(upscaled_path, indices)

    width, height = source_meta['resolution']
    psnrs, ssims, psnrs_vs_resize = [], [], []

    for source, upscaled in zip(source_frames, upscaled_frames):
        # Bring the result back to the source size so the two are comparable.
        shrunk = cv2.resize(
            upscaled, (width, height), interpolation=cv2.INTER_AREA
        )

        try:
            psnrs.append(
                peak_signal_noise_ratio(source, shrunk, data_range=255)
            )
            ssims.append(
                structural_similarity(source, shrunk, channel_axis=2, data_range=255)
            )
        except ValueError as exc:
            logger.warning("Could not compare a frame: %s", exc)
            continue

        # How much the model changed things relative to a plain enlargement.
        naive = cv2.resize(
            source,
            (upscaled.shape[1], upscaled.shape[0]),
            interpolation=cv2.INTER_LANCZOS4,
        )
        psnrs_vs_resize.append(
            peak_signal_noise_ratio(naive, upscaled, data_range=255)
        )

    if psnrs:
        result['psnr'] = float(np.mean(psnrs))
        result['ssim'] = float(np.mean(ssims))
        result['frames_compared'] = len(psnrs)
    if psnrs_vs_resize:
        result['psnr_vs_resize'] = float(np.mean(psnrs_vs_resize))

    try:
        result['stability_original'] = temporal_stability(original_path, sample)
        result['stability_upscaled'] = temporal_stability(upscaled_path, sample)
    except Exception as exc:  # noqa: BLE001 - metrics must never break a run
        logger.warning("Could not measure temporal stability: %s", exc)

    return result


def format_metric_rows(metrics: Dict) -> List[List[str]]:
    """Render compare_videos output as rows for the Comparison table."""
    rows = []

    if metrics.get('psnr') is not None:
        rows.append(['PSNR (round-trip)', '%.2f dB' % metrics['psnr']])
        rows.append(['SSIM (round-trip)', '%.4f' % metrics['ssim']])
    if metrics.get('psnr_vs_resize') is not None:
        rows.append(
            ['PSNR vs plain resize', '%.2f dB' % metrics['psnr_vs_resize']]
        )

    before = metrics.get('stability_original')
    after = metrics.get('stability_upscaled')
    if before is not None and after is not None:
        rows.append(['Temporal stability (source)', '%.3f' % before])
        rows.append(['Temporal stability (result)', '%.3f' % after])
        # A relative change is only meaningful when the source actually moves.
        # On near-static footage the baseline is ~0.003 and any result turns
        # into a four-digit percentage that says nothing, so report the
        # absolute difference instead below that threshold.
        if before >= STABILITY_FLOOR:
            change = (after - before) / before * 100.0
            rows.append(['Stability change', '%+.1f%% (lower is steadier)' % change])
        else:
            rows.append([
                'Stability change',
                '%+.3f (source too static for a ratio)' % (after - before),
            ])

    if metrics.get('frames_compared'):
        rows.append(['Frames compared', str(metrics['frames_compared'])])

    return rows
