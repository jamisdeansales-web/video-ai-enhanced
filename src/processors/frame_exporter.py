"""
Frame Exporter - write a video's frames out as still images.

Covers the whole video or a time range, optionally upscaling each frame through
the same Real-ESRGAN path the video upscaler uses, and optionally bundling the
result into a zip so the UI can offer a single download.

Frames are named after their index in the source video, so an exported still can
always be traced back to the frame it came from even when a range or a step was
used.
"""

import logging
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2

from utils.video_processor import VideoProcessor

logger = logging.getLogger(__name__)

# How the exported span is chosen.
#   all   - every frame in the video
#   from  - from start_time to the end
#   to    - from the beginning up to end_time
#   range - between start_time and end_time
RANGE_MODES = ('all', 'from', 'to', 'range')

IMAGE_FORMATS = ('png', 'jpg')


def resolve_range(
    metadata: Dict,
    range_mode: str = 'all',
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> Tuple[int, int]:
    """
    Turn a time range into a half-open frame range [start_frame, end_frame).

    Args:
        metadata: A VideoProcessor metadata dict.
        range_mode: One of RANGE_MODES.
        start_time: Seconds from the start. Used by 'from' and 'range'.
        end_time: Seconds from the start. Used by 'to' and 'range'.

    Raises:
        ValueError: If the mode is unknown, a needed bound is missing, or the
            range is empty or outside the video.
    """
    if range_mode not in RANGE_MODES:
        raise ValueError(
            'Unknown range mode %r; expected one of %s'
            % (range_mode, ', '.join(RANGE_MODES))
        )

    total = int(metadata['frame_count'])
    fps = metadata['fps'] or 25.0
    duration = metadata['duration']

    def to_frame(seconds: float) -> int:
        return int(round(seconds * fps))

    if range_mode == 'all':
        return 0, total

    if range_mode in ('from', 'range') and start_time is None:
        raise ValueError('Range mode %r needs a start time' % range_mode)
    if range_mode in ('to', 'range') and end_time is None:
        raise ValueError('Range mode %r needs an end time' % range_mode)

    start_frame = 0 if range_mode == 'to' else to_frame(float(start_time))
    end_frame = total if range_mode == 'from' else to_frame(float(end_time))

    if start_frame < 0:
        raise ValueError('Start time cannot be negative')
    if start_frame >= total:
        raise ValueError(
            'Start time %.2fs is past the end of the video (%.2fs)'
            % (float(start_time), duration)
        )

    # A requested end beyond the video is clamped rather than refused: asking
    # for "up to 60s" of a 42s clip plainly means "to the end".
    end_frame = min(end_frame, total)

    if end_frame <= start_frame:
        raise ValueError(
            'Empty range: end (%.2fs) must be after start (%.2fs)'
            % (end_frame / fps, start_frame / fps)
        )

    return start_frame, end_frame


def export_frames(
    video_path: str,
    output_dir: Optional[str] = None,
    range_mode: str = 'all',
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    every_n: int = 1,
    image_format: str = 'png',
    jpeg_quality: int = 95,
    upscale: bool = False,
    scale_factor: float = 2.0,
    model_name: str = 'realesrgan',
    device: str = 'auto',
    make_zip: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict:
    """
    Export a video's frames as still images.

    Args:
        video_path: Input video, any container VideoProcessor can decode.
        output_dir: Where to write the images. Defaults to a temporary folder.
        range_mode: One of RANGE_MODES; see resolve_range.
        start_time: Range start, in seconds.
        end_time: Range end, in seconds.
        every_n: Export one frame out of every N (1 = all frames in the range).
        image_format: 'png' (lossless) or 'jpg'.
        jpeg_quality: Quality for 'jpg', 1-100.
        upscale: Run each frame through the upscaler before writing it.
        scale_factor: Scale for that upscaling. Below 1.0 downscales.
        model_name: Upscaling model.
        device: 'cuda', 'cpu' or 'auto'.
        make_zip: Also bundle the images into a zip beside output_dir.
        progress_callback: Called as (frames_done, frames_total).

    Returns:
        dict with success, frames, frame_count, output_dir, zip_path, fps,
        range (start_frame, end_frame), resolution, upscaled and elapsed.
    """
    started = time.time()

    if image_format not in IMAGE_FORMATS:
        raise ValueError(
            'Unknown image format %r; expected one of %s'
            % (image_format, ', '.join(IMAGE_FORMATS))
        )
    if every_n < 1:
        raise ValueError('every_n must be 1 or more, got %r' % every_n)

    source = Path(video_path)

    if output_dir is None:
        output_dir = str(
            Path(tempfile.gettempdir()) / 'video_frame_exports' / source.stem
        )
    os.makedirs(output_dir, exist_ok=True)

    upscaler = None
    written: List[str] = []

    with VideoProcessor(video_path) as processor:
        metadata = processor.get_metadata()
        start_frame, end_frame = resolve_range(
            metadata, range_mode, start_time, end_time
        )

        planned = len(range(start_frame, end_frame, every_n))
        logger.info(
            'Exporting %d frames from %s (frames %d-%d, every %d)',
            planned, source.name, start_frame, end_frame, every_n,
        )

        if upscale:
            # Built here rather than by the caller so the model is only loaded
            # when frames are actually going to be upscaled.
            from processors.spatial_upscaler import create_upscaler

            upscaler = create_upscaler(
                model_name=model_name,
                scale_factor=scale_factor,
                device=device,
            )

        if image_format == 'jpg':
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
        else:
            encode_params = []

        for index, frame in processor.extract_frames(
            start_frame=start_frame, end_frame=end_frame, step=every_n
        ):
            if upscaler is not None:
                # _upscale_frame is the only entry point for a single frame;
                # upscale_video works on whole files.
                frame = upscaler._upscale_frame(frame)

            path = os.path.join(
                output_dir, 'frame_%06d.%s' % (index, image_format)
            )
            if not cv2.imwrite(path, frame, encode_params):
                raise RuntimeError('Could not write %s' % path)
            written.append(path)

            if progress_callback:
                progress_callback(len(written), planned)

    if not written:
        raise ValueError('No frames were exported from %s' % video_path)

    resolution = tuple(cv2.imread(written[0]).shape[1::-1])

    zip_path = None
    if make_zip:
        zip_path = str(Path(output_dir).with_suffix('.zip'))
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in written:
                archive.write(path, arcname=os.path.basename(path))

    elapsed = time.time() - started
    logger.info(
        'Exported %d frames from %s in %.1fs -> %s',
        len(written), source.name, elapsed, output_dir,
    )

    return {
        'success': True,
        'input_path': video_path,
        'frames': written,
        'frame_count': len(written),
        'output_dir': output_dir,
        'zip_path': zip_path,
        'fps': metadata['fps'],
        'range': (start_frame, end_frame),
        'source_resolution': metadata['resolution'],
        'resolution': resolution,
        'upscaled': bool(upscale),
        'scale_factor': scale_factor if upscale else 1.0,
        'elapsed': elapsed,
    }


def clear_export_dir(output_dir: str) -> None:
    """Remove a previous export so a re-run does not mix old and new frames."""
    shutil.rmtree(output_dir, ignore_errors=True)
