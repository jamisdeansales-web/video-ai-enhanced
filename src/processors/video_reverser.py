"""
Video Reverser - reverse videos frame by frame, with optional recombination.

Ported from https://github.com/mikecastrodemaria/ReverseVideo. That version
saved every frame to disk as a JPEG and rebuilt the clip with moviepy's
ImageSequenceClip, which re-encodes the footage twice and loses quality; it also
imports moviepy.editor, a module that no longer exists in moviepy 2.x. The logic
is reimplemented here on the project's own VideoProcessor/VideoWriter, so it
shares the cv2/FFmpeg path the upscaler already uses.

Frames are spooled to a temporary directory as lossless PNG instead of being
held in memory, so reversing a long or high-resolution video does not depend on
how much RAM is free.
"""

import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2

from utils.cancellation import OperationCancelled, check as check_cancelled
from utils.image_io import imread, imwrite
from utils.video_processor import VideoProcessor, VideoWriter

logger = logging.getLogger(__name__)

# Order in which to concatenate the original (a) and the reversed clip (b).
COMBINE_MODES = ('none', 'ab', 'ba')


def _spool_frames(
    processor: VideoProcessor,
    spool_dir: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[str]:
    """Write every frame to spool_dir as PNG and return the paths in order."""
    total = processor.get_frame_count()
    paths = []

    for index, frame in processor.extract_frames():
        check_cancelled()
        path = os.path.join(spool_dir, 'frame_%06d.png' % index)
        if not imwrite(path, frame):
            raise RuntimeError('Could not write frame %d to %s' % (index, path))
        paths.append(path)

        if progress_callback and total:
            progress_callback(len(paths), total)

    return paths


def reverse_video(
    video_path: str,
    output_path: Optional[str] = None,
    keep_frames: bool = False,
    frames_dir: Optional[str] = None,
    codec: str = 'mp4v',
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict:
    """
    Reverse a video frame by frame.

    Args:
        video_path: Input video. Any container VideoProcessor can decode.
        output_path: Where to write the result. Defaults to a temporary
            "<name>_reversed.mp4".
        keep_frames: Also leave the individual frames on disk, as JPEGs, for
            display in the UI gallery.
        frames_dir: Where to put those JPEGs. Defaults to a temporary directory.
        codec: FourCC for the output video.
        progress_callback: Called as (current_frame, total_frames).

    Returns:
        dict with success, output_path, frame_count, fps, resolution, frames
        (JPEG paths, empty unless keep_frames) and elapsed seconds.
    """
    started = time.time()
    source = Path(video_path)

    if output_path is None:
        output_dir = Path(tempfile.gettempdir()) / 'video_reverser_outputs'
        output_dir.mkdir(exist_ok=True)
        output_path = str(output_dir / ('%s_reversed.mp4' % source.stem))

    spool_dir = tempfile.mkdtemp(prefix='reverse_spool_')
    kept_frames = []

    try:
        with VideoProcessor(video_path) as processor:
            metadata = processor.get_metadata()
            fps = metadata['fps'] or 25.0
            resolution = metadata['resolution']

            logger.info(
                'Reversing %s (%dx%d, %d frames, %.2f fps)',
                source.name, resolution[0], resolution[1],
                metadata['frame_count'], fps,
            )

            # Pass 1: spool forward. Reading forward and replaying the spool
            # backwards avoids per-frame seeking, which cv2 does unreliably on
            # inter-frame codecs such as VP9.
            frame_paths = _spool_frames(processor, spool_dir, progress_callback)

        if not frame_paths:
            raise ValueError('No frames could be read from %s' % video_path)

        if keep_frames:
            if frames_dir is None:
                frames_dir = tempfile.mkdtemp(prefix='reverse_frames_')
            os.makedirs(frames_dir, exist_ok=True)

        # Pass 2: write the spool back out in reverse.
        writer = VideoWriter(
            output_path=output_path,
            fps=fps,
            resolution=resolution,
            codec=codec,
        )
        try:
            for position, path in enumerate(reversed(frame_paths)):
                check_cancelled()
                frame = imread(path)
                if frame is None:
                    raise RuntimeError('Could not read spooled frame %s' % path)
                writer.write_frame(frame)

                if keep_frames:
                    jpeg = os.path.join(frames_dir, 'frame_%06d.jpg' % position)
                    imwrite(jpeg, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                    kept_frames.append(jpeg)
        finally:
            writer.close(copy_audio=False)

    finally:
        shutil.rmtree(spool_dir, ignore_errors=True)

    elapsed = time.time() - started
    logger.info('Reversed %s in %.1fs -> %s', source.name, elapsed, output_path)

    return {
        'success': True,
        'input_path': video_path,
        'output_path': output_path,
        'frame_count': len(frame_paths),
        'fps': fps,
        'resolution': resolution,
        'frames': kept_frames,
        'elapsed': elapsed,
    }


def combine_videos(
    first_path: str,
    second_path: str,
    output_path: str,
    codec: str = 'mp4v',
) -> str:
    """
    Concatenate two videos into one.

    Both clips must share a resolution; the original and its reversed copy do by
    construction. The first clip's frame rate is used for the result.
    """
    with VideoProcessor(first_path) as first:
        first_meta = first.get_metadata()
    with VideoProcessor(second_path) as second:
        second_meta = second.get_metadata()

    if first_meta['resolution'] != second_meta['resolution']:
        raise ValueError(
            'Cannot combine clips of different sizes: %s and %s'
            % (first_meta['resolution'], second_meta['resolution'])
        )

    writer = VideoWriter(
        output_path=output_path,
        fps=first_meta['fps'] or 25.0,
        resolution=first_meta['resolution'],
        codec=codec,
    )
    try:
        for path in (first_path, second_path):
            with VideoProcessor(path) as processor:
                for _, frame in processor.extract_frames():
                    writer.write_frame(frame)
    finally:
        writer.close(copy_audio=False)

    return output_path


def reverse_and_combine(
    video_path: str,
    combine_mode: str = 'none',
    keep_frames: bool = False,
    output_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict:
    """
    Reverse a video and optionally concatenate it with the original.

    Args:
        combine_mode: 'none' to only reverse, 'ab' for original then reversed,
            'ba' for reversed then original.

    Returns:
        The dict from reverse_video, plus combined_path when a clip was joined.
    """
    if combine_mode not in COMBINE_MODES:
        raise ValueError(
            'Unknown combine mode %r; expected one of %s'
            % (combine_mode, ', '.join(COMBINE_MODES))
        )

    source = Path(video_path)
    if output_dir is None:
        output_dir = str(Path(tempfile.gettempdir()) / 'video_reverser_outputs')
    os.makedirs(output_dir, exist_ok=True)

    result = reverse_video(
        video_path,
        output_path=os.path.join(output_dir, '%s_reversed.mp4' % source.stem),
        keep_frames=keep_frames,
        progress_callback=progress_callback,
    )

    if combine_mode == 'none':
        result['combined_path'] = None
        return result

    combined = os.path.join(
        output_dir, '%s_combo_%s.mp4' % (source.stem, combine_mode)
    )
    if combine_mode == 'ab':
        combine_videos(video_path, result['output_path'], combined)
    else:
        combine_videos(result['output_path'], video_path, combined)

    result['combined_path'] = combined
    return result


def reverse_batch(
    video_paths: List[str],
    combine_mode: str = 'none',
    keep_frames: bool = False,
    output_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> List[Dict]:
    """
    Reverse several videos in turn.

    One failure does not abort the batch: the entry for that file carries
    success=False and an error message, so the caller can report per-file
    outcomes.

    Args:
        progress_callback: Called as (index, total, video_name).
    """
    results = []
    total = len(video_paths)

    for index, path in enumerate(video_paths, start=1):
        if progress_callback:
            progress_callback(index, total, os.path.basename(path))

        try:
            results.append(
                reverse_and_combine(
                    path,
                    combine_mode=combine_mode,
                    keep_frames=keep_frames,
                    output_dir=output_dir,
                )
            )
        except OperationCancelled:
            # A stop ends the whole batch; it is not one file failing.
            raise
        except Exception as exc:  # noqa: BLE001 - keep the batch going
            logger.error('Failed to reverse %s: %s', path, exc)
            results.append({
                'success': False,
                'input_path': path,
                'error': str(exc),
                'frames': [],
                'combined_path': None,
            })

    return results
