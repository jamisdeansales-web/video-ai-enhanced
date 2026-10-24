"""
Temporal Upscaler - whole-video upscaling with a sequence model.

SpatialUpscaler walks a video one frame at a time, which is all Real-ESRGAN and
SwinIR need. BasicVSR++ instead reads a run of consecutive frames and
propagates information along it, so it needs its own driver: this one feeds the
video through in overlapping windows.

The overlap matters. A window only has context from the frames inside it, so
its first and last frames are the weakest - exactly the seam where flicker
would reappear. Each window therefore overlaps its neighbours and only the
middle of the result is kept, except at the two ends of the video where there
is nothing beyond.
"""

import logging
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2

from utils.cancellation import check as check_cancelled
from utils.system_manager import get_system_manager
from utils.video_processor import (
    DEFAULT_CODEC,
    DEFAULT_CRF,
    VideoProcessor,
    VideoWriter,
)

logger = logging.getLogger(__name__)


class TemporalUpscaler:
    """Upscale a video with a sequence model, in overlapping windows."""

    def __init__(
        self,
        model_name: str = 'basicvsrpp',
        scale_factor: float = 4.0,
        device: str = 'auto',
        window: int = 0,
        overlap: int = 0,
        codec: str = DEFAULT_CODEC,
        crf: int = DEFAULT_CRF,
    ):
        """
        Args:
            model_name: Currently only 'basicvsrpp'.
            scale_factor: Target scale. The model itself is 4x; any other
                value is reached by resizing its output.
            device: 'cuda', 'cpu' or 'auto'.
            window: Frames per window. 0 picks one from the available VRAM.
            overlap: Frames shared between consecutive windows; the ones at
                each inner edge are dropped. 0 derives it from the window.
        """
        if model_name.lower() != 'basicvsrpp':
            raise NotImplementedError(
                "Unknown temporal model %r; only basicvsrpp is available"
                % model_name
            )

        if scale_factor <= 0 or scale_factor > 16.0:
            raise ValueError(
                "Invalid scale factor: %s. Must be between 0.1 and 16.0."
                % scale_factor
            )

        self.model_name = model_name
        self.scale_factor = float(scale_factor)
        self.requested_overlap = max(0, int(overlap))
        self.codec = codec
        self.crf = crf
        self.requested_window = int(window)
        self.sys_manager = get_system_manager()

        if device == 'auto':
            device = self.sys_manager.get_device()
        self.device = device

        from models.basicvsrpp_model import create_basicvsrpp_model

        self.model = create_basicvsrpp_model(device=self.device)
        self.model_scale = self.model.scale

    def _choose_window(self, resolution) -> int:
        """Frames per window, from the request or from the free VRAM."""
        if self.requested_window:
            return max(2, self.requested_window)

        vram_gb = 8.0
        try:
            import torch

            if self.device == 'cuda' and torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:  # noqa: BLE001 - keep the conservative default
            pass

        return self.model.suggest_window(resolution, vram_gb)

    def upscale_video(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> Dict:
        """
        Upscale a whole video.

        Args:
            input_path: Source video.
            output_path: Destination.
            progress_callback: Called as (frames_done, frames_total, eta).

        Returns:
            dict with success, output_path, metrics.
        """
        started = time.time()

        with VideoProcessor(input_path) as processor:
            metadata = processor.get_metadata()
            resolution = metadata['resolution']
            total = metadata['frame_count']
            fps = metadata['fps'] or 25.0

            window = self._choose_window(resolution)

            # The overlap needs a floor, not just a ratio. BasicVSR++
            # propagates along the whole sequence, so a frame near a window's
            # edge has little context and comes out different from the same
            # frame processed mid-window - a seam every `step` frames.
            #
            # Measured on a 45-frame clip at window=16, comparing the
            # motion-compensated residual at the seams against everywhere else:
            #
            #   overlap=2  ->  10.36x   (a plain, visible jump)
            #   overlap=4  ->   0.39x   (gone)
            #   overlap=8  ->   0.94x   (gone, best overall residual)
            #
            # Four frames are enough for the propagation to warm up, so that is
            # the floor; a quarter of the window takes over once windows are
            # large.
            overlap = self.requested_overlap or max(4, window // 4)
            # The overlap cannot swallow the window, or no frame would ever be
            # committed and the loop would not advance.
            overlap = min(overlap, max(0, window - 1))
            step = window - overlap

            out_w = int(resolution[0] * self.scale_factor)
            out_h = int(resolution[1] * self.scale_factor)

            logger.info(
                "BasicVSR++ on %s: %d frames, window=%d overlap=%d, "
                "%dx%d -> %dx%d",
                Path(input_path).name, total, window, overlap,
                resolution[0], resolution[1], out_w, out_h,
            )

            writer = VideoWriter(
                output_path=output_path,
                fps=fps,
                resolution=(out_w, out_h),
                audio_source=input_path if processor.has_audio() else None,
                codec=self.codec,
                crf=self.crf,
            )

            buffer: List = []
            buffer_start = 0
            written = 0

            def flush(is_last: bool):
                """Upscale the buffered window and write the frames it owns."""
                nonlocal written

                if not buffer:
                    return

                upscaled = self.model.upscale_sequence(buffer)

                # Drop the half-overlap at each inner edge; keep everything at
                # the true start and end of the video.
                head = 0 if buffer_start == 0 else overlap // 2
                tail = len(upscaled) if is_last else len(upscaled) - (overlap - overlap // 2)

                for frame in upscaled[head:tail]:
                    if (frame.shape[1], frame.shape[0]) != (out_w, out_h):
                        frame = cv2.resize(
                            frame, (out_w, out_h),
                            interpolation=cv2.INTER_LANCZOS4
                                if self.scale_factor < self.model_scale
                                else cv2.INTER_CUBIC,
                        )
                    writer.write_frame(frame)
                    written += 1

                if progress_callback:
                    elapsed = time.time() - started
                    rate = written / elapsed if elapsed else 0
                    eta = (total - written) / rate if rate else 0
                    progress_callback(written, total, eta)

            try:
                for index, frame in processor.extract_frames():
                    check_cancelled()
                    buffer.append(frame)

                    if len(buffer) == window:
                        is_last = index >= total - 1
                        flush(is_last)
                        if is_last:
                            buffer = []
                            break
                        # Carry the overlap into the next window.
                        buffer = buffer[step:] if overlap else []
                        buffer_start += step
                        self.model.clear_cache()

                if buffer:
                    flush(True)
            finally:
                writer.close()

        elapsed = time.time() - started
        fallbacks = self.model.fallback_count

        if fallbacks:
            logger.warning(
                "%d of %d frames fell back to Lanczos: they are upscaled but "
                "not temporally coherent",
                fallbacks, written,
            )

        logger.info("BasicVSR++ finished in %.1fs (%d frames)", elapsed, written)

        return {
            'success': True,
            'output_path': output_path,
            'metrics': {
                'frames_processed': written,
                'processing_time': elapsed,
                'time_per_frame': elapsed / written if written else 0,
                'input_resolution': resolution,
                'output_resolution': (out_w, out_h),
                'scale_factor': self.scale_factor,
                'window': window,
                'overlap': overlap,
                'fallback_frames': fallbacks,
            },
        }

    def upscale_preview(
        self,
        input_path: str,
        output_path: str,
        duration: float = 5.0,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> Dict:
        """
        Upscale the first few seconds only.

        Same signature as SpatialUpscaler.upscale_preview so the Preview button
        works with either kind of model. The clip is cut first and then run
        through the normal path, because the model needs a real run of frames:
        upscaling a few isolated ones would not show the coherence that is the
        point of this model.
        """
        segment = tempfile.mktemp(suffix='_bvsrpp_segment.mp4')

        with VideoProcessor(input_path) as processor:
            if not processor.extract_segment(0.0, duration, segment):
                return {
                    'success': False,
                    'error': 'Could not extract the preview segment',
                }

        try:
            return self.upscale_video(segment, output_path, progress_callback)
        finally:
            try:
                Path(segment).unlink(missing_ok=True)
            except OSError:
                pass

    def estimate_processing_time(self, input_path: str) -> Dict:
        """Rough estimate, from a measured rate on an RTX 5090."""
        with VideoProcessor(input_path) as processor:
            metadata = processor.get_metadata()

        megapixels = (metadata['width'] * metadata['height']) / (1024 * 1024)
        per_frame = 0.25 * max(megapixels, 0.05) * (4.0 if self.device == 'cpu' else 1.0)
        total_seconds = per_frame * metadata['frame_count']

        return {
            'estimated_total_time': total_seconds,
            'estimated_total_time_formatted': '%dm %ds' % (
                total_seconds // 60, total_seconds % 60
            ),
            'time_per_frame': per_frame,
        }


def create_temporal_upscaler(
    model_name: str = 'basicvsrpp',
    scale_factor: float = 4.0,
    device: str = 'auto',
    window: int = 0,
    overlap: int = 0,
    codec: str = DEFAULT_CODEC,
    crf: int = DEFAULT_CRF,
) -> TemporalUpscaler:
    """Factory for the temporal upscaler."""
    return TemporalUpscaler(
        model_name=model_name,
        scale_factor=scale_factor,
        device=device,
        window=window,
        overlap=overlap,
        codec=codec,
        crf=crf,
    )
