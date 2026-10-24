"""
RIFE Model Wrapper
Real-Time Intermediate Flow Estimation for frame interpolation.

RIFE has no pip package and is not bundled with basicsr, so the architecture
lives in rife_arch.py and only the weights are downloaded. The weights are
read from a safetensors file rather than a pickle: `torch.load` on a
downloaded .pkl executes whatever the file says, and the published RIFE
releases are pickles inside zips. safetensors is pure data.

If the weights cannot be fetched the wrapper falls back to optical flow, which
is what this module did for every frame before. That fallback is measurably
poor - on a clip with steady motion it scored 22.79 dB against RIFE's 33.86 dB,
and worse than simply cross-fading the two frames (25.95 dB) - so it is a last
resort that says so in the log, not a quiet default.
"""

import json
import logging
import struct
from typing import List, Optional

import cv2
import numpy as np
import torch

from utils.model_downloader import download_model

logger = logging.getLogger(__name__)

try:
    from models.rife_arch import IFNet
    RIFE_AVAILABLE = True
except ImportError:  # pragma: no cover - only if the file is missing
    RIFE_AVAILABLE = False
    logging.warning("RIFE architecture unavailable")

# RIFE downsamples by 32 across its four stages.
SIZE_MULTIPLE = 32


def load_safetensors(path):
    """
    Read a safetensors file into a state dict.

    Implemented here rather than pulling in the safetensors package: the
    format is a length-prefixed JSON header followed by a flat buffer, and
    this is the only place the project needs it.
    """
    dtypes = {'F32': np.float32, 'F16': np.float16, 'BF16': None}

    with open(path, 'rb') as handle:
        header_len = struct.unpack('<Q', handle.read(8))[0]
        header = json.loads(handle.read(header_len))
        body = handle.read()

    state = {}
    for key, meta in header.items():
        if key == '__metadata__':
            continue
        dtype = dtypes.get(meta['dtype'])
        if dtype is None:
            raise ValueError('Unsupported tensor dtype %s' % meta['dtype'])
        begin, end = meta['data_offsets']
        array = np.frombuffer(body[begin:end], dtype=dtype)
        state[key] = torch.from_numpy(array.reshape(meta['shape']).copy())

    return state


class RIFEModel:
    """RIFE frame interpolation."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'cuda',
        fp16: bool = False,
        model_name: str = 'rife_flownet',
    ):
        """
        Args:
            model_path: Weights file. Downloaded when not given.
            device: 'cuda' or 'cpu'.
            fp16: Half precision on CUDA.
            model_name: Key in the model downloader registry.
        """
        self.device = device
        self.fp16 = fp16 and device == 'cuda'
        self.net = None
        self._use_simple_interpolation = True

        if not RIFE_AVAILABLE:
            logger.warning("RIFE architecture missing; using optical flow instead")
            return

        try:
            if model_path is None:
                model_path = download_model(model_name)
            if model_path is None:
                raise RuntimeError('weights could not be downloaded')

            net = IFNet()
            # strict on purpose: a partially loaded flow network still runs and
            # still returns frames, just bad ones.
            net.load_state_dict(load_safetensors(str(model_path)), strict=True)
            net.eval()

            try:
                self.net = net.to(device)
            except (torch.cuda.CudaError, RuntimeError) as exc:
                if device == 'cuda' and 'cudaErrorNoKernelImageForDevice' in str(exc):
                    logger.warning(
                        "CUDA kernel not available for this GPU, falling back to CPU"
                    )
                    self.device = 'cpu'
                    self.fp16 = False
                    self.net = net.to('cpu')
                else:
                    raise

            self._use_simple_interpolation = False
            logger.info("RIFE loaded successfully on %s", self.device)

        except Exception as exc:  # noqa: BLE001 - interpolation must still work
            logger.warning(
                "Could not load RIFE (%s); falling back to optical flow, which "
                "is markedly worse", exc
            )
            self._use_simple_interpolation = True

    @property
    def using_rife(self) -> bool:
        """Whether the real network is in use rather than the fallback."""
        return not self._use_simple_interpolation

    def _to_tensor(self, frame: np.ndarray):
        rgb = frame[:, :, [2, 1, 0]].astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self.device)

    def _to_frame(self, tensor, height: int, width: int) -> np.ndarray:
        array = tensor[0, :, :height, :width].clamp(0, 1).float().cpu().numpy()
        rgb = np.transpose(array[[2, 1, 0]], (1, 2, 0))
        return (rgb * 255.0).round().astype(np.uint8)

    def interpolate_frames(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        num_intermediates: int = 1,
        timestep: Optional[float] = None,
    ) -> List[np.ndarray]:
        """
        Generate frames between two consecutive frames.

        Args:
            frame1, frame2: BGR frames of the same size.
            num_intermediates: How many frames to put between them.
            timestep: A single position in 0-1 instead of evenly spaced ones.

        Returns:
            The intermediate frames, in order.
        """
        if self._use_simple_interpolation:
            return self._simple_interpolation(
                frame1, frame2, num_intermediates, timestep
            )

        if timestep is not None:
            steps = [float(timestep)]
        else:
            steps = [(i + 1) / (num_intermediates + 1) for i in range(num_intermediates)]

        height, width = frame1.shape[:2]
        pad_h = (SIZE_MULTIPLE - height % SIZE_MULTIPLE) % SIZE_MULTIPLE
        pad_w = (SIZE_MULTIPLE - width % SIZE_MULTIPLE) % SIZE_MULTIPLE

        try:
            tensor0 = self._to_tensor(frame1)
            tensor1 = self._to_tensor(frame2)
            if pad_h or pad_w:
                tensor0 = torch.nn.functional.pad(
                    tensor0, (0, pad_w, 0, pad_h), mode='replicate')
                tensor1 = torch.nn.functional.pad(
                    tensor1, (0, pad_w, 0, pad_h), mode='replicate')

            results = []
            with torch.no_grad():
                for step in steps:
                    output = self.net(tensor0, tensor1, timestep=step)
                    if not torch.isfinite(output).all():
                        raise RuntimeError('model returned non-finite values')
                    results.append(self._to_frame(output, height, width))
            return results

        except Exception as exc:  # noqa: BLE001 - never kill a whole video
            logger.error(
                "RIFE failed on a frame pair (%s); using optical flow for it", exc
            )
            return self._simple_interpolation(
                frame1, frame2, num_intermediates, timestep
            )

    def _simple_interpolation(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        num_intermediates: int = 1,
        timestep: Optional[float] = None,
    ) -> List[np.ndarray]:
        """
        Optical flow fallback.

        Warps the first frame part-way along the Farneback flow. Kept so
        interpolation still works without the weights, but it is poor: on a
        clip with steady motion it scored below a plain cross-fade.
        """
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )

        height, width = frame1.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))

        if timestep is not None:
            steps = [float(timestep)]
        else:
            steps = [(i + 1) / (num_intermediates + 1) for i in range(num_intermediates)]

        results = []
        for step in steps:
            map_x = (grid_x + flow[..., 0] * step).astype(np.float32)
            map_y = (grid_y + flow[..., 1] * step).astype(np.float32)
            warped = cv2.remap(
                frame1, map_x, map_y, cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            # Blend towards the second frame so the sequence does not jump at
            # the far end, where a one-sided warp drifts furthest.
            results.append(cv2.addWeighted(warped, 1 - step, frame2, step, 0))

        return results

    def clear_cache(self):
        """Clear the CUDA cache."""
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class SimpleFrameInterpolator:
    """Plain cross-fade, used when no motion model is wanted."""

    def __init__(self, device: str = 'cpu'):
        self.device = device

    def interpolate_frames(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        num_intermediates: int = 1,
        timestep: Optional[float] = None,
    ) -> List[np.ndarray]:
        if timestep is not None:
            steps = [float(timestep)]
        else:
            steps = [(i + 1) / (num_intermediates + 1) for i in range(num_intermediates)]
        return [
            cv2.addWeighted(frame1, 1 - step, frame2, step, 0) for step in steps
        ]

    def clear_cache(self):
        """Nothing to clear."""


def create_rife_model(
    device: str = 'cuda',
    fp16: bool = False,
    model_path: Optional[str] = None,
) -> RIFEModel:
    """Factory for the RIFE model."""
    return RIFEModel(model_path=model_path, device=device, fp16=fp16)
