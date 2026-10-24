"""
BasicVSR++ Model Wrapper
Video super-resolution with temporal coherence.

Unlike Real-ESRGAN and SwinIR, which see one frame at a time, BasicVSR++
propagates information along the whole sequence in both directions and aligns
neighbouring frames with optical flow. That is what removes the flicker a
per-frame upscaler leaves behind, and it is why this wrapper takes a list of
frames rather than a single image.

The architecture comes from the installed basicsr; only the weights are
downloaded. It is a 4x model: there is no 2x checkpoint, so other factors are
reached by resizing the 4x output.
"""

import logging
import re
from typing import List, Optional

import cv2
import numpy as np
import torch

try:
    from basicsr.archs.basicvsrpp_arch import BasicVSRPlusPlus
    BASICVSRPP_AVAILABLE = True
except ImportError:
    BASICVSRPP_AVAILABLE = False
    logging.warning(
        "BasicVSR++ not available. Please install: pip install basicsr "
        "(and run scripts/patch_basicsr.py)"
    )

from utils.model_downloader import download_model

logger = logging.getLogger(__name__)


# The architecture refuses anything smaller; frames below this are padded.
MIN_INPUT_SIZE = 64

# Fixed by the checkpoint.
MODEL_SCALE = 4


def remap_state_dict(state_dict: dict) -> dict:
    """
    Translate the published checkpoint's key names to basicsr's.

    The released weights come from mmediting, whose module names differ from
    basicsr's for the same tensors (the shapes match exactly):

      upsampleN.upsample_conv.*               -> upconvN.*
      spynet...basic_module.M.conv.*          -> spynet...basic_module.2M.*

    The second one is an indexing difference: mmediting wraps each conv in a
    block and numbers them consecutively, while basicsr puts conv and ReLU in
    one Sequential, so only the even positions are convolutions.

    Getting this wrong is quiet rather than loud: loading with strict=False
    leaves SPyNet randomly initialised, the forward pass still runs and still
    returns plausible-looking frames, but the optical flow is noise and the
    temporal coherence this model exists for is gone.
    """
    remapped = {}

    for key, value in state_dict.items():
        match = re.match(r'^upsample([12])\.upsample_conv\.(weight|bias)$', key)
        if match:
            remapped['upconv%s.%s' % (match.group(1), match.group(2))] = value
            continue

        match = re.match(
            r'^spynet\.basic_module\.(\d+)\.basic_module\.(\d+)\.conv\.(weight|bias)$',
            key,
        )
        if match:
            remapped[
                'spynet.basic_module.%s.basic_module.%d.%s'
                % (match.group(1), 2 * int(match.group(2)), match.group(3))
            ] = value
            continue

        remapped[key] = value

    return remapped


class BasicVSRPlusPlusModel:
    """BasicVSR++ model wrapper"""

    def __init__(
        self,
        device: str = 'cuda',
        model_name: str = 'basicvsrpp_reds',
        mid_channels: int = 64,
        num_blocks: int = 7,
    ):
        """
        Initialize BasicVSR++.

        Args:
            device: 'cuda' or 'cpu'.
            model_name: Key in the model downloader registry.
            mid_channels: Must match the checkpoint (c64 in its name).
            num_blocks: Must match the checkpoint (n7 in its name).
        """
        if not BASICVSRPP_AVAILABLE:
            raise ImportError(
                "BasicVSR++ is not available. basicsr must be installed and "
                "patched; run scripts/patch_basicsr.py."
            )

        self.device = device
        self.model_name = model_name
        self.scale = MODEL_SCALE
        self.fallback_count = 0

        logger.info("Loading BasicVSR++ model: %s", model_name)
        model_path = download_model(model_name)
        if model_path is None:
            raise RuntimeError("Failed to download model: %s" % model_name)

        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        state = checkpoint.get('state_dict', checkpoint)
        # mmediting stores the network under "generator." and adds a training
        # counter that is not part of the model.
        state = {
            key[len('generator.'):]: value
            for key, value in state.items()
            if key.startswith('generator.')
        }

        self.model = BasicVSRPlusPlus(
            mid_channels=mid_channels, num_blocks=num_blocks
        )
        # strict=True on purpose: a silently half-loaded network is the failure
        # mode this model is most likely to hit, and the hardest to notice.
        self.model.load_state_dict(remap_state_dict(state), strict=True)
        self.model.eval()

        try:
            self.model = self.model.to(device)
        except (torch.cuda.CudaError, RuntimeError) as exc:
            if device == 'cuda' and 'cudaErrorNoKernelImageForDevice' in str(exc):
                logger.warning(
                    "CUDA kernel not available for this GPU, falling back to CPU"
                )
                self.device = 'cpu'
                self.model = self.model.to('cpu')
            else:
                raise

        logger.info("BasicVSR++ loaded successfully on %s", self.device)

    def upscale_sequence(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Upscale a run of consecutive frames by 4x.

        Args:
            frames: Consecutive frames, BGR uint8, all the same size. Order
                matters: the model propagates along it in both directions.

        Returns:
            One upscaled BGR uint8 frame per input frame.
        """
        if not frames:
            return []

        source_h, source_w = frames[0].shape[:2]

        try:
            batch = np.stack([
                frame[:, :, [2, 1, 0]].astype(np.float32) / 255.0
                for frame in frames
            ])
            tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).unsqueeze(0)
            tensor = tensor.to(self.device)

            # The architecture asserts a 64px minimum on both sides.
            pad_h = max(0, MIN_INPUT_SIZE - source_h)
            pad_w = max(0, MIN_INPUT_SIZE - source_w)
            if pad_h or pad_w:
                n, t, c, h, w = tensor.shape
                tensor = torch.nn.functional.pad(
                    tensor.view(n * t, c, h, w),
                    (0, pad_w, 0, pad_h),
                    mode='replicate',
                ).view(n, t, c, h + pad_h, w + pad_w)

            with torch.no_grad():
                output = self.model(tensor)

            if not torch.isfinite(output).all():
                raise RuntimeError("model returned non-finite values (NaN or inf)")

            output = output[
                :, :, :, : source_h * self.scale, : source_w * self.scale
            ]
            output = output.squeeze(0).float().clamp_(0, 1).cpu().numpy()

            results = []
            for frame in output:
                rgb = np.transpose(frame[[2, 1, 0], :, :], (1, 2, 0))
                results.append((rgb * 255.0).round().astype(np.uint8))
            return results

        except Exception as exc:  # noqa: BLE001 - never kill a whole video
            # Resizing keeps the job alive but loses exactly the temporal
            # coherence this model was chosen for, so make it countable.
            self.fallback_count += len(frames)
            if self.fallback_count <= len(frames):
                logger.error(
                    "BasicVSR++ failed on a sequence and fell back to Lanczos: %s",
                    exc,
                )
            return [
                cv2.resize(
                    frame,
                    (source_w * self.scale, source_h * self.scale),
                    interpolation=cv2.INTER_LANCZOS4,
                )
                for frame in frames
            ]

    def estimate_vram_usage(self, input_resolution: tuple, sequence_length: int = 1) -> float:
        """
        Estimate VRAM usage in GB for one window.

        Cost grows with the window length as well as the frame size, because
        the whole window and its propagated features stay resident.
        """
        width, height = input_resolution
        megapixels = (width * height) / (1024 * 1024)
        return 2.0 + megapixels * sequence_length * 2.5

    def suggest_window(self, input_resolution: tuple, vram_gb: float) -> int:
        """Pick how many frames to process at once for the VRAM available."""
        width, height = input_resolution
        megapixels = max((width * height) / (1024 * 1024), 0.01)

        # Leave roughly a third of the card free for fragmentation and the
        # output buffers.
        budget = max(vram_gb * 0.6 - 2.0, 1.0)
        window = int(budget / (megapixels * 2.5))
        return max(2, min(window, 30))

    def clear_cache(self):
        """Clear the CUDA cache."""
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def create_basicvsrpp_model(
    device: str = 'cuda',
    model_name: str = 'basicvsrpp_reds',
) -> BasicVSRPlusPlusModel:
    """Factory for the BasicVSR++ model."""
    return BasicVSRPlusPlusModel(device=device, model_name=model_name)
