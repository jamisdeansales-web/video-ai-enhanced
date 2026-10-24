"""
SwinIR Model Wrapper
Transformer-based super-resolution, slower than Real-ESRGAN but higher quality.

The architecture ships inside basicsr, so no extra dependency is needed; only
the pretrained weights are fetched. Unlike Real-ESRGAN there is no helper class
equivalent to RealESRGANer, so padding, tiling and normalisation are handled
here.

The public surface matches RealESRGANModel (upscale_image, estimate_vram_usage,
get_optimal_tile_size, clear_cache), so SpatialUpscaler can drive either one.
"""

import logging
from typing import Optional

import cv2
import numpy as np
import torch

try:
    from basicsr.archs.swinir_arch import SwinIR
    SWINIR_AVAILABLE = True
except ImportError:
    SWINIR_AVAILABLE = False
    logging.warning(
        "SwinIR not available. Please install: pip install basicsr "
        "(and run scripts/patch_basicsr.py)"
    )

from utils.model_downloader import download_model

logger = logging.getLogger(__name__)


# Hyper-parameters of the released checkpoints. They are not stored in the
# files, so loading the wrong set silently produces a shape mismatch.
MODEL_CONFIGS = {
    # The official x2 real-world checkpoint cannot be used: it was trained with
    # the 'nearest+conv' upsampler, and basicsr's SwinIR asserts upscale == 4
    # for that branch (it always builds conv_up1 *and* conv_up2, while the x2
    # weights only carry conv_up1). The classical-SR x2 checkpoint uses
    # pixelshuffle, which basicsr supports at any scale, so that is the x2
    # model here. It is trained on bicubic downscaling rather than real
    # degradations, so it is sharper on clean sources and less forgiving of
    # compression artefacts than the x4 model.
    'swinir_x2': {
        'scale': 2,
        'window_size': 8,
        'embed_dim': 180,
        'depths': (6, 6, 6, 6, 6, 6),
        'num_heads': (6, 6, 6, 6, 6, 6),
        'mlp_ratio': 2,
        'upsampler': 'pixelshuffle',
        'resi_connection': '1conv',
    },
    'swinir_x4': {
        'scale': 4,
        'window_size': 8,
        'embed_dim': 180,
        'depths': (6, 6, 6, 6, 6, 6),
        'num_heads': (6, 6, 6, 6, 6, 6),
        'mlp_ratio': 2,
        'upsampler': 'nearest+conv',
        'resi_connection': '1conv',
    },
    'swinir_x4_large': {
        'scale': 4,
        'window_size': 8,
        'embed_dim': 240,
        'depths': (6, 6, 6, 6, 6, 6, 6, 6, 6),
        'num_heads': (8, 8, 8, 8, 8, 8, 8, 8, 8),
        'mlp_ratio': 2,
        'upsampler': 'nearest+conv',
        'resi_connection': '3conv',
    },
}


class SwinIRModel:
    """SwinIR model wrapper"""

    def __init__(
        self,
        scale: int = 4,
        model_name: str = 'swinir_x4',
        device: str = 'cuda',
        tile_size: int = 0,
        tile_pad: int = 32,
        fp16: bool = False,
    ):
        """
        Initialize the SwinIR model.

        Args:
            scale: Upscaling factor (2 or 4).
            model_name: Key into MODEL_CONFIGS.
            device: 'cuda' or 'cpu'.
            tile_size: Tile edge in pixels; 0 processes the whole frame at once.
            tile_pad: Overlap between tiles, trimmed after inference so seams do
                not show.
            fp16: Half precision. CUDA only.
        """
        if not SWINIR_AVAILABLE:
            raise ImportError(
                "SwinIR is not available. basicsr must be installed and "
                "patched; run scripts/patch_basicsr.py."
            )

        if model_name not in MODEL_CONFIGS:
            raise ValueError(
                "Unknown SwinIR model %r; expected one of %s"
                % (model_name, ', '.join(MODEL_CONFIGS))
            )

        config = MODEL_CONFIGS[model_name]
        if config['scale'] != scale:
            raise ValueError(
                "%s is a %dx model, not %dx" % (model_name, config['scale'], scale)
            )

        self.scale = scale
        self.model_name = model_name
        self.device = device
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        # SwinIR runs in fp32 whatever the caller asks for. Its attention
        # overflows the fp16 range and the network returns all-NaN, which is
        # worse than slow because nothing raises: the NaNs only show up as a
        # corrupted frame. bfloat16 is numerically fine but measured ~15x
        # slower than fp32 here, so it is not a useful middle ground either.
        if fp16:
            logger.info(
                "Ignoring the fp16 request: SwinIR produces NaN in half "
                "precision, so it runs in fp32."
            )
        self.fp16 = False
        self.window_size = config['window_size']
        # Frames that failed inference and were resized instead.
        self.fallback_count = 0

        logger.info("Loading SwinIR model: %s", model_name)
        model_path = download_model(model_name)
        if model_path is None:
            raise RuntimeError("Failed to download model: %s" % model_name)

        self.model = SwinIR(
            upscale=scale,
            in_chans=3,
            img_size=64,
            window_size=config['window_size'],
            img_range=1.0,
            depths=list(config['depths']),
            embed_dim=config['embed_dim'],
            num_heads=list(config['num_heads']),
            mlp_ratio=config['mlp_ratio'],
            upsampler=config['upsampler'],
            resi_connection=config['resi_connection'],
        )

        state = torch.load(model_path, map_location='cpu', weights_only=True)
        # The released checkpoints wrap the weights; EMA is the one to prefer.
        for key in ('params_ema', 'params'):
            if key in state:
                state = state[key]
                break
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

        try:
            self._to_device(self.device)
        except (torch.cuda.CudaError, RuntimeError) as exc:
            # Same guard as the Real-ESRGAN wrapper: a GPU newer than the torch
            # build has no kernels and fails only once weights reach the device.
            if device == 'cuda' and 'cudaErrorNoKernelImageForDevice' in str(exc):
                logger.warning(
                    "CUDA kernel not available for this GPU, falling back to CPU"
                )
                self.device = 'cpu'
                self.fp16 = False
                self._to_device('cpu')
            else:
                raise

        logger.info(
            "SwinIR %dx model loaded successfully on %s%s",
            scale, self.device, " (fp16)" if self.fp16 else "",
        )

    def _to_device(self, device: str):
        # The weights stay in fp32 and half precision is applied with autocast
        # instead of model.half(). basicsr's SwinIR rebuilds its attention mask
        # at run time whenever the input size differs from img_size, and builds
        # it in fp32; added to half attention scores it promotes them back to
        # fp32, and the following matmul against a half `v` raises
        # "expected scalar type Float but found Half". Autocast casts around
        # each op, so the mixed dtypes never meet.
        self.model = self.model.to(device)

    def _infer(self, tensor: torch.Tensor) -> torch.Tensor:
        """Run the network over a whole tensor, tiling it when asked to."""
        if not self.tile_size:
            return self.model(tensor)

        batch, channels, height, width = tensor.shape
        output = tensor.new_zeros(
            (batch, channels, height * self.scale, width * self.scale)
        )

        for top in range(0, height, self.tile_size):
            for left in range(0, width, self.tile_size):
                bottom = min(top + self.tile_size, height)
                right = min(left + self.tile_size, width)

                # Grow the tile by tile_pad on each side so the network has
                # context, then keep only the middle of the result.
                pad_top = max(top - self.tile_pad, 0)
                pad_left = max(left - self.tile_pad, 0)
                pad_bottom = min(bottom + self.tile_pad, height)
                pad_right = min(right + self.tile_pad, width)

                tile = tensor[:, :, pad_top:pad_bottom, pad_left:pad_right]
                upscaled = self.model(tile)

                keep_top = (top - pad_top) * self.scale
                keep_left = (left - pad_left) * self.scale
                output[
                    :, :,
                    top * self.scale:bottom * self.scale,
                    left * self.scale:right * self.scale,
                ] = upscaled[
                    :, :,
                    keep_top:keep_top + (bottom - top) * self.scale,
                    keep_left:keep_left + (right - left) * self.scale,
                ]

        return output

    def upscale_image(
        self,
        image: np.ndarray,
        outscale: Optional[float] = None,
    ) -> np.ndarray:
        """
        Upscale a single image.

        Args:
            image: Input image, BGR, as read by cv2.
            outscale: Final scale. Defaults to the model's own scale; any other
                value is reached by resizing the model output.

        Returns:
            numpy.ndarray: Upscaled image, BGR.
        """
        try:
            if image.dtype != np.uint8:
                image = (image * 255).clip(0, 255).astype(np.uint8)

            source_h, source_w = image.shape[:2]

            # BGR uint8 HWC -> RGB float NCHW
            rgb = image[:, :, [2, 1, 0]].astype(np.float32) / 255.0
            tensor = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0)
            tensor = tensor.to(self.device)

            # SwinIR needs both sides to be a multiple of the window size.
            # Reflecting the edge avoids the border artefacts zero padding
            # leaves behind.
            pad_h = (self.window_size - source_h % self.window_size) % self.window_size
            pad_w = (self.window_size - source_w % self.window_size) % self.window_size
            if pad_h or pad_w:
                tensor = torch.cat([tensor, torch.flip(tensor, [2])], 2)[
                    :, :, : source_h + pad_h, :
                ]
                tensor = torch.cat([tensor, torch.flip(tensor, [3])], 3)[
                    :, :, :, : source_w + pad_w
                ]

            with torch.no_grad():
                output = self._infer(tensor)

            # A numerically broken forward pass returns NaN rather than
            # raising, and NaN survives clamp() to become arbitrary bytes. Turn
            # it into a failure so it goes through the fallback and gets
            # counted instead of being written out as a corrupted frame.
            if not torch.isfinite(output).all():
                raise RuntimeError(
                    "model returned non-finite values (NaN or inf)"
                )

            # Drop the padding, in output-resolution pixels.
            output = output[:, :, : source_h * self.scale, : source_w * self.scale]

            result = output.squeeze(0).float().clamp_(0, 1).cpu().numpy()
            result = np.transpose(result[[2, 1, 0], :, :], (1, 2, 0))
            result = (result * 255.0).round().astype(np.uint8)

            if outscale is not None and float(outscale) != float(self.scale):
                result = cv2.resize(
                    result,
                    (int(source_w * outscale), int(source_h * outscale)),
                    interpolation=cv2.INTER_LANCZOS4,
                )

            return result

        except Exception as exc:  # noqa: BLE001 - never kill a whole video
            # Falling back keeps a long job alive, but a frame resized with
            # Lanczos is not an upscaled frame. Count them so a systematic
            # failure cannot pass for a successful run.
            self.fallback_count += 1
            if self.fallback_count == 1:
                logger.error(
                    "SwinIR failed on a frame and fell back to Lanczos: %s", exc
                )
            elif self.fallback_count % 50 == 0:
                logger.error(
                    "SwinIR has now fallen back to Lanczos on %d frames",
                    self.fallback_count,
                )

            scale = self.scale if outscale is None else outscale
            height, width = image.shape[:2]
            return cv2.resize(
                image,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_LANCZOS4,
            )

    def upscale_image_pil(self, pil_image):
        """Upscale a PIL Image and return a PIL Image."""
        from PIL import Image

        img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        output_bgr = self.upscale_image(img_bgr)
        return Image.fromarray(cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB))

    def estimate_vram_usage(self, input_resolution: tuple) -> float:
        """
        Estimate VRAM usage in GB.

        SwinIR holds attention maps over the whole input, so it needs
        substantially more memory than Real-ESRGAN at the same resolution.
        """
        width, height = input_resolution

        if self.tile_size:
            # Tiling caps the working set at one padded tile.
            edge = self.tile_size + 2 * self.tile_pad
            megapixels = (edge * edge) / (1024 * 1024)
        else:
            megapixels = (width * height) / (1024 * 1024)

        base_usage = 3.0
        per_megapixel = 4.0 if self.fp16 else 8.0

        return base_usage + megapixels * per_megapixel * self.scale

    def get_optimal_tile_size(self, vram_gb: float) -> int:
        """Pick a tile size for the VRAM available."""
        if vram_gb >= 24:
            return 512
        elif vram_gb >= 16:
            return 384
        elif vram_gb >= 12:
            return 256
        elif vram_gb >= 8:
            return 192
        return 128

    def clear_cache(self):
        """Clear the CUDA cache."""
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def create_swinir_model(
    scale: int = 4,
    device: str = 'cuda',
    fp16: bool = False,
    tile_size: int = 0,
    large: bool = False,
) -> SwinIRModel:
    """
    Factory for the SwinIR model.

    Args:
        scale: 2 or 4.
        device: 'cuda' or 'cpu'.
        fp16: Half precision on CUDA.
        tile_size: 0 lets the VRAM decide.
        large: Use the larger x4 checkpoint, slower and heavier.

    Returns:
        SwinIRModel
    """
    if scale == 2:
        model_name = 'swinir_x2'
    elif scale == 4:
        model_name = 'swinir_x4_large' if large else 'swinir_x4'
    else:
        raise ValueError("Unsupported scale: %s. Use 2 or 4." % scale)

    # SwinIR is far hungrier than Real-ESRGAN, so tile by default unless there
    # is plenty of VRAM; without this a 1080p frame exhausts most cards.
    if tile_size == 0 and device == 'cuda':
        try:
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                if vram_gb >= 24:
                    tile_size = 512
                elif vram_gb >= 16:
                    tile_size = 384
                elif vram_gb >= 12:
                    tile_size = 256
                else:
                    tile_size = 192
        except Exception:  # noqa: BLE001 - fall back to no tiling
            tile_size = 256

    return SwinIRModel(
        scale=scale,
        model_name=model_name,
        device=device,
        tile_size=tile_size,
        fp16=fp16,
    )
