"""
RIFE IFNet architecture (v4.x).

RIFE has no pip package, and unlike SwinIR or BasicVSR++ the architecture is
not bundled with basicsr either, so it is defined here. Every layer below was
derived from the published checkpoint's own tensor names and shapes rather
than written from memory, and `load_state_dict(..., strict=True)` in
rife_model.py is what proves the two agree: a mismatch raises instead of
quietly leaving layers randomly initialised.

Reference implementation: https://github.com/hzwer/Practical-RIFE (MIT).

Shapes the checkpoint dictates:

    encode.cnn0   [32, 3, 3, 3]     3 -> 32, stride 2
    encode.cnn3   [32, 8, 4, 4]     transposed, 32 -> 8
    block0.conv0  [96, 23, 3, 3]    23 = img0 3 + img1 3 + t 1 + f0 8 + f1 8
    block1..3     [.., 28, 3, 3]    28 = 23 + flow 4 + mask 1
    lastconv      [c, 24, 4, 4]     transposed, then PixelShuffle(2) -> 6
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    """Conv + LeakyReLU, as one Sequential so index .0 is the convolution."""
    return nn.Sequential(
        nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=True,
        ),
        nn.LeakyReLU(0.2, True),
    )


class ResConv(nn.Module):
    """Residual conv with a learnable per-channel scale on the residual."""

    def __init__(self, c, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class Head(nn.Module):
    """Feature encoder run on each input frame (the `encode` weights)."""

    def __init__(self):
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 32, 3, 2, 1)
        self.cnn1 = nn.Conv2d(32, 32, 3, 1, 1)
        self.cnn2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(32, 8, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = self.relu(self.cnn0(x))
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        return self.cnn3(x)


class IFBlock(nn.Module):
    """One refinement stage: predicts a flow update and a blending mask."""

    def __init__(self, in_planes, c=64):
        super().__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(*[ResConv(c) for _ in range(8)])
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4 * 6, 4, 2, 1),
            nn.PixelShuffle(2),
        )

    def forward(self, x, flow=None, scale=1):
        # Each stage works at a coarser resolution than the last; the flow it
        # inherits has to be rescaled with the image, in value as well as size.
        x = F.interpolate(
            x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
        )
        if flow is not None:
            flow = F.interpolate(
                flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
            ) * (1.0 / scale)
            x = torch.cat((x, flow), 1)

        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = F.interpolate(
            tmp, scale_factor=scale, mode="bilinear", align_corners=False
        )
        return tmp[:, :4] * scale, tmp[:, 4:5]


def warp(frame, flow):
    """Backward-warp `frame` by `flow`, in pixels, with a normalised grid."""
    _, _, height, width = frame.shape
    device = frame.device

    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, device=device, dtype=frame.dtype),
        torch.arange(width, device=device, dtype=frame.dtype),
        indexing='ij',
    )
    grid = torch.stack((grid_x, grid_y), 0).unsqueeze(0)

    moved = grid + flow
    # grid_sample wants coordinates in [-1, 1].
    moved_x = 2.0 * moved[:, 0:1] / max(width - 1, 1) - 1.0
    moved_y = 2.0 * moved[:, 1:2] / max(height - 1, 1) - 1.0
    normalised = torch.cat((moved_x, moved_y), 1).permute(0, 2, 3, 1)

    return F.grid_sample(
        frame, normalised, mode='bilinear',
        padding_mode='border', align_corners=True,
    )


class IFNet(nn.Module):
    """
    RIFE v4.x.

    Four stages refine a bidirectional flow from coarse to fine; each warps
    both inputs towards the requested timestep, and the final mask decides how
    much of each warped frame the output takes.
    """

    def __init__(self):
        super().__init__()
        self.block0 = IFBlock(7 + 16, c=192)
        self.block1 = IFBlock(8 + 4 + 16, c=128)
        self.block2 = IFBlock(8 + 4 + 16, c=96)
        self.block3 = IFBlock(8 + 4 + 16, c=64)
        self.encode = Head()

    def forward(self, img0, img1, timestep=0.5, scale_list=(8, 4, 2, 1)):
        """
        Args:
            img0, img1: (N, 3, H, W) in [0, 1].
            timestep: where between them to land, 0 to 1. A tensor broadcast
                over the batch also works.
            scale_list: resolution divisor per stage, coarse to fine.

        Returns:
            (N, 3, H, W) interpolated frame.
        """
        if not torch.is_tensor(timestep):
            timestep = torch.full(
                (img0.shape[0], 1, img0.shape[2], img0.shape[3]),
                float(timestep), device=img0.device, dtype=img0.dtype,
            )

        feat0 = self.encode(img0)
        feat1 = self.encode(img1)

        flow = None
        mask = None
        warped0, warped1 = img0, img1

        for index, block in enumerate(
            (self.block0, self.block1, self.block2, self.block3)
        ):
            if flow is None:
                flow, mask = block(
                    torch.cat((img0, img1, feat0, feat1, timestep), 1),
                    None, scale=scale_list[index],
                )
            else:
                # Later stages see the current warps and correct the flow
                # rather than predicting it afresh.
                warped_feat0 = warp(feat0, flow[:, :2])
                warped_feat1 = warp(feat1, flow[:, 2:4])
                delta_flow, mask = block(
                    torch.cat(
                        (warped0, warped1, warped_feat0, warped_feat1,
                         timestep, mask), 1
                    ),
                    flow, scale=scale_list[index],
                )
                flow = flow + delta_flow

            warped0 = warp(img0, flow[:, :2])
            warped1 = warp(img1, flow[:, 2:4])

        blend = torch.sigmoid(mask)
        return torch.clamp(warped0 * blend + warped1 * (1 - blend), 0, 1)
