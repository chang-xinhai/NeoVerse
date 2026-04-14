from __future__ import annotations

import torch
import torch.nn as nn

from .worldmirror.models.models.rasterization import Gaussians, Rasterizer
from .worldmirror.models.utils.geometry import depth_to_world_coords_points
from .worldmirror.models.utils.sh_utils import RGB2SH


class PseudoGaussianRenderer:
    """Matching pipe.reconstructor.gs_renderer.rasterizer."""

    def __init__(self) -> None:
        self.rasterizer = Rasterizer()


class PseudoGaussianReconstructor(nn.Module):
    def __init__(self, gaussian_scale: float) -> None:
        super().__init__()
        self.gs_renderer = PseudoGaussianRenderer()
        self.gaussian_scale = gaussian_scale

    def build_predictions(
        self,
        views: dict[str, torch.Tensor],
        depth: torch.Tensor,
        c2w: torch.Tensor,
        intrinsics: torch.Tensor,
        depth_conf: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list[list[Gaussians]]]:
        imgs = views["img"]
        timestamps = views["timestamp"]
        static_mask = views["is_static"]
        batch_size, sequence_length, _, height, width = imgs.shape
        if static_mask.shape[1] == 1:
            static_mask = static_mask.expand(batch_size, sequence_length)

        if depth.dim() == 5 and depth.shape[-1] == 1:
            depth_map = depth.squeeze(-1)
        else:
            depth_map = depth
        depth_map = depth_map.float()
        c2w = c2w.float()
        intrinsics = intrinsics.float()

        world_coords, _, valid_mask = depth_to_world_coords_points(
            depth_map.reshape(batch_size * sequence_length, height, width),
            c2w.reshape(batch_size * sequence_length, 4, 4),
            intrinsics.reshape(batch_size * sequence_length, 3, 3),
        )
        world_coords = world_coords.view(batch_size, sequence_length, height, width, 3)
        valid_mask = valid_mask.view(batch_size, sequence_length, height, width)

        pixel_rgb = imgs.permute(0, 1, 3, 4, 2)
        splats: list[list[Gaussians]] = []
        for batch_index in range(batch_size):
            batch_gaussians: list[Gaussians] = []
            for frame_index in range(sequence_length):
                mask = valid_mask[batch_index, frame_index]
                points = world_coords[batch_index, frame_index][mask]
                rgb = pixel_rgb[batch_index, frame_index][mask]
                if points.shape[0] == 0:
                    continue
                harmonics = RGB2SH(rgb).unsqueeze(1)
                scales = points.new_full((points.shape[0], 3), self.gaussian_scale)
                rotations = points.new_zeros(points.shape[0], 4)
                rotations[:, 0] = 1.0
                opacities = points.new_ones(points.shape[0])
                confidences = points.new_ones(points.shape[0])
                timestamp = -1 if bool(static_mask[batch_index, frame_index]) else int(timestamps[batch_index, frame_index].item())
                batch_gaussians.append(
                    Gaussians(
                        means=points,
                        harmonics=harmonics,
                        opacities=opacities,
                        scales=scales,
                        rotations=rotations,
                        confidences=confidences,
                        timestamp=timestamp,
                    )
                )
            splats.append(batch_gaussians)

        if depth_conf is None:
            depth_conf = valid_mask.unsqueeze(-1).float()
        elif depth_conf.dim() == 4:
            depth_conf = depth_conf.unsqueeze(-1)
        depth_conf = depth_conf.float()

        return {
            "splats": splats,
            "rendered_extrinsics": c2w,
            "rendered_intrinsics": intrinsics,
            "rendered_timestamps": timestamps,
            "gs_depth": depth_map.unsqueeze(-1),
            "gs_depth_conf": depth_conf,
        }
