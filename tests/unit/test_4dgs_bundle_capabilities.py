import json

import torch

from diffsynth.auxiliary_models.worldmirror.models.models.rasterization import Gaussians
from diffsynth.utils.gaussian_bundle import export_neoverse_4dgs_bundle



def make_dynamic_gaussian(timestamp: int, next_timestamp: int | None = None) -> Gaussians:
    means = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    harmonics = torch.tensor([[[0.1, 0.2, 0.3]]], dtype=torch.float32)
    opacities = torch.tensor([0.7], dtype=torch.float32)
    scales = torch.tensor([[0.1, 0.1, 0.1]], dtype=torch.float32)
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    forward_vel = torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float32) if next_timestamp is not None else None
    return Gaussians(
        means=means,
        harmonics=harmonics,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        timestamp=timestamp,
        forward_timestamp=next_timestamp,
        forward_vel=forward_vel,
    )


def test_sequence_bundle_advertises_supersplat_export_without_reimport(tmp_path):
    predictions = {
        "splats": [[make_dynamic_gaussian(0, next_timestamp=1), make_dynamic_gaussian(1)]],
        "rendered_intrinsics": torch.eye(3, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1),
        "rendered_extrinsics": torch.eye(4, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1),
        "rendered_timestamps": torch.tensor([[0.0, 1.0]], dtype=torch.float32),
    }

    manifest_path = export_neoverse_4dgs_bundle(
        predictions,
        tmp_path / "bundle_dynamic",
        image_width=560,
        image_height=336,
        reconstructor_name="neoverse",
        scene_type="General scene",
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["capabilities"]["export_format"] == "ply_sequence"
    assert manifest["capabilities"]["super_splat_compatible"] is True
    assert manifest["capabilities"]["supports_neoverse_reimport"] is False
    assert manifest["capabilities"]["reimport_mode"] == "unsupported"
    assert "supports_blender_viewer_import" not in manifest["capabilities"]
