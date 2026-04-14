import json

import torch
from plyfile import PlyData

from diffsynth.auxiliary_models.worldmirror.models.models.rasterization import Gaussians
from diffsynth.utils.gaussian_bundle import export_neoverse_4dgs_bundle


def make_gaussian(timestamp: int, x_offset: float) -> Gaussians:
    means = torch.tensor([[x_offset, 0.0, 0.0], [x_offset + 0.5, 0.25, 0.1]], dtype=torch.float32)
    harmonics = torch.tensor(
        [
            [[0.1, 0.2, 0.3]],
            [[0.3, 0.4, 0.5]],
        ],
        dtype=torch.float32,
    )
    opacities = torch.tensor([0.7, 0.85], dtype=torch.float32)
    scales = torch.tensor([[0.1, 0.2, 0.3], [0.12, 0.22, 0.32]], dtype=torch.float32)
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    return Gaussians(
        means=means,
        harmonics=harmonics,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        timestamp=timestamp,
    )


def test_4dgs_bundle_export_writes_supersplat_style_ply_sequence(tmp_path):
    predictions = {
        "splats": [[make_gaussian(0, 0.0), make_gaussian(1, 2.0)]],
        "rendered_intrinsics": torch.eye(3, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1),
        "rendered_extrinsics": torch.eye(4, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1),
        "rendered_timestamps": torch.tensor([[0.0, 1.0]], dtype=torch.float32),
    }

    manifest_path = export_neoverse_4dgs_bundle(
        predictions,
        tmp_path / "bundle",
        image_width=560,
        image_height=336,
        reconstructor_name="neoverse",
        scene_type="General scene",
        source_path="examples/videos/driving.mp4",
    )

    manifest = json.loads(manifest_path.read_text())
    bundle_dir = manifest_path.parent
    timeline = json.loads((bundle_dir / manifest["files"]["timeline"]).read_text())

    assert manifest["bundle_version"] == "1.0"
    assert manifest["capabilities"]["export_format"] == "ply_sequence"
    assert manifest["capabilities"]["super_splat_compatible"] is True
    assert (bundle_dir / manifest["files"]["cameras"]).exists()
    assert manifest["files"]["gaussian_frames_dir"] == "gaussians"
    assert manifest["files"]["gaussian_frame_pattern"] == "animation_{:04d}.ply"
    assert manifest["files"]["first_frame_ply"] == "gaussians/animation_0001.ply"

    assert timeline["frame_count"] == 2
    assert [frame["ply_path"] for frame in timeline["frames"]] == [
        "gaussians/animation_0001.ply",
        "gaussians/animation_0002.ply",
    ]

    for frame in timeline["frames"]:
        ply = PlyData.read(bundle_dir / frame["ply_path"])
        assert "ttt" not in ply["vertex"].data.dtype.names

    first_frame = PlyData.read(bundle_dir / "gaussians/animation_0001.ply")
    second_frame = PlyData.read(bundle_dir / "gaussians/animation_0002.ply")
    assert first_frame["vertex"].count == 2
    assert second_frame["vertex"].count == 2
