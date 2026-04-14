import json

import torch

from diffsynth.auxiliary_models.worldmirror.models.models.rasterization import Gaussians
from diffsynth.utils.gaussian_bundle import export_neoverse_4dgs_bundle


def make_gaussian(timestamp: int, points: list[list[float]]) -> Gaussians:
    means = torch.tensor(points, dtype=torch.float32)
    harmonics = torch.tensor([[[0.1, 0.2, 0.3]]] * len(points), dtype=torch.float32)
    opacities = torch.tensor([0.7] * len(points), dtype=torch.float32)
    scales = torch.tensor([[0.1, 0.1, 0.1]] * len(points), dtype=torch.float32)
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * len(points), dtype=torch.float32)
    return Gaussians(
        means=means,
        harmonics=harmonics,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        timestamp=timestamp,
    )


def test_bundle_timeline_points_to_ordered_frame_sequence(tmp_path):
    predictions = {
        "splats": [[
            make_gaussian(0, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            make_gaussian(2, [[2.0, 0.0, 0.0]]),
            make_gaussian(5, [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
        ]],
        "rendered_intrinsics": torch.eye(3, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1),
        "rendered_extrinsics": torch.eye(4, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1),
        "rendered_timestamps": torch.tensor([[0.0, 2.0, 5.0]], dtype=torch.float32),
    }

    manifest_path = export_neoverse_4dgs_bundle(
        predictions,
        tmp_path / "bundle_sequence",
        image_width=560,
        image_height=336,
        reconstructor_name="neoverse",
        scene_type="General scene",
    )

    bundle_dir = manifest_path.parent
    timeline = json.loads((bundle_dir / "timeline.json").read_text())

    assert timeline["frame_count"] == 3
    assert "timestamps" not in timeline
    assert [frame["ply_path"] for frame in timeline["frames"]] == [
        "gaussians/animation_0001.ply",
        "gaussians/animation_0002.ply",
        "gaussians/animation_0003.ply",
    ]
    assert [frame["timestamp"] for frame in timeline["frames"]] == [0.0, 2.0, 5.0]
    for frame in timeline["frames"]:
        assert (bundle_dir / frame["ply_path"]).exists()
