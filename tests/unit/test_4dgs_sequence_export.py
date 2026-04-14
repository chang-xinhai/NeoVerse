import json

import torch
from plyfile import PlyData

from diffsynth.auxiliary_models.worldmirror.models.models.rasterization import Gaussians
from diffsynth.utils.gaussian_bundle import export_neoverse_4dgs_bundle


def make_gaussian(timestamp: int, points: list[list[float]], flattened: bool = False) -> Gaussians:
    means = torch.tensor(points, dtype=torch.float32)
    if flattened:
        harmonics = torch.tensor([[0.1, 0.2, 0.3]] * len(points), dtype=torch.float32)
    else:
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


def test_sequence_export_accepts_flattened_harmonics(tmp_path):
    predictions = {
        "splats": [[make_gaussian(0, [[0.0, 0.0, 0.0]], flattened=True)]],
        "rendered_intrinsics": torch.eye(3, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
        "rendered_extrinsics": torch.eye(4, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
        "rendered_timestamps": torch.tensor([[0.0]], dtype=torch.float32),
    }

    manifest_path = export_neoverse_4dgs_bundle(
        predictions,
        tmp_path / "bundle_flattened",
        image_width=560,
        image_height=336,
        reconstructor_name="neoverse",
        scene_type="General scene",
    )
    manifest = json.loads(manifest_path.read_text())
    first_frame = manifest["files"]["first_frame_ply"]
    ply = PlyData.read(manifest_path.parent / first_frame)
    assert ply["vertex"].count == 1


def test_sequence_export_cleans_stale_frame_files(tmp_path):
    bundle_dir = tmp_path / "bundle_stale"
    stale_dir = bundle_dir / "gaussians"
    stale_dir.mkdir(parents=True)
    (stale_dir / "animation_0003.ply").write_text("stale", encoding="utf-8")

    predictions = {
        "splats": [[make_gaussian(0, [[0.0, 0.0, 0.0]]), make_gaussian(1, [[1.0, 0.0, 0.0]])]],
        "rendered_intrinsics": torch.eye(3, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1),
        "rendered_extrinsics": torch.eye(4, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1),
        "rendered_timestamps": torch.tensor([[0.0, 1.0]], dtype=torch.float32),
    }

    export_neoverse_4dgs_bundle(
        predictions,
        bundle_dir,
        image_width=560,
        image_height=336,
        reconstructor_name="neoverse",
        scene_type="General scene",
    )

    assert (stale_dir / "animation_0001.ply").exists()
    assert (stale_dir / "animation_0002.ply").exists()
    assert not (stale_dir / "animation_0003.ply").exists()
