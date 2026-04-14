import torch
from plyfile import PlyData

from diffsynth.auxiliary_models.worldmirror.models.models.rasterization import Gaussians
from diffsynth.utils.gaussian_bundle import write_static_3dgs_snapshot_ply


def make_gaussian(timestamp: int, x_offset: float) -> Gaussians:
    means = torch.tensor([[x_offset, 0.0, 0.0], [x_offset + 1.0, 0.5, 0.25]], dtype=torch.float32)
    harmonics = torch.tensor(
        [
            [[0.1, 0.2, 0.3]],
            [[0.4, 0.5, 0.6]],
        ],
        dtype=torch.float32,
    )
    opacities = torch.tensor([0.7, 0.8], dtype=torch.float32)
    scales = torch.tensor([[0.1, 0.2, 0.3], [0.2, 0.25, 0.35]], dtype=torch.float32)
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    return Gaussians(
        means=means,
        harmonics=harmonics,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        timestamp=timestamp,
    )


def test_snapshot_ply_uses_plain_frame_schema_without_ttt(tmp_path):
    output_path = tmp_path / "animation_0001.ply"
    write_static_3dgs_snapshot_ply(output_path, [make_gaussian(0, 0.0), make_gaussian(1, 2.0)], reference_timestamp=0)

    ply = PlyData.read(output_path)
    field_names = ply["vertex"].data.dtype.names

    assert "ttt" not in field_names
    assert ply["vertex"].count == 2
    assert {"x", "y", "z", "opacity", "scale_0", "rot_0", "f_dc_0"}.issubset(set(field_names))
