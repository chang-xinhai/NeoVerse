from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from .app import build_scene_glb, extract_point_cloud
from ..auxiliary_models.worldmirror.models.models.rasterization import Gaussians

BUNDLE_VERSION = "1.0"
FRAME_FILE_PATTERN = "animation_{:04d}.ply"


def _to_numpy(tensor: torch.Tensor, dtype: torch.dtype = torch.float32) -> np.ndarray:
    return tensor.detach().cpu().to(dtype=dtype).numpy()


def _inverse_sigmoid(opacities: torch.Tensor) -> torch.Tensor:
    return torch.logit(opacities.clamp(1e-6, 1 - 1e-6))


def _gaussian_dc(harmonics: torch.Tensor) -> torch.Tensor:
    if harmonics.ndim == 3:
        return harmonics[:, 0, :]
    if harmonics.ndim == 2 and harmonics.shape[1] >= 3:
        return harmonics[:, :3]
    raise ValueError(f"Expected harmonics with shape [N, K, 3] or [N, >=3], got {tuple(harmonics.shape)}")


def _base_vertex_dtype() -> list[tuple[str, str]]:
    return [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"),
        ("scale_1", "f4"),
        ("scale_2", "f4"),
        ("rot_0", "f4"),
        ("rot_1", "f4"),
        ("rot_2", "f4"),
        ("rot_3", "f4"),
    ]


def flatten_gaussians_batch(gaussians_batch: list[Gaussians]) -> dict[str, np.ndarray]:
    positions = []
    dc_features = []
    opacity_logits = []
    log_scales = []
    rotations = []
    timestamps = []

    for gs in gaussians_batch:
        if gs.means.numel() == 0:
            continue
        positions.append(_to_numpy(gs.means))
        dc_features.append(_to_numpy(_gaussian_dc(gs.harmonics)))
        opacity_logits.append(_to_numpy(_inverse_sigmoid(gs.opacities)))
        log_scales.append(_to_numpy(gs.scales.log()))
        rotations.append(_to_numpy(gs.rotations))
        timestamp_value = float(gs.timestamp if gs.timestamp >= 0 else 0)
        timestamps.append(np.full((gs.means.shape[0],), timestamp_value, dtype=np.float32))

    if not positions:
        return {
            "positions": np.zeros((0, 3), dtype=np.float32),
            "dc_features": np.zeros((0, 3), dtype=np.float32),
            "opacity_logits": np.zeros((0,), dtype=np.float32),
            "log_scales": np.zeros((0, 3), dtype=np.float32),
            "rotations": np.zeros((0, 4), dtype=np.float32),
            "timestamps": np.zeros((0,), dtype=np.float32),
        }

    return {
        "positions": np.concatenate(positions, axis=0).astype(np.float32),
        "dc_features": np.concatenate(dc_features, axis=0).astype(np.float32),
        "opacity_logits": np.concatenate(opacity_logits, axis=0).astype(np.float32),
        "log_scales": np.concatenate(log_scales, axis=0).astype(np.float32),
        "rotations": np.concatenate(rotations, axis=0).astype(np.float32),
        "timestamps": np.concatenate(timestamps, axis=0).astype(np.float32),
    }


def _snapshot_gaussians(gaussians_batch: list[Gaussians], timestamp: int | float) -> dict[str, np.ndarray]:
    positions = []
    dc_features = []
    opacity_logits = []
    log_scales = []
    rotations = []

    for gs in gaussians_batch:
        can_transition_forward = (
            gs.forward_timestamp is not None and gs.timestamp < timestamp < gs.forward_timestamp
        )
        can_transition_backward = (
            gs.backward_timestamp is not None and gs.backward_timestamp < timestamp < gs.timestamp
        )
        if gs.timestamp == -1 or gs.timestamp == timestamp:
            selected = gs
        elif can_transition_forward or can_transition_backward:
            selected = gs.transition(timestamp)
        else:
            continue
        if selected.means.numel() == 0:
            continue
        positions.append(_to_numpy(selected.means))
        dc_features.append(_to_numpy(_gaussian_dc(selected.harmonics)))
        opacity_logits.append(_to_numpy(_inverse_sigmoid(selected.opacities)))
        log_scales.append(_to_numpy(selected.scales.log()))
        rotations.append(_to_numpy(selected.rotations))

    if not positions:
        return {
            "positions": np.zeros((0, 3), dtype=np.float32),
            "dc_features": np.zeros((0, 3), dtype=np.float32),
            "opacity_logits": np.zeros((0,), dtype=np.float32),
            "log_scales": np.zeros((0, 3), dtype=np.float32),
            "rotations": np.zeros((0, 4), dtype=np.float32),
        }

    return {
        "positions": np.concatenate(positions, axis=0).astype(np.float32),
        "dc_features": np.concatenate(dc_features, axis=0).astype(np.float32),
        "opacity_logits": np.concatenate(opacity_logits, axis=0).astype(np.float32),
        "log_scales": np.concatenate(log_scales, axis=0).astype(np.float32),
        "rotations": np.concatenate(rotations, axis=0).astype(np.float32),
    }


def _build_vertex_elements(data: dict[str, np.ndarray]) -> np.ndarray:
    vertex_elements = np.empty(data["positions"].shape[0], dtype=_base_vertex_dtype())
    vertex_elements["x"] = data["positions"][:, 0]
    vertex_elements["y"] = data["positions"][:, 1]
    vertex_elements["z"] = data["positions"][:, 2]
    vertex_elements["nx"] = 0.0
    vertex_elements["ny"] = 0.0
    vertex_elements["nz"] = 0.0
    vertex_elements["f_dc_0"] = data["dc_features"][:, 0]
    vertex_elements["f_dc_1"] = data["dc_features"][:, 1]
    vertex_elements["f_dc_2"] = data["dc_features"][:, 2]
    vertex_elements["opacity"] = data["opacity_logits"]
    vertex_elements["scale_0"] = data["log_scales"][:, 0]
    vertex_elements["scale_1"] = data["log_scales"][:, 1]
    vertex_elements["scale_2"] = data["log_scales"][:, 2]
    vertex_elements["rot_0"] = data["rotations"][:, 0]
    vertex_elements["rot_1"] = data["rotations"][:, 1]
    vertex_elements["rot_2"] = data["rotations"][:, 2]
    vertex_elements["rot_3"] = data["rotations"][:, 3]
    return vertex_elements


def write_snapshot_ply(path: str | Path, snapshot_data: dict[str, np.ndarray]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    elements = _build_vertex_elements(snapshot_data)
    PlyData([PlyElement.describe(elements, "vertex")]).write(output_path)
    return output_path


def write_static_3dgs_snapshot_ply(
    path: str | Path,
    gaussians_batch: list[Gaussians],
    reference_timestamp: int | float,
) -> Path:
    snapshot_data = _snapshot_gaussians(gaussians_batch, reference_timestamp)
    return write_snapshot_ply(path, snapshot_data)


def write_frame_sequence_plys(
    output_dir: str | Path,
    gaussians_batch: list[Gaussians],
    timestamps: np.ndarray,
) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []
    for index, timestamp in enumerate(timestamps, start=1):
        frame_path = output_path / FRAME_FILE_PATTERN.format(index)
        snapshot_data = _snapshot_gaussians(gaussians_batch, float(timestamp))
        write_snapshot_ply(frame_path, snapshot_data)
        frame_paths.append(frame_path)
    return frame_paths


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
    return path


def export_neoverse_4dgs_bundle(
    predictions: dict[str, Any],
    output_dir: str | Path,
    *,
    image_width: int,
    image_height: int,
    reconstructor_name: str,
    scene_type: str,
    source_path: str | None = None,
) -> Path:
    if len(predictions["splats"]) != 1:
        raise ValueError("Bundle export currently supports batch size 1 only.")

    bundle_dir = Path(output_dir)
    gaussians_dir = bundle_dir / "gaussians"
    metadata_dir = bundle_dir / "metadata"
    if gaussians_dir.exists():
        for stale_frame in gaussians_dir.glob("animation_*.ply"):
            stale_frame.unlink()
    gaussians_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    gaussians_batch = predictions["splats"][0]
    intrinsics = predictions["rendered_intrinsics"][0].detach().cpu().float().numpy()
    extrinsics = predictions["rendered_extrinsics"][0].detach().cpu().float().numpy()
    timestamps = predictions["rendered_timestamps"][0].detach().cpu().float().numpy()

    frame_paths = write_frame_sequence_plys(gaussians_dir, gaussians_batch, timestamps)

    cameras_payload = {
        "image_width": int(image_width),
        "image_height": int(image_height),
        "cameras": [
            {
                "camera_id": int(index),
                "frame_index": int(index),
                "timestamp": float(timestamps[index]),
                "intrinsics_3x3": intrinsics[index],
                "cam2world_4x4": extrinsics[index],
                "world2cam_4x4": np.linalg.inv(extrinsics[index]),
            }
            for index in range(len(timestamps))
        ],
    }
    cameras_path = _write_json(bundle_dir / "cameras.json", cameras_payload)

    timeline_payload = {
        "frame_count": int(len(timestamps)),
        "timestamp_units": "index",
        "is_static_scene": bool(scene_type == "Static scene"),
        "playback_hint_fps": 16,
        "frames": [
            {
                "frame_index": int(index),
                "timestamp": float(timestamp),
                "ply_path": str(frame_path.relative_to(bundle_dir)),
            }
            for index, (timestamp, frame_path) in enumerate(zip(timestamps, frame_paths))
        ],
    }
    timeline_path = _write_json(bundle_dir / "timeline.json", timeline_payload)

    flat_data = flatten_gaussians_batch(gaussians_batch)
    reconstruction_info_path = _write_json(
        metadata_dir / "reconstruction_info.json",
        {
            "reconstructor_name": reconstructor_name,
            "scene_type": scene_type,
            "source_path": source_path,
            "gaussian_count": int(flat_data["positions"].shape[0]),
            "timestamp_count": int(len(np.unique(flat_data["timestamps"]))),
            "camera_count": int(len(timestamps)),
            "frame_count": int(len(frame_paths)),
            "image_width": int(image_width),
            "image_height": int(image_height),
            "sh_export": "dc_only",
            "export_format": "ply_sequence",
        },
    )

    coordinate_conventions_path = _write_json(
        metadata_dir / "coordinate_conventions.json",
        {
            "camera_pose_field": "cam2world_4x4",
            "camera_convention_note": "GLB preview applies the existing OpenCV-to-OpenGL flip from diffsynth.utils.app.build_scene_glb.",
            "rotation_representation": "quaternion_wxyz",
            "scale_representation": "log_scale",
            "opacity_representation": "logit",
            "time_representation": "external_timeline_sequence",
        },
    )

    attribute_schema_path = _write_json(
        metadata_dir / "attribute_schema.json",
        {
            "vertex_properties": [name for name, _ in _base_vertex_dtype()],
            "per_frame_ply": True,
            "sequence_file_pattern": FRAME_FILE_PATTERN,
        },
    )

    manifest_path = _write_json(
        bundle_dir / "bundle_manifest.json",
        {
            "bundle_version": BUNDLE_VERSION,
            "source": {
                "reconstructor_name": reconstructor_name,
                "scene_type": scene_type,
                "source_path": source_path,
            },
            "counts": {
                "gaussians": int(flat_data["positions"].shape[0]),
                "timestamps": int(len(np.unique(flat_data["timestamps"]))),
                "cameras": int(len(timestamps)),
                "frames": int(len(frame_paths)),
            },
            "capabilities": {
                "export_format": "ply_sequence",
                "super_splat_compatible": True,
                "supports_neoverse_reimport": False,
                "reimport_mode": "unsupported",
                "dynamic_transition_metadata_preserved": False,
                "editable_fields_supported": [
                    "position",
                    "scale",
                    "rotation",
                    "dc_color",
                    "opacity",
                ],
            },
            "files": {
                "bundle_manifest": "bundle_manifest.json",
                "cameras": str(cameras_path.relative_to(bundle_dir)),
                "timeline": str(timeline_path.relative_to(bundle_dir)),
                "gaussian_frames_dir": str(gaussians_dir.relative_to(bundle_dir)),
                "gaussian_frame_pattern": FRAME_FILE_PATTERN,
                "first_frame_ply": str(frame_paths[0].relative_to(bundle_dir)) if frame_paths else None,
                "reconstruction_info": str(reconstruction_info_path.relative_to(bundle_dir)),
                "coordinate_conventions": str(coordinate_conventions_path.relative_to(bundle_dir)),
                "attribute_schema": str(attribute_schema_path.relative_to(bundle_dir)),
            },
        },
    )
    return manifest_path
