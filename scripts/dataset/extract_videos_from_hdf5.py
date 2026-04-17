"""
Extract video streams from HDF5 episode files as standard MP4.

Supports:
  - front   : RGB camera (3-channel, 1080p typically)
  - ir_left : Left IR camera (grayscale)
  - ir_right: Right IR camera (grayscale)

Usage:
    # Single file, all streams
    python scripts/dataset/extract_videos_from_hdf5.py \
        data/vcl_umi/umi_0202/episode_10.hdf5

    # Single file, specific streams
    python scripts/dataset/extract_videos_from_hdf5.py \
        data/vcl_umi/umi_0202/episode_10.hdf5 \
        --streams front

    # Batch: glob pattern
    python scripts/dataset/extract_videos_from_hdf5.py \
        "data/vcl_umi/umi_0202/episode_*.hdf5" \
        --output-dir data/vcl_umi/umi_0202/video/

    # Per-episode subdirectory layout (default)
    #   episode_10.hdf5 → data/vcl_umi/umi_0202/video/episode_10/
    #                       ├── front.mp4
    #                       ├── ir_left.mp4
    #                       └── ir_right.mp4
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import h5py
import numpy as np


STREAM_CONFIGS = {
    "front": {"fps": 30, "pix_fmt": "yuv420p", "crf": 18, "grayscale": False},
    "ir_left": {"fps": 30, "pix_fmt": "yuv420p", "crf": 18, "grayscale": True},
    "ir_right": {"fps": 30, "pix_fmt": "yuv420p", "crf": 18, "grayscale": True},
}
REQUIRED_COMPRESSED_KEYS = ("data", "offsets", "lengths")
SUPPORTED_FRAME_DTYPES = (np.uint8,)


def get_stream_key(stream: str) -> str:
    return f"observations/images/{stream}"


def describe_stream_node(node: h5py.Dataset | h5py.Group) -> dict:
    if isinstance(node, h5py.Dataset):
        return {
            "storage": "dataset",
            "shape": node.shape,
            "dtype": str(node.dtype),
        }

    if isinstance(node, h5py.Group):
        missing = [name for name in REQUIRED_COMPRESSED_KEYS if name not in node]
        if missing:
            return {
                "storage": "group",
                "missing_keys": missing,
            }

        lengths = node["lengths"]
        return {
            "storage": "compressed",
            "frames": int(lengths.shape[0]),
            "dtype": str(node["data"].dtype),
        }

    return {"storage": type(node).__name__}


def inspect_hdf5(path: str) -> dict:
    """Print HDF5 structure and return stream info."""
    with h5py.File(path, "r") as h5_file:
        lines = []

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                shape = obj.shape if hasattr(obj, "shape") else ()
                dtype = obj.dtype if hasattr(obj, "dtype") else ""
                lines.append(f"  {name}: shape={shape}, dtype={dtype}")
            elif isinstance(obj, h5py.Group):
                lines.append(f"  {name}/")

        h5_file.visititems(visitor)
        structure = "\n".join(lines)
        print(f"\n=== HDF5 structure: {path} ===")
        print(structure)
        print("=" * 60)

        stream_info = {}
        for name in STREAM_CONFIGS:
            key = get_stream_key(name)
            if key not in h5_file:
                continue
            stream_info[name] = describe_stream_node(h5_file[key])
        return stream_info


def validate_uint8_dtype(frames: np.ndarray, stream: str) -> np.ndarray:
    normalized = np.asarray(frames)
    if normalized.dtype.type not in SUPPORTED_FRAME_DTYPES:
        raise ValueError(
            f"Unsupported dtype for stream '{stream}': {normalized.dtype}. Expected uint8 frames."
        )
    return normalized


def normalize_frame(frame: np.ndarray, stream: str, grayscale: bool) -> np.ndarray:
    if frame is None:
        raise ValueError(f"Decoded frame is None for stream '{stream}'")

    normalized = validate_uint8_dtype(frame, stream)

    if grayscale:
        if normalized.ndim == 3 and normalized.shape[2] == 1:
            normalized = normalized[:, :, 0]
        if normalized.ndim != 2:
            raise ValueError(
                f"Expected grayscale frame for stream '{stream}', got shape {normalized.shape}"
            )
    else:
        if normalized.ndim != 3 or normalized.shape[2] != 3:
            raise ValueError(
                f"Expected 3-channel frame for stream '{stream}', got shape {normalized.shape}"
            )

    return np.ascontiguousarray(normalized)


def normalize_frames(frames: np.ndarray, stream: str, grayscale: bool) -> np.ndarray:
    normalized = validate_uint8_dtype(frames, stream)
    if normalized.size == 0:
        raise ValueError(f"No frames found for stream '{stream}'")

    if grayscale:
        if normalized.ndim == 4 and normalized.shape[-1] == 1:
            normalized = normalized[..., 0]
        if normalized.ndim != 3:
            raise ValueError(
                f"Expected grayscale frame batch for stream '{stream}', got shape {normalized.shape}"
            )
    else:
        if normalized.ndim != 4 or normalized.shape[-1] != 3:
            raise ValueError(
                f"Expected color frame batch for stream '{stream}', got shape {normalized.shape}"
            )

    return np.ascontiguousarray(normalized)


def decode_compressed_images(group: h5py.Group, stream: str, grayscale: bool) -> np.ndarray:
    """Decode JPEG-compressed images stored as data/offsets/lengths in an HDF5 group."""
    missing = [name for name in REQUIRED_COMPRESSED_KEYS if name not in group]
    if missing:
        raise ValueError(f"Compressed stream '{stream}' is missing keys: {missing}")

    raw_data = group["data"][:]
    data = validate_uint8_dtype(raw_data, stream)
    if data.ndim != 1:
        raise ValueError(f"Compressed byte buffer for stream '{stream}' must be a 1D array")

    offsets = np.asarray(group["offsets"][:], dtype=np.int64)
    lengths = np.asarray(group["lengths"][:], dtype=np.int64)

    if offsets.ndim != 1 or lengths.ndim != 1:
        raise ValueError(f"Offsets/lengths for stream '{stream}' must be 1D arrays")
    if len(offsets) != len(lengths):
        raise ValueError(
            f"Offsets/lengths length mismatch for stream '{stream}': {len(offsets)} != {len(lengths)}"
        )

    total_bytes = int(data.shape[0])
    decode_flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    frames = []

    for index, (offset, length) in enumerate(zip(offsets, lengths)):
        offset_int = int(offset)
        length_int = int(length)
        end = offset_int + length_int

        if offset_int < 0 or length_int < 0:
            raise ValueError(
                f"Negative offset/length at frame {index} for stream '{stream}': "
                f"offset={offset_int}, length={length_int}"
            )
        if end > total_bytes:
            raise ValueError(
                f"Out-of-bounds slice at frame {index} for stream '{stream}': "
                f"end={end}, total_bytes={total_bytes}"
            )

        jpg_bytes = np.ascontiguousarray(data[offset_int:end], dtype=np.uint8)
        frame = cv2.imdecode(jpg_bytes, decode_flag)
        if frame is None:
            raise ValueError(f"Failed to decode frame {index} for stream '{stream}'")
        frames.append(normalize_frame(frame, stream=stream, grayscale=grayscale))

    if not frames:
        raise ValueError(f"No compressed frames found for stream '{stream}'")

    return np.ascontiguousarray(np.stack(frames, axis=0))


def load_stream_frames(h5_path: str, stream: str) -> np.ndarray:
    grayscale = bool(STREAM_CONFIGS[stream]["grayscale"])
    with h5py.File(h5_path, "r") as h5_file:
        key = get_stream_key(stream)
        if key not in h5_file:
            raise KeyError(f"Stream '{stream}' not found in '{h5_path}'")

        node = h5_file[key]
        if isinstance(node, h5py.Dataset):
            return normalize_frames(node[:], stream=stream, grayscale=grayscale)
        if isinstance(node, h5py.Group):
            return decode_compressed_images(node, stream=stream, grayscale=grayscale)

        raise ValueError(f"Unsupported HDF5 node type for stream '{stream}': {type(node)}")


def extract_frames(h5_path: str, stream: str) -> tuple[np.ndarray, int, int]:
    """Load frames for a given stream. Returns (frames, h, w)."""
    frames = load_stream_frames(h5_path, stream)

    if frames.ndim == 4:
        h, w = frames.shape[1:3]
    elif frames.ndim == 3:
        h, w = frames.shape[1:3]
    else:
        raise ValueError(f"Unexpected frame shape {frames.shape} for stream '{stream}'")

    return frames, h, w


def frames_to_images(frames: np.ndarray, tmp_dir: str) -> list[str]:
    """Save frames as JPEG files in tmp_dir. Returns list of frame file paths."""
    paths = []
    for i in range(frames.shape[0]):
        frame = frames[i]
        path = os.path.join(tmp_dir, f"frame_{i:04d}.jpg")
        success = cv2.imwrite(path, frame)
        if not success:
            raise RuntimeError(f"Failed to write frame image: {path}")
        paths.append(path)
    return paths


def encode_video(frame_paths: list[str], output_path: str, fps: int, crf: int, pix_fmt: str) -> str:
    """Use ffmpeg to encode frames into a compressed MP4."""
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(os.path.dirname(frame_paths[0]), "frame_%04d.jpg"),
        "-c:v", "libx264",
        "-pix_fmt", pix_fmt,
        "-crf", str(crf),
        "-an",
        output_path,
    ]
    result = subprocess.run(
        ffmpeg_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")
    return output_path


def extract_single(
    h5_path: str,
    output_dir: str,
    streams: list[str],
    flat: bool = False,
    fps: int = 30,
    crf: int = 18,
    pix_fmt: str = "yuv420p",
    cleanup: bool = True,
):
    """Extract video streams from a single HDF5 file."""
    episode_name = Path(h5_path).stem

    if flat:
        ep_dir = Path(output_dir)
    else:
        ep_dir = Path(output_dir) / episode_name
    ep_dir.mkdir(parents=True, exist_ok=True)

    tmp_base = tempfile.mkdtemp(prefix="hdf5_extract_")
    try:
        for stream in streams:
            tmp_dir = os.path.join(tmp_base, stream)
            os.makedirs(tmp_dir, exist_ok=True)

            print(f"  [{stream}] Loading frames from {h5_path}...")
            frames, h, w = extract_frames(h5_path, stream)

            print(f"  [{stream}] Saving {frames.shape[0]} frames ({h}x{w})...")
            frame_paths = frames_to_images(frames, tmp_dir)

            out_path = ep_dir / f"{stream}.mp4"
            print(f"  [{stream}] Encoding to {out_path} (fps={fps}, crf={crf})...")
            encode_video(frame_paths, str(out_path), fps=fps, crf=crf, pix_fmt=pix_fmt)

            size_mb = os.path.getsize(out_path) / 1024 / 1024
            print(f"  [{stream}] Done: {out_path} ({size_mb:.1f} MB)")

            if cleanup:
                shutil.rmtree(tmp_dir, ignore_errors=True)
    finally:
        if cleanup:
            shutil.rmtree(tmp_base, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Extract videos from HDF5 episode files")
    parser.add_argument("input", help="HDF5 file or glob pattern (e.g. 'data/episode_*.hdf5')")
    parser.add_argument(
        "--streams",
        nargs="+",
        default=list(STREAM_CONFIGS.keys()),
        choices=list(STREAM_CONFIGS.keys()),
        help="Which streams to extract (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/vcl_umi/video",
        help="Output directory (default: data/vcl_umi/video)",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Flat layout: stream.mp4 instead of per-episode subdirectory",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--crf", type=int, default=18, help="H.264 CRF quality (lower=better, 18-28)")
    parser.add_argument("--no-cleanup", action="store_true", help="Keep temp frame files (for debugging)")
    parser.add_argument("--inspect", action="store_true", help="Inspect HDF5 structure and exit")
    args = parser.parse_args()

    paths = glob.glob(args.input)
    if not paths:
        print(f"No files match: {args.input}")
        sys.exit(1)
    paths = sorted(paths)

    print(f"Input pattern: {args.input}")
    print(f"Files found: {len(paths)}")
    print(f"Streams: {args.streams}")
    print(f"Output dir: {args.output_dir}")

    if args.inspect:
        for path in paths:
            inspect_hdf5(path)
        return

    for h5_path in paths:
        print(f"\n>>> Processing: {h5_path}")
        try:
            extract_single(
                h5_path=h5_path,
                output_dir=args.output_dir,
                streams=args.streams,
                flat=args.flat,
                fps=args.fps,
                crf=args.crf,
                cleanup=not args.no_cleanup,
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            import traceback

            traceback.print_exc()
            continue

    print(f"\nDone. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
