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
import os
import sys
import glob
import argparse
import tempfile
import shutil
import subprocess
from pathlib import Path

import h5py
import cv2
import numpy as np


STREAM_CONFIGS = {
    "front":    {"fps": 30, "pix_fmt": "yuv420p", "crf": 18},
    "ir_left":  {"fps": 30, "pix_fmt": "yuv420p", "crf": 18},
    "ir_right": {"fps": 30, "pix_fmt": "yuv420p", "crf": 18},
}


def inspect_hdf5(path: str) -> dict:
    """Print HDF5 structure and return stream info."""
    with h5py.File(path, "r") as f:
        lines = []
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                shape = obj.shape if hasattr(obj, "shape") else ()
                dtype = obj.dtype if hasattr(obj, "dtype") else ""
                lines.append(f"  {name}: shape={shape}, dtype={dtype}")
            elif isinstance(obj, h5py.Group):
                lines.append(f"  {name}/")
        f.visititems(visitor)
        structure = "\n".join(lines)
        print(f"\n=== HDF5 structure: {path} ===")
        print(structure)
        print("=" * 60)

        stream_info = {}
        images_group = f.get("observations/images", {})
        for name in STREAM_CONFIGS:
            if name in images_group:
                ds = images_group[name]
                stream_info[name] = {
                    "shape": ds.shape,
                    "dtype": ds.dtype,
                }
        return stream_info


def extract_frames(h5_path: str, stream: str) -> tuple:
    """Load frames for a given stream. Returns (frames, h, w)."""
    with h5py.File(h5_path, "r") as f:
        key = f"observations/images/{stream}"
        if key not in f:
            raise KeyError(f"Stream '{stream}' not found in '{h5_path}'")
        frames = f[key][:]

    if frames.ndim == 4:  # HWC (front RGB)
        h, w = frames.shape[1:3]
    elif frames.ndim == 3:  # HW grayscale (IR cameras)
        h, w = frames.shape[1:3]
    else:
        raise ValueError(f"Unexpected frame shape {frames.shape} for stream '{stream}'")

    return frames, h, w


def frames_to_images(frames: np.ndarray, tmp_dir: str) -> list:
    """Save frames as JPEG files in tmp_dir. Returns list of frame file paths."""
    paths = []
    for i in range(frames.shape[0]):
        # HDF5 stores BGR uint8 (cv2.imdecode returns BGR); write directly.
        frame = frames[i]
        path = os.path.join(tmp_dir, f"frame_{i:04d}.jpg")
        cv2.imwrite(path, frame)
        paths.append(path)
    return paths


def encode_video(frame_paths: list, output_path: str, fps: int, crf: int, pix_fmt: str):
    """Use ffmpeg to encode frames into a compressed MP4."""
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(os.path.dirname(frame_paths[0]), "frame_%04d.jpg"),
        "-c:v", "libx264",
        "-pix_fmt", pix_fmt,
        "-crf", str(crf),
        "-an",  # no audio
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
    episode_name = Path(h5_path).stem  # e.g. "episode_10"

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

            out_name = f"{stream}.mp4"
            out_path = ep_dir / out_name

            print(f"  [{stream}] Encoding to {out_path} (fps={fps}, crf={crf})...")
            encode_video(frame_paths, str(out_path), fps=fps, crf=crf, pix_fmt=pix_fmt)

            size_mb = os.path.getsize(out_path) / 1024 / 1024
            print(f"  [{stream}] Done: {out_path} ({size_mb:.1f} MB)")

            shutil.rmtree(tmp_dir, ignore_errors=True)
    finally:
        if cleanup:
            shutil.rmtree(tmp_base, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Extract videos from HDF5 episode files")
    parser.add_argument("input", help="HDF5 file or glob pattern (e.g. 'data/episode_*.hdf5')")
    parser.add_argument("--streams", nargs="+",
                        default=list(STREAM_CONFIGS.keys()),
                        choices=list(STREAM_CONFIGS.keys()),
                        help="Which streams to extract (default: all)")
    parser.add_argument("--output-dir", default="data/vcl_umi/video",
                        help="Output directory (default: data/vcl_umi/video)")
    parser.add_argument("--flat", action="store_true",
                        help="Flat layout: stream.mp4 instead of per-episode subdirectory")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--crf", type=int, default=18, help="H.264 CRF quality (lower=better, 18-28)")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Keep temp frame files (for debugging)")
    parser.add_argument("--inspect", action="store_true",
                        help="Inspect HDF5 structure and exit")
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
        for p in paths:
            inspect_hdf5(p)
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
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nDone. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
