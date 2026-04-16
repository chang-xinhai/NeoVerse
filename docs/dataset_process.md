# Dataset Processing

This directory covers tools and workflows for processing raw episode HDF5 files into usable formats.

## extract_videos_from_hdf5

**Script:** `scripts/dataset/extract_videos_from_hdf5.py`

Extract standard H.264 MP4 videos from episode HDF5 files.

### HDF5 Structure

Each episode file contains three video streams:

```
episode_*.hdf5
├── action:                    (N, 7) float64
├── observations/
│   ├── images/
│   │   ├── front:             (N, 1080, 1920, 3) uint8  ← RGB camera (BGR)
│   │   ├── ir_left:            (N, 480, 848) uint8        ← Left IR camera
│   │   └── ir_right:           (N, 480, 848) uint8        ← Right IR camera
│   └── qpos:                  (N, 7) float64
```

- `N` — number of frames
- `front` stores BGR uint8 (same as `cv2.imdecode` output)
- `ir_left` / `ir_right` are grayscale IR camera frames

### Usage

```bash
# Single file, all streams
python scripts/dataset/extract_videos_from_hdf5.py \
    data/vcl_umi/umi_0202/episode_10.hdf5

# Specific streams
python scripts/dataset/extract_videos_from_hdf5.py \
    data/vcl_umi/umi_0202/episode_10.hdf5 \
    --streams front

# Batch: glob pattern
python scripts/dataset/extract_videos_from_hdf5.py \
    "data/vcl_umi/umi_0202/episode_*.hdf5" \
    --output-dir data/vcl_umi/umi_0202/video/

# Inspect HDF5 structure without extracting
python scripts/dataset/extract_videos_from_hdf5.py --inspect data/vcl_umi/umi_0202/episode_10.hdf5
```

### Output Layout

Per-episode subdirectory (default):

```
data/vcl_umi/umi_0202/video/episode_10/
├── front.mp4      (H.264, 1080×1920, 30fps)
├── ir_left.mp4    (H.264, 480×848, 30fps)
└── ir_right.mp4   (H.264, 480×848, 30fps)
```

Flat layout (`--flat`):

```
data/vcl_umi/umi_0202/video_flat/
├── episode_10_front.mp4
├── episode_10_ir_left.mp4
└── episode_10_ir_right.mp4
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--streams` | all | Which streams to extract |
| `--output-dir` | `data/vcl_umi/video` | Output base directory |
| `--flat` | off | Flat layout instead of per-episode subdirectory |
| `--fps` | 30 | Frame rate |
| `--crf` | 18 | H.264 CRF quality (18=high quality, 28=smaller file) |
| `--no-cleanup` | off | Keep temp JPEG frames (for debugging) |
| `--inspect` | off | Print HDF5 structure and exit |
