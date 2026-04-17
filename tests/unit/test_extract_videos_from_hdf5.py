import importlib.util
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "dataset" / "extract_videos_from_hdf5.py"
spec = importlib.util.spec_from_file_location("extract_videos_from_hdf5", MODULE_PATH)
extractor = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(extractor)


@pytest.fixture
def sample_frames():
    front = np.array(
        [
            [
                [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                [[9, 10, 11], [12, 13, 14], [15, 16, 17]],
            ],
            [
                [[18, 19, 20], [21, 22, 23], [24, 25, 26]],
                [[27, 28, 29], [30, 31, 32], [33, 34, 35]],
            ],
        ],
        dtype=np.uint8,
    )
    ir_left = np.array(
        [
            [[10, 20, 30], [40, 50, 60]],
            [[70, 80, 90], [100, 110, 120]],
        ],
        dtype=np.uint8,
    )
    ir_right = np.array(
        [
            [[5, 15, 25], [35, 45, 55]],
            [[65, 75, 85], [95, 105, 115]],
        ],
        dtype=np.uint8,
    )
    return {
        "front": front,
        "ir_left": ir_left,
        "ir_right": ir_right,
    }


def write_compressed_stream(group: h5py.Group, frames: np.ndarray) -> None:
    encoded_frames = []
    offsets = []
    lengths = []
    cursor = 0

    for frame in frames:
        ok, buffer = cv2.imencode(".png", frame)
        assert ok
        chunk = np.asarray(buffer, dtype=np.uint8).reshape(-1)
        encoded_frames.append(chunk)
        offsets.append(cursor)
        lengths.append(len(chunk))
        cursor += len(chunk)

    group.create_dataset("data", data=np.concatenate(encoded_frames).astype(np.uint8))
    group.create_dataset("offsets", data=np.array(offsets, dtype=np.int64))
    group.create_dataset("lengths", data=np.array(lengths, dtype=np.int32))


def create_direct_hdf5(path: Path, frames: dict[str, np.ndarray]) -> None:
    with h5py.File(path, "w") as h5_file:
        observations = h5_file.create_group("observations")
        images = observations.create_group("images")
        for stream, stream_frames in frames.items():
            images.create_dataset(stream, data=stream_frames)


def create_compressed_hdf5(path: Path, frames: dict[str, np.ndarray]) -> None:
    with h5py.File(path, "w") as h5_file:
        observations = h5_file.create_group("observations")
        images = observations.create_group("images")
        for stream, stream_frames in frames.items():
            stream_group = images.create_group(stream)
            write_compressed_stream(stream_group, stream_frames)


def test_load_stream_frames_reads_direct_datasets(tmp_path, sample_frames):
    h5_path = tmp_path / "direct.hdf5"
    create_direct_hdf5(h5_path, sample_frames)

    for stream, expected in sample_frames.items():
        actual = extractor.load_stream_frames(str(h5_path), stream)
        assert np.array_equal(actual, expected)
        assert actual.dtype == np.uint8


def test_load_stream_frames_reads_compressed_groups(tmp_path, sample_frames):
    h5_path = tmp_path / "compressed.hdf5"
    create_compressed_hdf5(h5_path, sample_frames)

    for stream, expected in sample_frames.items():
        actual = extractor.load_stream_frames(str(h5_path), stream)
        assert np.array_equal(actual, expected)
        assert actual.dtype == np.uint8


def test_inspect_hdf5_reports_compressed_streams(tmp_path, sample_frames):
    h5_path = tmp_path / "compressed_inspect.hdf5"
    create_compressed_hdf5(h5_path, sample_frames)

    stream_info = extractor.inspect_hdf5(str(h5_path))

    assert stream_info["front"]["storage"] == "compressed"
    assert stream_info["front"]["frames"] == sample_frames["front"].shape[0]
    assert stream_info["ir_left"]["storage"] == "compressed"


def test_decode_compressed_images_rejects_out_of_bounds_slices(tmp_path, sample_frames):
    h5_path = tmp_path / "broken.hdf5"
    create_compressed_hdf5(h5_path, sample_frames)

    with h5py.File(h5_path, "a") as h5_file:
        lengths = h5_file["observations/images/front/lengths"]
        lengths[0] = int(lengths[0]) + 10_000

    with pytest.raises(ValueError, match="Out-of-bounds slice"):
        extractor.load_stream_frames(str(h5_path), "front")


def test_load_stream_frames_rejects_missing_compressed_keys(tmp_path, sample_frames):
    h5_path = tmp_path / "missing_keys.hdf5"
    with h5py.File(h5_path, "w") as h5_file:
        observations = h5_file.create_group("observations")
        images = observations.create_group("images")
        front = images.create_group("front")
        front.create_dataset("data", data=np.array([1, 2, 3], dtype=np.uint8))
        front.create_dataset("offsets", data=np.array([0], dtype=np.int64))

    with pytest.raises(ValueError, match="missing keys"):
        extractor.load_stream_frames(str(h5_path), "front")


def test_load_stream_frames_rejects_non_uint8_direct_frames(tmp_path, sample_frames):
    h5_path = tmp_path / "non_uint8.hdf5"
    non_uint8_frames = dict(sample_frames)
    non_uint8_frames["front"] = sample_frames["front"].astype(np.uint16)
    create_direct_hdf5(h5_path, non_uint8_frames)

    with pytest.raises(ValueError, match="Unsupported dtype"):
        extractor.load_stream_frames(str(h5_path), "front")


def test_decode_compressed_images_rejects_negative_offsets(tmp_path, sample_frames):
    h5_path = tmp_path / "negative_offset.hdf5"
    create_compressed_hdf5(h5_path, sample_frames)

    with h5py.File(h5_path, "a") as h5_file:
        offsets = h5_file["observations/images/front/offsets"]
        offsets[0] = -1

    with pytest.raises(ValueError, match="Negative offset/length"):
        extractor.load_stream_frames(str(h5_path), "front")


def test_decode_compressed_images_rejects_non_1d_offsets(tmp_path, sample_frames):
    h5_path = tmp_path / "non_1d_offsets.hdf5"
    create_compressed_hdf5(h5_path, sample_frames)

    with h5py.File(h5_path, "a") as h5_file:
        offsets_path = "observations/images/front/offsets"
        offsets = h5_file[offsets_path][:]
        del h5_file[offsets_path]
        h5_file["observations/images/front"].create_dataset(offsets_path.split("/")[-1], data=offsets.reshape(1, -1))

    with pytest.raises(ValueError, match="must be 1D arrays"):
        extractor.load_stream_frames(str(h5_path), "front")


def test_decode_compressed_images_rejects_decode_failures(tmp_path, sample_frames):
    h5_path = tmp_path / "decode_failure.hdf5"
    create_compressed_hdf5(h5_path, sample_frames)

    with h5py.File(h5_path, "a") as h5_file:
        data = h5_file["observations/images/front/data"]
        data[:8] = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.uint8)

    with pytest.raises(ValueError, match="Failed to decode frame"):
        extractor.load_stream_frames(str(h5_path), "front")


def test_decode_compressed_images_rejects_non_uint8_byte_buffer(tmp_path, sample_frames):
    h5_path = tmp_path / "compressed_non_uint8.hdf5"
    create_compressed_hdf5(h5_path, sample_frames)

    with h5py.File(h5_path, "a") as h5_file:
        group = h5_file["observations/images/front"]
        data = group["data"][:].astype(np.uint16)
        del group["data"]
        group.create_dataset("data", data=data)

    with pytest.raises(ValueError, match="Unsupported dtype"):
        extractor.load_stream_frames(str(h5_path), "front")


def test_decode_compressed_images_rejects_non_1d_byte_buffer(tmp_path, sample_frames):
    h5_path = tmp_path / "compressed_non_1d.hdf5"
    create_compressed_hdf5(h5_path, sample_frames)

    with h5py.File(h5_path, "a") as h5_file:
        group = h5_file["observations/images/front"]
        data = group["data"][:]
        del group["data"]
        group.create_dataset("data", data=data.reshape(1, -1))

    with pytest.raises(ValueError, match="must be a 1D array"):
        extractor.load_stream_frames(str(h5_path), "front")
