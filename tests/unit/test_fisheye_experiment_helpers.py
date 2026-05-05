import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "inbox" / "fisheye_experiment_helpers.py"
spec = importlib.util.spec_from_file_location("fisheye_experiment_helpers", MODULE_PATH)
helpers = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(helpers)


def test_build_experiment_root_uses_input_name_method_and_reconstructor(tmp_path):
    actual = helpers.build_experiment_root(
        output_root=tmp_path,
        input_path="/nvme1/xinhai/projects/NeoVerse/data/vcl_umi/umi_0202/video/episode_10/front.mp4",
        method="crop",
        reconstructor="page4d",
    )

    assert actual == tmp_path / "front_mp4" / "crop" / "page4d"


def test_parse_crop_box_accepts_left_top_right_bottom():
    assert helpers.parse_crop_box("360,180,1560,900") == (360, 180, 1560, 900)


@pytest.mark.parametrize("value", ["360,180,1560", "360,180,100,100", "a,b,c,d"])
def test_parse_crop_box_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        helpers.parse_crop_box(value)


def test_compute_center_crop_box_matches_episode_10_baseline_crop():
    actual = helpers.compute_center_crop_box(
        frame_width=1920,
        frame_height=1080,
        width_ratio=0.625,
        height_ratio=2 / 3,
    )

    assert actual == (360, 180, 1560, 900)


def test_clamp_crop_box_stays_inside_frame_bounds():
    actual = helpers.clamp_crop_box(
        crop_box=(-20, 15, 250, 160),
        frame_width=200,
        frame_height=120,
    )

    assert actual == (0, 15, 200, 120)


def test_compute_non_black_crop_box_finds_valid_region():
    frame = np.zeros((6, 8, 3), dtype=np.uint8)
    frame[1:5, 2:7] = 255

    actual = helpers.compute_non_black_crop_box(frame, threshold=8)

    assert actual == (2, 1, 7, 5)


def test_compute_non_black_crop_box_falls_back_to_full_frame_when_empty():
    frame = np.zeros((5, 7, 3), dtype=np.uint8)

    actual = helpers.compute_non_black_crop_box(frame, threshold=8)

    assert actual == (0, 0, 7, 5)


def test_sample_source_frames_sorts_directory_inputs_naturally(tmp_path):
    for name, value in (("frame10.png", 10), ("frame2.png", 2), ("frame1.png", 1)):
        Image.fromarray(np.full((2, 2, 3), value, dtype=np.uint8)).save(tmp_path / name)

    frames = helpers.sample_source_frames(tmp_path, num_frames=3, static_scene=False)

    assert [int(np.asarray(frame)[0, 0, 0]) for frame in frames] == [1, 2, 10]
