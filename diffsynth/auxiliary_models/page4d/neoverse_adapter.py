from typing import Any

import torch

from diffsynth.models.utils import hash_state_dict_keys, load_state_dict

from .models.vggt import VGGT
from .utils.pose_enc import pose_encoding_to_extri_intri
from ..pseudo_gaussian_adapter import PseudoGaussianReconstructor


class _VGGTFamilyReconstructor(PseudoGaussianReconstructor):
    DEFAULT_IMAGE_SIZE = 518
    DEFAULT_PATCH_SIZE = 14
    DEFAULT_EMBED_DIM = 1024
    DEFAULT_GAUSSIAN_SCALE = 0.001

    def __init__(
        self,
        gaussian_scale: float = DEFAULT_GAUSSIAN_SCALE,
        enable_track: bool = False,
        mask_hold_start: int = 0,
        mask_hold_end: int = 0,
        **model_kwargs,
    ):
        super().__init__(gaussian_scale=gaussian_scale)
        self.model = self.build_model(
            enable_track=enable_track,
            mask_hold_start=mask_hold_start,
            mask_hold_end=mask_hold_end,
            **model_kwargs,
        )

    def build_model(self, **model_kwargs) -> VGGT:
        return VGGT(**model_kwargs)

    def _prepare_images(self, imgs: torch.Tensor) -> torch.Tensor:
        parameter = next(self.model.parameters())
        return imgs.to(device=parameter.device, dtype=parameter.dtype)

    def forward(self, views: dict[str, torch.Tensor], **_: Any) -> dict[str, torch.Tensor | list[list[object]]]:
        imgs = views["img"]
        prepared = self._prepare_images(imgs)
        predictions = self.model(prepared)

        pose_enc = predictions["pose_enc"]
        _, _, _, height, width = imgs.shape
        w2c, intrinsics = pose_encoding_to_extri_intri(pose_enc, image_size_hw=(height, width))
        c2w = torch.eye(4, device=w2c.device, dtype=w2c.dtype).view(1, 1, 4, 4).repeat(*w2c.shape[:2], 1, 1)
        c2w[..., :3, :] = w2c
        c2w = torch.linalg.inv(c2w)

        depth = predictions["depth"]
        depth_conf = predictions.get("depth_conf")
        return self.build_predictions(
            views=views,
            depth=depth,
            c2w=c2w,
            intrinsics=intrinsics,
            depth_conf=depth_conf,
        )

    def load_checkpoint(self, checkpoint_path: str, strict: bool = False):
        state_dict = load_state_dict(checkpoint_path, device="cpu")
        if "model" in state_dict and isinstance(state_dict["model"], dict):
            state_dict = state_dict["model"]
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=strict)
        return missing_keys, unexpected_keys

    @staticmethod
    def state_dict_converter():
        return VGGTFamilyModelDictConverter()


class PAGE4DReconstructor(_VGGTFamilyReconstructor):
    pass


class VGGTReconstructor(_VGGTFamilyReconstructor):
    pass


class VGGTFamilyModelDictConverter:
    def from_civitai(self, state_dict):
        if "model" in state_dict and isinstance(state_dict["model"], dict):
            config = {
                "strict_load": False,
                "checkpoint_has_model_key": True,
            }
            return state_dict, config

        hashed = hash_state_dict_keys(state_dict)
        config = {
            "strict_load": False,
        }
        if hashed:
            config["state_dict_hash"] = hashed
        return state_dict, config


def load_vggt_family_reconstructor(reconstructor_name: str, checkpoint_path: str, device: str | torch.device, torch_dtype: torch.dtype):
    reconstructor_cls = PAGE4DReconstructor if reconstructor_name == "page4d" else VGGTReconstructor
    model = reconstructor_cls()
    missing_keys, unexpected_keys = model.load_checkpoint(checkpoint_path, strict=False)
    if missing_keys:
        print(f"[{reconstructor_name}] Missing keys during load: {len(missing_keys)}")
    if unexpected_keys:
        print(f"[{reconstructor_name}] Unexpected keys during load: {len(unexpected_keys)}")
    model = model.to(device=device)
    return model
