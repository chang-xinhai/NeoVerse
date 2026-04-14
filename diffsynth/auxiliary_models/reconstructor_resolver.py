from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_RECONSTRUCTOR = "neoverse"
RECONSTRUCTOR_ALIASES = {
    "worldmirror": "neoverse",
}
RECONSTRUCTOR_DEFAULT_PATHS = {
    "neoverse": os.path.join("NeoVerse", "reconstructor.ckpt"),
    "da3": "da3_giant_1.1.safetensors",
    "page4d": os.path.join("PAGE4D", "checkpoint_nomask.pt"),
}
SUPPORTED_RECONSTRUCTORS = tuple(
    [DEFAULT_RECONSTRUCTOR, "da3", "page4d", *RECONSTRUCTOR_ALIASES.keys()]
)
DETECTABLE_RECONSTRUCTORS = {"neoverse", "da3", "page4d"}


@dataclass(frozen=True)
class ReconstructorSpec:
    name: str
    resolved_path: str
    used_custom_path: bool
    deprecated_path_only: bool


class ReconstructorResolutionError(ValueError):
    pass


def normalize_reconstructor_name(name: str | None) -> str:
    if not name:
        return DEFAULT_RECONSTRUCTOR
    normalized = name.lower()
    return RECONSTRUCTOR_ALIASES.get(normalized, normalized)


def resolve_reconstructor_spec(
    reconstructor: str | None,
    reconstructor_path: str | None,
    model_root: str = "models",
) -> ReconstructorSpec:
    normalized_name = normalize_reconstructor_name(reconstructor)
    deprecated_path_only = False

    if normalized_name not in RECONSTRUCTOR_DEFAULT_PATHS:
        raise ReconstructorResolutionError(
            f"Unsupported reconstructor '{reconstructor}'. Choose from: {', '.join(SUPPORTED_RECONSTRUCTORS)}"
        )

    if reconstructor_path and reconstructor is None:
        deprecated_path_only = True
        inferred_name = infer_reconstructor_name_from_path(reconstructor_path)
        if inferred_name is None:
            raise ReconstructorResolutionError(
                "Could not infer reconstructor type from --reconstructor_path. "
                "Please pass --reconstructor explicitly when using a custom checkpoint path."
            )
        return ReconstructorSpec(
            name=inferred_name,
            resolved_path=reconstructor_path,
            used_custom_path=True,
            deprecated_path_only=deprecated_path_only,
        )

    if reconstructor_path:
        return ReconstructorSpec(
            name=normalized_name,
            resolved_path=reconstructor_path,
            used_custom_path=True,
            deprecated_path_only=False,
        )

    relative_path = RECONSTRUCTOR_DEFAULT_PATHS[normalized_name]
    return ReconstructorSpec(
        name=normalized_name,
        resolved_path=os.path.join(model_root, relative_path),
        used_custom_path=False,
        deprecated_path_only=False,
    )


def infer_reconstructor_name_from_path(path: str) -> str | None:
    basename = os.path.basename(path).lower()
    lowered = path.lower()
    if "da3" in lowered or "depth-anything" in lowered:
        return "da3"
    if "page4d" in lowered or "checkpoint_nomask" in lowered:
        return "page4d"
    if basename == "reconstructor.ckpt" or "neoverse" in lowered or "worldmirror" in lowered:
        return "neoverse"
    return None


def format_missing_reconstructor_message(spec: ReconstructorSpec) -> str:
    return (
        f"Reconstructor checkpoint for '{spec.name}' was not found at '{spec.resolved_path}'. "
        f"Place the checkpoint there or override with --reconstructor_path."
    )
