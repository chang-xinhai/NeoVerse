# AGENTS.md

## Outputs layout

Keep `outputs/` organized by artifact type:

```text
outputs/
├── bundles/
│   └── 4dgs/
├── demos/
│   ├── driving/
│   ├── robot/
│   ├── smoke/
│   └── trajectory/
├── tools/
└── trajectories/
```

Guidelines:
- Put rendered videos under `outputs/demos/...`
- Put exportable 4DGS bundles under `outputs/bundles/4dgs/...`
- Put cached external tooling under `outputs/tools/...`
- Keep `outputs/trajectories/` for trajectory-specific artifacts rather than general demos
- Do not leave new `.mp4` files or bundle directories at the root of `outputs/`

## 4DGS visualization handoff

4DGS visualization should stay headless and bundle-based.

Use `diffsynth/utils/gaussian_bundle.py:export_neoverse_4dgs_bundle` as the canonical export path.

Expected bundle contract:
- `bundle_manifest.json`
- `cameras.json`
- `timeline.json`
- `gaussians/animation_0001.ply`, `animation_0002.ply`, ...
- `metadata/...`

The bundle is designed for local visualization in SuperSplat:
- file pattern: `animation_{:04d}.ply`
- manifest advertises `export_format: "ply_sequence"`
- manifest advertises `super_splat_compatible: true`
- Neoverse reimport is currently unsupported

## Avoid

- Do not reintroduce Blender-specific packaging or `.blend` export flows
- Do not invent parallel 4DGS export layouts when `export_neoverse_4dgs_bundle` already fits
- Do not mix deliverable bundles with transient demo videos in the same folder
