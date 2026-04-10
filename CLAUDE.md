# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

NeoVerse is the base repository for a data-generation system built around monocular-video-driven 4D scene understanding and rendering.

The current repository already supports:
- reconstructing scene structure from monocular video or images,
- designing or loading camera trajectories,
- rendering target views from the reconstructed representation,
- refining the rendered sequence with a WAN-based video generation model.

The intended direction for this fork is to evolve NeoVerse toward a complete editable/exportable 4DGS workflow:
- given raw monocular video, optionally with depth or camera pose information, produce a complete 4DGS intermediate representation,
- keep that intermediate exportable and editable,
- render it under designed camera trajectories,
- use a video generation model for refinement and high-quality output.

The downstream goal is robot-manipulation data augmentation from reference videos, including:
- high-quality novel-view synthesis,
- editable scene interventions such as replacing backgrounds,
- editable object interventions such as replacing manipulated objects,
- scaling data generation for large embodied-intelligence / robot-learning training pipelines.

Expected future work in this repo includes:
- 4DGS export and visualization,
- 4DGS editing,
- camera trajectory generation,
- improving or replacing reconstruction components with PAGE4D-style alternatives where appropriate,
- dataset loading and preprocessing scripts for data-generation workflows.

Unless there is a strong reason otherwise, extend the existing NeoVerse pipeline structure instead of building parallel ad hoc flows.

## Common commands

- Environment: use the `neoverse` conda environment.
- CLI inference:
  ```bash
  python inference.py --input_path examples/videos/robot.mp4 --trajectory tilt_up --output_path outputs/tilt_up.mp4
  ```
- Trajectory validation:
  ```bash
  python inference.py --trajectory_file examples/trajectories/orbit_left_pull_out.json --validate_only
  ```
- Gradio app:
  ```bash
  python app.py
  python app.py --low_vram
  ```

There is no repository-level automated test command documented today. Before commit, use trajectory validation, a minimal CLI inference run, and app startup as smoke tests.

## Machine and runtime guidance

Current machine snapshot:
- CPU: 2x Intel Xeon Platinum 8374C @ 2.70GHz, 144 logical CPUs total.
- GPU: 5x NVIDIA L40, each with ~46 GB VRAM.

When running experiments, prefer `--low_vram` and run on a single idle GPU rather than occupying multiple cards. Check GPU availability first and choose a mostly free device before starting long runs.

## High-level architecture

- `inference.py`: CLI path for loading inputs, building `CameraTrajectory`, reconstructing the scene, rendering target views, and generating output video.
- `app.py`: Gradio workflow for upload -> reconstruct -> trajectory design/upload -> preview -> generate.
- `diffsynth/pipelines/wan_video_neoverse.py`: core NeoVerse pipeline; `from_pretrained(...)` wires reconstructor, WAN diffusion models, LoRA, tokenizer, and low-VRAM behavior.
- `diffsynth/utils/auxiliary.py`: shared input loading and trajectory logic; `CameraTrajectory` is the canonical trajectory abstraction.
- `diffsynth/utils/__init__.py`: preview/export helpers for point clouds and GLB scene building.
- `diffsynth/models/model_manager.py`: generic model loading/detection layer.
- `docs/trajectory_format.md` and `docs/coordinate_system.md`: source of truth for trajectory schema and camera conventions.

Big picture: reconstruct scene structure from monocular input, then render target views and refine them with the WAN-based video model.

## Working conventions

- Read relevant files before proposing edits or explanations.
- Do not implement changes unless the user clearly asked for changes.
- Prefer the smallest change that solves the problem.
- Avoid speculative abstractions and unrelated cleanup.
- Prefer editing existing files over creating new ones.
- Use parallel tool calls for independent reads/searches when possible.
- Verify the result before finishing.
- Do only what was asked.

Project-specific workflow rules:
- Follow Conventional Commits.
- Keep commits focused on one change set.
- Run smoke tests before commit.
- Do not commit a broken pipeline state.
- Reuse existing abstractions like `CameraTrajectory` and the existing pipeline/model-loading layers instead of bypassing them.

## Code style notes

- Follow the existing code style in the touched area.
- Keep changes surgical; avoid cleanup unrelated to the requested task.
- Reuse existing trajectory, pipeline, and model-loading abstractions before introducing new ones.
- If working on trajectory features, align with `CameraTrajectory`, `docs/trajectory_format.md`, and `docs/coordinate_system.md` rather than inventing a parallel format.
- If working on memory-sensitive inference paths, preserve low-VRAM execution behavior.

## Memory and context management

Before starting substantial work, check whether any repo-local Claude context files exist and use them if present. These are optional overlays, not guaranteed files:
- `CLAUDE-activeContext.md`
- `CLAUDE-patterns.md`
- `CLAUDE-decisions.md`
- `CLAUDE-troubleshooting.md`
- `CLAUDE-config-variables.md`
- `CLAUDE-temp.md` (only when directly relevant)

If future work updates durable workflow conventions or stable project context, keep the repository’s Claude context files aligned.

Do not accidentally commit temporary scratch files.

## Repository state reminder

At session start, the working tree already contained local modifications in:
- `.gitignore`
- `diffsynth/utils/__init__.py`
- `diffsynth/vram_management/layers.py`

Avoid overwriting unrelated user changes in those files.