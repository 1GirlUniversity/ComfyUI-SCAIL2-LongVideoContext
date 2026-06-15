# ComfyUI-SCAIL2-LongVideoContext

Smart long-video context window nodes for SCAIL-2 workflows in ComfyUI.

This custom node pack adds two model patch nodes:

- **SCAIL-2 Smart Long Video Context**
- **SCAIL-2 Long Video Context**

It is designed for SCAIL-2 / Wan long-video workflows that need temporal context windows, overlap handling, and 4n+1 frame alignment.

## What it does

The node patches the incoming `MODEL` with ComfyUI's `comfy.context_windows.IndexListContextHandler`. It automatically converts pixel frame counts to latent context lengths, clamps frame counts to the `4n+1` pattern, and can preserve the reference condition index through `cond_retain_index_list`.

It does **not** replace the official SCAIL-2 nodes such as `WanSCAILToVideo`, `SCAIL2ColoredMask`, `replacement_mode`, `previous_frames`, or `video_frame_offset`. It is an additional model-context patch that can be used before sampling.

## Installation

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/YOUR_GITHUB_USERNAME/ComfyUI-SCAIL2-LongVideoContext.git
```

Restart ComfyUI.

## Requirements

- Recent ComfyUI build with `comfy.context_windows` support.
- SCAIL-2-capable ComfyUI core if you are using the official SCAIL-2 workflow.
- No extra Python package is required by this node pack.

## Nodes

### SCAIL-2 Smart Long Video Context

Automatic mode. Inputs:

- `model`
- `total_frames`
- `strategy`
- `manual_segment_len`
- `manual_segment_overlap`
- `context_schedule`
- `context_stride`
- `fuse_method`
- `freenoise`
- `causal_window_fix`
- `bypass_when_single_pass`
- `cond_retain_index_list`

Strategies:

- `auto_smooth_rtx6000`
- `auto_safe`
- `auto_quality_rtx6000`
- `force_official_81_5`
- `manual`

### SCAIL-2 Long Video Context

Manual mode. Inputs:

- `model`
- `segment_len`
- `segment_overlap`
- `context_schedule`
- `context_stride`
- `fuse_method`
- `freenoise`
- `causal_window_fix`
- `cond_retain_index_list`

## Recommended workflow placement

Place the node after the diffusion model loader and before the sampler:

```text
Diffusion Model Loader
        ↓
SCAIL-2 Smart Long Video Context
        ↓
Sampler
```

For `total_frames`, use the same effective frame count that goes into `WanSCAILToVideo length`, preferably adjusted to `4n+1`:

```text
frame_count -> a - ((a - 1) % 4) -> total_frames / length
```

## Initial settings

Good starting point for high-VRAM NVIDIA GPUs:

```text
strategy: auto_smooth_rtx6000
causal_window_fix: true
freenoise: false
cond_retain_index_list: 0
```

If you get OOM or excessive slowdown:

```text
strategy: auto_safe
```

If you want conservative official-style chunking:

```text
strategy: force_official_81_5
```

## License

GPL-3.0
