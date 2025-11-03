# LaCoreV

## make_videomae_tubes.py

`make_videomae_tubes.py` builds a space‑time mutual‑kNN graph from the attention maps of a VideoMAE model and exports the result as a Blender‑compatible OBJ+MTL file or as an MP4 video with visual overlays.

### Features
- Loads a video, extracts frames, and processes them with a pretrained VideoMAE model.
- Computes mutual‑kNN edges in both spatial and temporal dimensions based on attention similarity.
- Writes a 3‑D tube representation (OBJ + MTL) that can be imported into Blender.
- Generates an MP4 video where spatial edges are drawn on the original frames.
- Supports custom model, layer, k‑nearest‑neighbour parameters, window size, and device selection.

### Installation

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124  # or CPU wheel
pip install transformers av opencv-python numpy
```

### Usage

#### OBJ output (single chunk)

```bash
python make_videomae_tubes.py \
    --video path/to/video.mp4 \
    --start 100 \
    --num_frames 16 \
    --out tubes.obj
```

- `--start` specifies the first frame index.
- `--num_frames` defines how many consecutive frames are processed.
- The script writes `tubes.obj` and `tubes.mtl`. Import them in Blender via **File → Import → Wavefront (.obj)**.

#### MP4 output (full video)

```bash
python make_videomae_tubes.py \
    --video path/to/video.mp4 \
    --out video_output.mp4
```

- The script processes the entire video in chunks, builds spatial edges, and overlays them on the original frames.
- The resulting MP4 shows orange edges (spatial) on top of the video with a grid overlay.

### Optional arguments

- `--model_id` – Hugging Face model identifier (default: `MCG-NJU/videomae-base`).
- `--layer` – Encoder layer index to extract attentions from (default: `6`).
- `--k_spatial` – Number of nearest neighbours for spatial edges (default: `8`).
- `--k_temporal` – Number of nearest neighbours for temporal edges (default: `4`).
- `--win` – Chebyshev window radius around a patch (default: `2`).
- `--frame_spacing` – Distance between frames in OBJ units (default: `6.0`).
- `--unit` – Size of a patch cell in OBJ units (default: `1.0`).
- `--device` – `cuda` or `cpu` (default: `cuda`).
- `--dtype` – Data type for the model (`float32`, `float16`, `bfloat16`; default: `float32`).

### Output description

- **OBJ + MTL**: Contains vertices for patch centers, grid lines, and edges. Colors:
  - Grid lines: gray.
  - Spatial edges: orange.
  - Temporal edges: blue.
- **MP4**: Shows the original video frames with a black grid, orange patch centers, and orange lines representing spatial edges.

### Notes

- The script uses the attention maps from the selected encoder layer, averaged across heads.
- For OBJ output, the generated file can be visualized in Blender; the Z‑axis corresponds to time.
- For MP4 output, only spatial edges are drawn; temporal edges are omitted for clarity.
