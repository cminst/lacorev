"""
Build a space–time mutual‑kNN graph from VideoMAE attentions and export a Blender‑compatible OBJ+MTL or MP4 video.

Requirements:
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124  # or CPU wheel
  pip install transformers av opencv-python numpy

Usage for OBJ (single chunk):
  python make_videomae_tubes_obj.py \
      --video path/to/video.mp4 \
      --start 100 \
      --out tubes.obj

Usage for MP4 (entire video):
  python make_videomae_tubes_obj.py \
      --video path/to/video.mp4 \
      --out video_output.mp4

Optional:
  --model_id MCG-NJU/videomae-base --layer 6 --k_spatial 8 --k_temporal 4 --win 2
  --frame_spacing 6.0 --unit 1.0 --device cuda --dtype float32
Notes:
  • Uses attentions from a chosen encoder layer, averaged across heads.
  • For OBJ output: Orange = spatial edges, Blue = temporal edges, Grey = patch grid lines.
  • For MP4 output: Shows spatial edges (orange) on original video frames with a grid overlay.
  • OBJ+MTL can be imported in Blender (File → Import → Wavefront (.obj)).
"""

import argparse
import os
import sys
from pathlib import Path
import logging
import numpy as np

# --- Video decoding and processing ---

logger = logging.getLogger(__name__)

def get_video_properties(video_path):
    """Get video properties using OpenCV."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, width, height, total_frames

def read_frames_chunk(video_path, start, num, resize_to=None):
    """
    Returns a list of RGB frames (H,W,3) uint8 for indices [start, start+num).
    If resize_to is not None, it should be (H, W).
    """
    frames = []
    try:
        import av  # PyAV (recommended for accurate seeking)
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        container.seek(int((start / stream.average_rate) * stream.time_base.denominator),
                       any_frame=False, backward=True, stream=stream)
        idx = -1
        for packet in container.demux(stream):
            for frame in packet.decode():
                idx += 1
                if idx < start:
                    continue
                if idx >= start + num:
                    break
                frames.append(frame.to_ndarray(format="rgb24"))
            if idx >= start + num:
                break
        container.close()
    except Exception:
        # Fallback to OpenCV
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _ in range(num):
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
    if not frames:
        logger.warn(f"Warning: No frames decoded for start index {start}.")
    if resize_to is not None:
        frames = [cv2_resize_safe(f, resize_to) for f in frames]
    return frames

def cv2_resize_safe(img, size_hw):
    import cv2
    H, W = size_hw
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

# --- Model and processor ---
def load_model_and_processor(model_id, device="cpu", dtype_str="float32"):
    import torch
    from transformers import AutoImageProcessor, VideoMAEModel
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_str]
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = VideoMAEModel.from_pretrained(model_id, attn_implementation="eager")
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, processor

# --- Edge building ---
def build_edges_from_attn(attn_L, T_frames, H_p, W_p, tubelet_size, k_spatial=8, k_temporal=4, win=2):
    P = H_p * W_p
    T_tokens = T_frames // tubelet_size
    L_total = attn_L.shape[0]
    has_cls = (L_total == (T_tokens * P) + 1)
    offset = 1 if has_cls else 0

    def idx(tk, r, c):
        return offset + tk * P + r * W_p + c

    A = (attn_L + attn_L.T) * 0.5
    A_np = A.detach().cpu().numpy() if hasattr(A, "detach") else np.asarray(A)

    spatial_pairs = set()
    for tk in range(T_tokens):
        for r in range(H_p):
            for c in range(W_p):
                i = idx(tk, r, c)
                r0, r1 = max(0, r - win), min(H_p - 1, r + win)
                c0, c1 = max(0, c - win), min(W_p - 1, c + win)
                cand = [idx(tk, rr, cc) for rr in range(r0, r1 + 1) for cc in range(c0, c1 + 1) if (rr, cc) != (r, c)]
                if not cand: continue

                scores = A_np[i, cand]
                k = min(k_spatial, len(cand))
                top_idx = np.argpartition(-scores, k - 1)[:k]
                nbrs = {cand[j] for j in top_idx}

                for j in nbrs:
                    j_rel = j - offset - tk * P
                    rr, cc = divmod(j_rel, W_p)
                    r0j, r1j = max(0, rr - win), min(H_p - 1, rr + win)
                    c0j, c1j = max(0, cc - win), min(W_p - 1, cc + win)
                    cand_j = [idx(tk, rjj, cjj) for rjj in range(r0j, r1j + 1) for cjj in range(c0j, c1j + 1) if (rjj, cjj) != (rr, cc)]
                    if not cand_j: continue

                    scores_j = A_np[j, cand_j]
                    kj = min(k_spatial, len(cand_j))
                    top_j = {cand_j[u] for u in np.argpartition(-scores_j, kj - 1)[:kj]}
                    if i in top_j:
                        spatial_pairs.add(tuple(sorted((i, j))))

    temporal_pairs = set()
    for tk in range(T_tokens - 1):
        for r in range(H_p):
            for c in range(W_p):
                i = idx(tk, r, c)
                r0, r1 = max(0, r - win), min(H_p - 1, r + win)
                c0, c1 = max(0, c - win), min(W_p - 1, c + win)
                cand = [idx(tk + 1, rr, cc) for rr in range(r0, r1 + 1) for cc in range(c0, c1 + 1)]
                if not cand: continue

                scores = A_np[i, cand]
                k = min(k_temporal, len(cand))
                top_idx = np.argpartition(-scores, k - 1)[:k]
                nbrs = {cand[j] for j in top_idx}

                for j in nbrs:
                    j_rel = j - offset - (tk + 1) * P
                    rr, cc = divmod(j_rel, W_p)
                    r0j, r1j = max(0, rr - win), min(H_p - 1, rr + win)
                    c0j, c1j = max(0, cc - win), min(W_p - 1, cc + win)
                    cand_j = [idx(tk, rjj, cjj) for rjj in range(r0j, r1j + 1) for cjj in range(c0j, c1j + 1)]
                    if not cand_j: continue

                    scores_j = A_np[j, cand_j]
                    kj = min(k_temporal, len(cand_j))
                    top_j = {cand_j[u] for u in np.argpartition(-scores_j, kj - 1)[:kj]}
                    if i in top_j:
                        temporal_pairs.add(tuple(sorted((i, j))))

    spatial_edges, temporal_edges = [], []
    for (i_idx, j_idx) in spatial_pairs:
        tk_i, pos_i = divmod(i_idx - offset, P); ri, ci = divmod(pos_i, W_p)
        tk_j, pos_j = divmod(j_idx - offset, P); rj, cj = divmod(pos_j, W_p)
        f_start = tk_i * tubelet_size
        for df in range(tubelet_size):
            f_raw = f_start + df
            if f_raw < T_frames:
                spatial_edges.append((f_raw, ri, ci, f_raw, rj, cj))

    for (i_idx, j_idx) in temporal_pairs:
        tk_i, pos_i = divmod(i_idx - offset, P); ri, ci = divmod(pos_i, W_p)
        tk_j, pos_j = divmod(j_idx - offset, P); rj, cj = divmod(pos_j, W_p)
        f0, f1 = tk_i * tubelet_size, tk_j * tubelet_size
        if f0 < T_frames and f1 < T_frames:
            temporal_edges.append((f0, ri, ci, f1, rj, cj))

    return spatial_edges, temporal_edges, has_cls

# --- OBJ/MTL writer ---
def write_obj_with_edges(out_path, H_p, W_p, frames_rgb, spatial_edges, temporal_edges, frame_spacing=6.0, unit=1.0):
    # This function remains unchanged, as it was correct.
    out_path = Path(out_path)
    # ... (rest of the function is identical to your original correct version)
    out_mtl = out_path.with_suffix(".mtl")

    verts_centers = {}
    vertices = []
    grid_lines = []
    vcount = 0

    def add_vertex(x, y, z=0.0):
        nonlocal vcount
        vertices.append((x, y, z))
        vcount += 1
        return vcount

    frame_w = W_p * unit
    frame_h = H_p * unit

    for f_idx in range(len(frames_rgb)):
        z0 = f_idx * frame_spacing
        for r in range(H_p):
            for c in range(W_p):
                x = (c + 0.5) * unit
                y = (H_p - 1 - r + 0.5) * unit
                idx_v = add_vertex(x, y, z0)
                verts_centers[(f_idx, r, c)] = idx_v
        for c in range(W_p + 1):
            x = c * unit
            v1 = add_vertex(x, 0, z0); v2 = add_vertex(x, frame_h, z0)
            grid_lines.append((v1, v2))
        for r in range(H_p + 1):
            y = r * unit
            v1 = add_vertex(0, y, z0); v2 = add_vertex(frame_w, y, z0)
            grid_lines.append((v1, v2))

    spatial_lines = [(verts_centers[(f, r1, c1)], verts_centers[(f, r2, c2)]) for (f, r1, c1, f2, r2, c2) in spatial_edges]
    temporal_lines = [(verts_centers[(f0, r1, c1)], verts_centers[(f1, r2, c2)]) for (f0, r1, c1, f1, r2, c2) in temporal_edges]

    with open(out_mtl, "w", encoding="utf-8") as m:
        m.write("# Materials for tubes OBJ\n")
        m.write("newmtl Grid\nKd 0.3 0.3 0.3\nKa 0.0 0.0 0.0\nd 1.0\nillum 1\n\n")
        m.write("newmtl Spatial\nKd 1.000 0.500 0.000\nKa 0.0 0.0 0.0\nd 1.0\nillum 1\n\n")
        m.write("newmtl Temporal\nKd 0.000 0.450 1.000\nKa 0.0 0.0 0.0\nd 1.0\nillum 1\n\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Space–time mutual-kNN visualization\nmtllib {out_mtl.name}\n")
        for v in vertices: f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("o grid_dividers\nusemtl Grid\n")
        for a, b in grid_lines: f.write(f"l {a} {b}\n")
        f.write("o spatial_edges\nusemtl Spatial\n")
        for a, b in spatial_lines: f.write(f"l {a} {b}\n")
        f.write("o temporal_edges\nusemtl Temporal\n")
        for a, b in temporal_lines: f.write(f"l {a} {b}\n")

    return str(out_path), str(out_mtl)

# --- MP4 generator ---
def generate_video_output(out_path, input_video_path, all_spatial_edges,
                          original_width, original_height, patch_size, image_size, fps):
    """
    Renders the output video by reading frames from the original video,
    drawing overlays, and writing to the new output path.
    """
    import cv2

    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video for writing: {input_video_path}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (original_width, original_height))

    H_p = W_p = image_size // patch_size
    patch_width_orig = original_width / W_p
    patch_height_orig = original_height / H_p

    # For faster lookups, group edges by frame
    edges_by_frame = {}
    for edge in all_spatial_edges:
        frame_idx = edge[0]
        if frame_idx not in edges_by_frame:
            edges_by_frame[frame_idx] = []
        edges_by_frame[frame_idx].append(edge)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw grid lines (thin black)
        for c in range(W_p + 1):
            x = int(c * patch_width_orig)
            cv2.line(frame, (x, 0), (x, original_height), (0, 0, 0), 1)
        for r in range(H_p + 1):
            y = int(r * patch_height_orig)
            cv2.line(frame, (0, y), (original_width, y), (0, 0, 0), 1)

        # Draw patch centers and spatial edges if they exist for this frame
        if frame_idx in edges_by_frame:
            for r in range(H_p):
                for c in range(W_p):
                    center_x = int((c + 0.5) * patch_width_orig)
                    center_y = int((r + 0.5) * patch_height_orig)
                    cv2.circle(frame, (center_x, center_y), 3, (0, 165, 255), -1)

            for (f, r1, c1, f2, r2, c2) in edges_by_frame[frame_idx]:
                x1 = int((c1 + 0.5) * patch_width_orig)
                y1 = int((r1 + 0.5) * patch_height_orig)
                x2 = int((c2 + 0.5) * patch_width_orig)
                y2 = int((r2 + 0.5) * patch_height_orig)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 165, 255), 4)

        video_writer.write(frame)
        frame_idx += 1

    cap.release()
    video_writer.release()
    return str(out_path)

def main():
    parser = argparse.ArgumentParser(description="VideoMAE mutual‑kNN → Blender OBJ or MP4")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--start", type=int, default=0, help="Start frame index (for .obj output)")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames (for .obj output)")
    parser.add_argument("--model_id", type=str, default="MCG-NJU/videomae-base", help="HF model id")
    parser.add_argument("--layer", type=int, default=6, help="Encoder layer index to read attentions from (0-based)")
    parser.add_argument("--k_spatial", type=int, default=8, help="Mutual‑kNN k for spatial edges")
    parser.add_argument("--k_temporal", type=int, default=4, help="Mutual‑kNN k for temporal edges")
    parser.add_argument("--win", type=int, default=2, help="Chebyshev window radius around a patch")
    parser.add_argument("--frame_spacing", type=float, default=6.0, help="Spacing between frames in OBJ units")
    parser.add_argument("--unit", type=float, default=1.0, help="Patch cell size in OBJ units")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--out", type=str, required=True, help="Output path (.obj or .mp4)")
    args = parser.parse_args()

    # Load model and processor
    model, processor = load_model_and_processor(args.model_id, device=args.device, dtype_str=args.dtype)
    image_size = getattr(model.config, "image_size", 224)
    patch_size = model.config.patch_size
    tubelet_size = model.config.tubelet_size
    H_p = W_p = image_size // patch_size

    if args.out.lower().endswith('.mp4'):
        fps, orig_w, orig_h, total_frames = get_video_properties(args.video)
        chunk_size = 16
        all_spatial_edges = []

        num_chunks = total_frames // chunk_size
        logger.info(f"Video has {total_frames} frames, processing in {num_chunks} chunks of {chunk_size}...")

        for i, chunk_start in enumerate(range(0, total_frames, chunk_size)):
            num_in_chunk = min(chunk_size, total_frames - chunk_start)
            if num_in_chunk < chunk_size:
                logger.info(f"Skipping final incomplete chunk of {num_in_chunk} frames.")
                continue

            logger.info(f"Processing chunk {i + 1}/{num_chunks} (frames {chunk_start}-{chunk_start + num_in_chunk - 1})...")
            frames = read_frames_chunk(args.video, chunk_start, num_in_chunk, resize_to=(image_size, image_size))
            if not frames: continue

            pixel_values = processor(frames, return_tensors="pt").pixel_values.to(args.device)

            import torch
            with torch.no_grad():
                outputs = model(pixel_values, output_attentions=True)

            attn = outputs.attentions[args.layer][0].mean(dim=0)

            spatial, _, _ = build_edges_from_attn(
                attn, T_frames=num_in_chunk, H_p=H_p, W_p=W_p, tubelet_size=tubelet_size,
                k_spatial=args.k_spatial, k_temporal=args.k_temporal, win=args.win
            )
            all_spatial_edges.extend([(f + chunk_start, r1, c1, f2, r2, c2) for f, r1, c1, f2, r2, c2 in spatial])

        logger.info("All chunks processed. Rendering final video...")
        video_path = generate_video_output(
            args.out, args.video, all_spatial_edges,
            orig_w, orig_h, patch_size, image_size, fps)
        logger.info(f"[OK] Wrote video: {video_path}")

    else: # OBJ Output
        frames = read_frames_chunk(args.video, args.start, args.num_frames, resize_to=(image_size, image_size))
        pixel_values = processor(frames, return_tensors="pt").pixel_values.to(args.device)

        import torch
        with torch.no_grad():
            outputs = model(pixel_values, output_attentions=True)

        attn = outputs.attentions[args.layer][0].mean(dim=0)

        spatial, temporal, _ = build_edges_from_attn(
            attn, T_frames=args.num_frames, H_p=H_p, W_p=W_p, tubelet_size=tubelet_size,
            k_spatial=args.k_spatial, k_temporal=args.k_temporal, win=args.win)

        obj_path, mtl_path = write_obj_with_edges(
            args.out, H_p, W_p, frames_rgb=frames,
            spatial_edges=spatial, temporal_edges=temporal,
            frame_spacing=args.frame_spacing, unit=args.unit)
        logger.info(f"[OK] Wrote: {obj_path}")
        logger.info(f"[OK] Wrote: {mtl_path}")
        logger.info("Import into Blender: File → Import → Wavefront (.obj)")

if __name__ == "__main__":
    main()
