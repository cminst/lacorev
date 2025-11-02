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
  • Orange = spatial edges, Blue = temporal edges, Grey = patch grid lines.
  • Frames are arranged in a row (front to back) rather than side-by-side.
  • OBJ+MTL can be imported in Blender (File → Import → Wavefront (.obj)).
  • MP4 shows spatial edges on original video frames with grid overlay.
"""

import argparse
import os
import sys
from pathlib import Path
import torch
from transformers import AutoImageProcessor, VideoMAEModel
import numpy as np

# --- Video decoding and processing ---
def get_video_properties(video_path):
    """Get video properties (fps, dimensions, total frames)"""
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, width, height, total_frames

def read_frames_chunk(video_path, start, num, resize_to=None):
    """Read a chunk of frames from video"""
    frames = []
    try:
        import av
        container = av.open(video_path)
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
                    container.close()
                    if resize_to is not None:
                        frames = [cv2_resize_safe(f.to_ndarray(format="rgb24"), resize_to) for f in frames]
                    return frames
                frames.append(frame.to_ndarray(format="rgb24"))
        container.close()
    except Exception:
        import cv2
        cap = cv2.VideoCapture(video_path)
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
        raise RuntimeError("No frames decoded. Check start index and video.")
    if resize_to is not None:
        frames = [cv2_resize_safe(f, resize_to) for f in frames]
    return frames

def cv2_resize_safe(img, size_hw):
    import cv2
    H, W = size_hw
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

# --- Model and processor ---
def load_model_and_processor(model_id, device="cpu", dtype_str="float32"):
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
    temporal_pairs = set()

    # Spatial edges within each time token
    for tk in range(T_tokens):
        for r in range(H_p):
            for c in range(W_p):
                i = idx(tk, r, c)
                r0, r1 = max(0, r - win), min(H_p - 1, r + win)
                c0, c1 = max(0, c - win), min(W_p - 1, c + win)
                cand = []
                for rr in range(r0, r1 + 1):
                    for cc in range(c0, c1 + 1):
                        if rr == r and cc == c:
                            continue
                        cand.append(idx(tk, rr, cc))
                if not cand:
                    continue
                scores = A_np[i, cand]
                k = min(k_spatial, len(cand))
                top_idx = np.argpartition(-scores, k - 1)[:k]
                nbrs = [cand[j] for j in top_idx]
                for j in nbrs:
                    j_rel = j - offset - tk * P
                    rr, cc = divmod(j_rel, W_p)
                    r0j, r1j = max(0, rr - win), min(H_p - 1, rr + win)
                    c0j, c1j = max(0, cc - win), min(W_p - 1, cc + win)
                    cand_j = []
                    for rjj in range(r0j, r1j + 1):
                        for cjj in range(c0j, c1j + 1):
                            if rjj == rr and cjj == cc:
                                continue
                            cand_j.append(idx(tk, rjj, cjj))
                    if not cand_j:
                        continue
                    scores_j = A_np[j, cand_j]
                    kj = min(k_spatial, len(cand_j))
                    top_j = set([cand_j[u] for u in np.argpartition(-scores_j, kj - 1)[:kj]])
                    if i in top_j:
                        pair = tuple(sorted((i, j)))
                        spatial_pairs.add(pair)

    # Temporal edges between consecutive tokens
    for tk in range(T_tokens - 1):
        for r in range(H_p):
            for c in range(W_p):
                i = idx(tk, r, c)
                r0, r1 = max(0, r - win), min(H_p - 1, r + win)
                c0, c1 = max(0, c - win), min(W_p - 1, c + win)
                cand = [idx(tk + 1, rr, cc)
                        for rr in range(r0, r1 + 1)
                        for cc in range(c0, c1 + 1)]
                scores = A_np[i, cand]
                k = min(k_temporal, len(cand))
                if k == 0:
                    continue
                top_idx = np.argpartition(-scores, k - 1)[:k]
                nbrs = [cand[j] for j in top_idx]
                for j in nbrs:
                    j_rel = j - offset - (tk + 1) * P
                    rr, cc = divmod(j_rel, W_p)
                    r0j, r1j = max(0, rr - win), min(H_p - 1, rr + win)
                    c0j, c1j = max(0, cc - win), min(W_p - 1, cc + win)
                    cand_j = [idx(tk, rjj, cjj)
                              for rjj in range(r0j, r1j + 1)
                              for cjj in range(c0j, c1j + 1)]
                    scores_j = A_np[j, cand_j]
                    kj = min(k_temporal, len(cand_j))
                    top_j = set([cand_j[u] for u in np.argpartition(-scores_j, kj - 1)[:kj]])
                    if i in top_j:
                        pair = tuple(sorted((i, j)))
                        temporal_pairs.add(pair)

    # Convert to edge lists with frame indices
    spatial_edges = []
    for (i_idx, j_idx) in spatial_pairs:
        tk_i, pos_i = divmod(i_idx - offset, P)
        ri, ci = divmod(pos_i, W_p)
        tk_j, pos_j = divmod(j_idx - offset, P)
        rj, cj = divmod(pos_j, W_p)
        assert tk_i == tk_j
        f_start = tk_i * tubelet_size
        for df in range(tubelet_size):
            f_raw = f_start + df
            if f_raw < T_frames:
                spatial_edges.append((f_raw, ri, ci, f_raw, rj, cj))

    temporal_edges = []
    for (i_idx, j_idx) in temporal_pairs:
        tk_i, pos_i = divmod(i_idx - offset, P)
        ri, ci = divmod(pos_i, W_p)
        tk_j, pos_j = divmod(j_idx - offset, P)
        rj, cj = divmod(pos_j, W_p)
        assert tk_j == tk_i + 1
        f0 = tk_i * tubelet_size
        f1 = tk_j * tubelet_size
        if f0 < T_frames and f1 < T_frames:
            temporal_edges.append((f0, ri, ci, f1, rj, cj))

    return spatial_edges, temporal_edges, has_cls

# --- OBJ/MTL writer ---
def write_obj_with_edges(out_path, H_p, W_p, frames_rgb, spatial_edges, temporal_edges,
                         frame_spacing=6.0, unit=1.0):
    out_path = Path(out_path)
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
            v1 = add_vertex(x, 0, z0)
            v2 = add_vertex(x, frame_h, z0)
            grid_lines.append((v1, v2))
        for r in range(H_p + 1):
            y = r * unit
            v1 = add_vertex(0, y, z0)
            v2 = add_vertex(frame_w, y, z0)
            grid_lines.append((v1, v2))

    spatial_lines = []
    for (f, r1, c1, f2, r2, c2) in spatial_edges:
        assert f == f2
        v1 = verts_centers[(f, r1, c1)]
        v2 = verts_centers[(f, r2, c2)]
        spatial_lines.append((v1, v2))

    temporal_lines = []
    for (f0, r1, c1, f1, r2, c2) in temporal_edges:
        v1 = verts_centers[(f0, r1, c1)]
        v2 = verts_centers[(f1, r2, c2)]
        temporal_lines.append((v1, v2))

    with open(out_mtl, "w", encoding="utf-8") as m:
        m.write("# Materials for tubes OBJ\n")
        m.write("newmtl Grid\nKd 0.3 0.3 0.3\nKa 0.0 0.0 0.0\nd 1.0\nillum 1\n\n")
        m.write("newmtl Spatial\nKd 1.000 0.500 0.000\nKa 0.0 0.0 0.0\nd 1.0\nillum 1\n\n")
        m.write("newmtl Temporal\nKd 0.000 0.450 1.000\nKa 0.0 0.0 0.0\nd 1.0\nillum 1\n\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Space–time mutual-kNN visualization from VideoMAE attentions\n")
        f.write(f"mtllib {out_mtl.name}\n")
        for (x, y, z) in vertices:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        f.write("o grid_dividers\nusemtl Grid\n")
        for (a, b) in grid_lines:
            f.write(f"l {a} {b}\n")
        f.write("o spatial_edges\nusemtl Spatial\n")
        for (a, b) in spatial_lines:
            f.write(f"l {a} {b}\n")
        f.write("o temporal_edges\nusemtl Temporal\n")
        for (a, b) in temporal_lines:
            f.write(f"l {a} {b}\n")

    return str(out_path), str(out_mtl)

# --- MP4 generator ---
def generate_video_output(out_path, all_edges, original_width, original_height,
                          image_size, patch_size, fps):
    import cv2
    import numpy as np

    scale_x = image_size / original_width
    scale_y = image_size / original_height
    patch_width_orig = patch_size / scale_x
    patch_height_orig = patch_size / scale_y
    H_p = image_size // patch_size
    W_p = image_size // patch_size

    cap = cv2.VideoCapture(out_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(out_path, fourcc, fps, (original_width, original_height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to BGR if needed
        if frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Draw grid lines
        for c in range(W_p + 1):
            x = int(round(c * patch_width_orig))
            cv2.line(frame, (x, 0), (x, original_height), (0, 0, 0), 1)
        for r in range(H_p + 1):
            y = int(round(r * patch_height_orig))
            cv2.line(frame, (0, y), (original_width, y), (0, 0, 0), 1)

        # Draw patch centers
        for r in range(H_p):
            for c in range(W_p):
                center_x = int(round((c + 0.5) * patch_width_orig))
                center_y = int(round((r + 0.5) * patch_height_orig))
                cv2.circle(frame, (center_x, center_y), 3, (0, 165, 255), -1)

        # Draw spatial edges
        edges_this_frame = [edge for edge in all_edges if edge[0] == frame_idx]
        for (f, r1, c1, f2, r2, c2) in edges_this_frame:
            center_x1 = int(round((c1 + 0.5) * patch_width_orig))
            center_y1 = int(round((r1 + 0.5) * patch_height_orig))
            center_x2 = int(round((c2 + 0.5) * patch_width_orig))
            center_y2 = int(round((r2 + 0.5) * patch_height_orig))
            cv2.line(frame, (center_x1, center_y1), (center_x2, center_y2), (0, 165, 255), 2)

        video_writer.write(frame)
        frame_idx += 1

    cap.release()
    video_writer.release()
    return str(out_path)

# --- Main processing ---
def main():
    parser = argparse.ArgumentParser(description="VideoMAE mutual‑kNN → Blender OBJ or MP4")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--start", type=int, default=0, help="Start frame (for OBJ)")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames (for OBJ)")
    parser.add_argument("--model_id", type=str, default="MCG-NJU/videomae-base",
                        help="HF model id")
    parser.add_argument("--layer", type=int, default=6, help="Encoder layer index")
    parser.add_argument("--k_spatial", type=int, default=8, help="Mutual‑kNN k for spatial edges")
    parser.add_argument("--k_temporal", type=int, default=4, help="Mutual‑kNN k for temporal edges")
    parser.add_argument("--win", type=int, default=2, help="Chebyshev window radius")
    parser.add_argument("--frame_spacing", type=float, default=6.0, help="Frame spacing in OBJ")
    parser.add_argument("--unit", type=float, default=1.0, help="Patch cell size in OBJ")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--out", type=str, required=True, help="Output path (.obj or .mp4)")
    args = parser.parse_args()

    # Load model and processor
    model, processor = load_model_and_processor(args.model_id, device=args.device, dtype_str=args.dtype)
    image_size = model.config.image_size
    patch_size = model.config.patch_size
    tubelet_size = model.config.tubelet_size

    if args.out.lower().endswith('.mp4'):
        # Process entire video for MP4 output
        fps, original_width, original_height, total_frames = get_video_properties(args.video)
        chunk_size = 16
        all_edges = []

        # Process video in chunks
        for chunk_start in range(0, total_frames, chunk_size):
            num_in_chunk = min(chunk_size, total_frames - chunk_start)
            if num_in_chunk < chunk_size:
                continue  # Skip incomplete chunks

            # Read and preprocess chunk
            frames = read_frames_chunk(args.video, chunk_start, num_in_chunk, resize_to=(image_size, image_size))
            iproc = processor if isinstance(processor, AutoImageProcessor.__mro__[0]) else processor
            pixel_values = iproc(frames, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(args.device)

            # Run model
            with torch.no_grad():
                outputs = model(pixel_values, output_attentions=True)

            assert outputs.attentions, "Output Attentions are null!"
            attentions = outputs.attentions
            if args.layer < 0 or args.layer >= len(attentions):
                print(f"[!] Layer index {args.layer} out of range (0..{len(attentions)-1})", file=sys.stderr)
                sys.exit(1)
            attn = attentions[args.layer][0]
            attn_mean = attn.mean(dim=0)

            # Build edges
            H_p = W_p = image_size // patch_size
            spatial_edges, _, _ = build_edges_from_attn(
                attn_mean, T_frames=num_in_chunk, H_p=H_p, W_p=W_p, tubelet_size=tubelet_size,
                k_spatial=args.k_spatial, k_temporal=args.k_temporal, win=args.win
            )

            # Offset frame indices
            spatial_edges_offseted = [(f + chunk_start, r1, c1, f2, r2, c2)
                                      for f, r1, c1, f2, r2, c2 in spatial_edges]
            all_edges.extend(spatial_edges_offseted)

        # Generate MP4
        video_path = generate_video_output(
            args.out, all_edges, original_width, original_height,
            image_size, patch_size, fps
        )
        print(f"[OK] Wrote video: {video_path}")

    else:
        # OBJ output (single chunk)
        frames = read_frames_chunk(args.video, args.start, args.num_frames, resize_to=(image_size, image_size))
        iproc = processor if isinstance(processor, AutoImageProcessor.__mro__[0]) else processor
        pixel_values = iproc(frames, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(args.device)

        with torch.no_grad():
            outputs = model(pixel_values, output_attentions=True)

        assert outputs.attentions, "Output Attentions are null!"
        attentions = outputs.attentions
        if args.layer < 0 or args.layer >= len(attentions):
            print(f"[!] Layer index {args.layer} out of range (0..{len(attentions)-1})", file=sys.stderr)
            sys.exit(1)
        attn = attentions[args.layer][0]
        attn_mean = attn.mean(dim=0)

        H_p = W_p = image_size // patch_size
        spatial_edges, temporal_edges, has_cls = build_edges_from_attn(
            attn_mean, T_frames=args.num_frames, H_p=H_p, W_p=W_p, tubelet_size=tubelet_size,
            k_spatial=args.k_spatial, k_temporal=args.k_temporal, win=args.win
        )

        obj_path, mtl_path = write_obj_with_edges(
            args.out, H_p, W_p, frames_rgb=frames,
            spatial_edges=spatial_edges, temporal_edges=temporal_edges,
            frame_spacing=args.frame_spacing, unit=args.unit
        )
        print(f"[OK] Wrote: {obj_path}\n[OK] Wrote: {mtl_path}")
        print("Import into Blender: File → Import → Wavefront (.obj)")

if __name__ == "__main__":
    main()
