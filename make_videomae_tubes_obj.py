"""
Build a space–time mutual‑kNN graph from VideoMAE attentions and export a Blender‑compatible OBJ+MTL.

Requirements:
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124  # or CPU wheel
  pip install transformers av opencv-python numpy

Usage:
  python make_videomae_tubes_obj.py \
      --video path/to/video.mp4 \
      --start 100 \
      --out tubes.obj
Optional:
  --model_id MCG-NJU/videomae-base --layer 6 --k_spatial 8 --k_temporal 4 --win 2
  --frame_spacing 6.0 --unit 1.0 --device cuda
Notes:
  • Uses attentions from a chosen encoder layer, averaged across heads.
  • Orange = spatial edges, Blue = temporal edges, Grey = patch grid lines.
  • Frames are arranged in a row (front to back) rather than side-by-side.
  • OBJ+MTL can be imported in Blender (File → Import → Wavefront (.obj)).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# --- Video decoding: prefer PyAV (accurate seeking), fallback to OpenCV ---
def read_frames(video_path, start, num, resize_to=None):
    """
    Returns a list of RGB frames (H,W,3) uint8 for indices [start, start+num).
    If resize_to is not None, it should be (H, W).
    """
    frames = []
    try:
        import av  # PyAV (recommended)
        container = av.open(video_path)
        stream = container.streams.video[0]
        # PyAV seeks by timestamp; approximate by frame
        container.seek(int((start / stream.average_rate) * stream.time_base.denominator),
                       any_frame=False, backward=True, stream=stream)
        idx = -1
        for packet in container.demux(stream):
            for frame in packet.decode():
                idx += 1
                # We’ll overshoot; collect when in range
                if idx < start:
                    continue
                if idx >= start + num:
                    container.close()
                    if resize_to is not None:
                        frames = [cv2_resize_safe(f, resize_to) for f in frames]
                    return frames
                img = frame.to_ndarray(format="rgb24")  # (H,W,3) uint8
                frames.append(img)
        container.close()
    except Exception:
        # Fallback to OpenCV (may be less precise on random seeks)
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if start >= total:
            raise ValueError(f"Start index {start} >= total frames {total}")
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

# --- Model / processor ---
def load_model_and_processor(model_id, device="cpu", dtype_str="float32"):
    import torch
    from transformers import AutoImageProcessor, VideoMAEModel
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_str]
    processor = AutoImageProcessor.from_pretrained(model_id)  # VideoMAEImageProcessor
    model = VideoMAEModel.from_pretrained(model_id, attn_implementation="eager")
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, processor

# --- Build attention-based mutual-kNN (spatial + temporal) ---
def build_edges_from_attn(attn_L, T_frames, H_p, W_p, tubelet_size, k_spatial=8, k_temporal=4, win=2):
    """
    attn_L: (L, L) attention (averaged across heads), including CLS if present.
    T_frames: number of raw frames fed to model (e.g., 8)
    H_p, W_p: patch grid dims per frame
    tubelet_size: e.g., 2
    Returns:
      spatial_edges: list of (t_raw, r, c, t_raw, r2, c2) with t_raw in [0, T_frames)
      temporal_edges: list of (t_raw, r, c, t_raw+1, r2, c2) for cross-time edges between consecutive raw frames
    """
    P = H_p * W_p
    T_tokens = T_frames // tubelet_size  # number of time tokens
    L_total = attn_L.shape[0]
    # Detect CLS by comparing sequence length
    has_cls = (L_total == (T_tokens * P) + 1)
    offset = 1 if has_cls else 0

    # Helper to map (t_token, r, c) -> sequence index
    def idx(tk, r, c):
        return offset + tk * P + r * W_p + c

    # For mutual-kNN, precompute top-k neighbor sets using attention similarity
    # We symmetrize attention to get an undirected weight
    A = (attn_L + attn_L.T) * 0.5
    A_np = A.detach().cpu().numpy() if hasattr(A, "detach") else np.asarray(A)

    spatial_pairs = set()
    temporal_pairs = set()

    # Spatial edges within each time token tk
    for tk in range(T_tokens):
        for r in range(H_p):
            for c in range(W_p):
                i = idx(tk, r, c)
                # Candidate indices in same tk within window
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
                # Top-k within window
                k = min(k_spatial, len(cand))
                top_idx = np.argpartition(-scores, k - 1)[:k]
                nbrs = [cand[j] for j in top_idx]
                # Mutual check
                for j in nbrs:
                    # map j back to (rr,cc) for window around j; we can simply check i in j's own top-k within same tk
                    # Recover (rr,cc) from j
                    j_rel = j - offset - tk * P
                    rr, cc = divmod(j_rel, W_p)
                    # j's candidate set
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

    # Temporal edges between tk and tk+1 within window
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
                    # Mutual from tk+1 back to tk
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

    # Map token indices to (raw frame index, r, c)
    # Tokens are at "time tokens" tk; each covers tubelet_size raw frames: [f0, ..., f0+tubelet_size-1]
    # For drawing spatial edges, replicate onto each raw frame inside the tubelet.
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
        f0 = tk_i * tubelet_size      # first frame of tk_i
        f1 = tk_j * tubelet_size      # first frame of tk_j
        if f0 < T_frames and f1 < T_frames:
            temporal_edges.append((f0, ri, ci, f1, rj, cj))

    return spatial_edges, temporal_edges, has_cls

# --- OBJ/MTL writer ---
def write_obj_with_edges(out_path, H_p, W_p, frames_rgb, spatial_edges, temporal_edges,
                         frame_spacing=6.0, unit=1.0):
    """
    Writes an OBJ + MTL with:
      - patch-grid dividers per frame,
      - per-frame patch centers as vertices,
      - orange lines for spatial (intra-frame) edges,
      - blue lines for temporal edges (across frames).
    Frame arrangement: Frames are positioned in a row along the Z-axis (front to back)
    rather than side-by-side.
    """
    out_path = Path(out_path)
    out_mtl = out_path.with_suffix(".mtl")

    # Build vertex positions for patch centers per frame
    # Frame layout: along +Z (front to back), each frame width = W_p*unit, gap = frame_spacing
    verts_centers = {}  # (f, r, c) -> vertex index (1-based)
    vertices = []

    # Also build grid line vertices
    grid_lines = []  # list of (v_idx1, v_idx2)
    vcount = 0

    def add_vertex(x, y, z=0.0):
        nonlocal vcount
        vertices.append((x, y, z))
        vcount += 1
        return vcount  # 1-based index

    # Precompute frame dimensions
    frame_w = W_p * unit
    frame_h = H_p * unit

    # Patch centers and grid dividers
    for f_idx in range(len(frames_rgb)):
        z0 = f_idx * frame_spacing  # Position frames along Z axis (front to back)
        # Patch centers
        for r in range(H_p):
            for c in range(W_p):
                x = (c + 0.5) * unit
                y = (H_p - 1 - r + 0.5) * unit  # origin at bottom-left
                idx_v = add_vertex(x, y, z0)
                verts_centers[(f_idx, r, c)] = idx_v
        # Grid vertical lines
        for c in range(W_p + 1):
            x = c * unit
            v1 = add_vertex(x, 0, z0)
            v2 = add_vertex(x, frame_h, z0)
            grid_lines.append((v1, v2))
        # Grid horizontal lines
        for r in range(H_p + 1):
            y = r * unit
            v1 = add_vertex(0, y, z0)
            v2 = add_vertex(frame_w, y, z0)
            grid_lines.append((v1, v2))

    # Build line lists for edges
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

    # Write MTL (materials: Grid, Spatial, Temporal)
    with open(out_mtl, "w", encoding="utf-8") as m:
        m.write("# Materials for tubes OBJ\n")
        m.write("newmtl Grid\nKd 0.3 0.3 0.3\nKa 0.0 0.0 0.0\nd 1.0\nillum 1\n\n")
        m.write("newmtl Spatial\n# orange-ish\nKd 1.000 0.500 0.000\nKa 0.0 0.0 0.0\nd 1.0\nillum 1\n\n")
        m.write("newmtl Temporal\n# blue-ish\nKd 0.000 0.450 1.000\nKa 0.0 0.0 0.0\nd 1.0\nillum 1\n\n")

    # Write OBJ
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Space–time mutual-kNN visualization from VideoMAE attentions\n")
        f.write(f"mtllib {out_mtl.name}\n")
        # All vertices
        for (x, y, z) in vertices:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")

        # Grid object
        f.write("o grid_dividers\nusemtl Grid\n")
        for (a, b) in grid_lines:
            f.write(f"l {a} {b}\n")

        # Spatial edges
        f.write("o spatial_edges\nusemtl Spatial\n")
        for (a, b) in spatial_lines:
            f.write(f"l {a} {b}\n")

        # Temporal edges
        f.write("o temporal_edges\nusemtl Temporal\n")
        for (a, b) in temporal_lines:
            f.write(f"l {a} {b}\n")

    return str(out_path), str(out_mtl)


def generate_video_output(out_path, H_p, W_p, frames_rgb, spatial_edges, temporal_edges,
                         frame_spacing=6.0, unit=1.0):
    """
    Generates an MP4 video with:
      - Original video frames displayed
      - Thin black grid lines for patches
      - Orange circles at patch centers
      - Orange lines for spatial connections
      - Orange lines for temporal connections
    """
    import cv2
    import numpy as np
    
    out_path = Path(out_path)
    
    # Get original frame dimensions
    original_height, original_width = frames_rgb[0].shape[:2]
    
    # Calculate patch dimensions in original coordinates
    patch_width = original_width // W_p
    patch_height = original_height // H_p
    
    # Create output frames
    output_frames = []
    
    for frame_idx, frame in enumerate(frames_rgb):
        # Convert RGB to BGR for OpenCV
        output_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Draw grid lines (thin black)
        # Vertical lines
        for c in range(W_p + 1):
            x = c * patch_width
            cv2.line(output_frame, (x, 0), (x, original_height), (0, 0, 0), 1)
        
        # Horizontal lines
        for r in range(H_p + 1):
            y = r * patch_height
            cv2.line(output_frame, (0, y), (original_width, y), (0, 0, 0), 1)
        
        # Draw orange circles at patch centers (BGR format: blue=0, green=165, red=255)
        for r in range(H_p):
            for c in range(W_p):
                center_x = c * patch_width + patch_width // 2
                center_y = r * patch_height + patch_height // 2
                cv2.circle(output_frame, (center_x, center_y), 3, (0, 165, 255), -1)
        
        # Draw spatial edges (orange lines, BGR format)
        for (f, r1, c1, f2, r2, c2) in spatial_edges:
            if f == frame_idx:
                x1 = c1 * patch_width + patch_width // 2
                y1 = r1 * patch_height + patch_height // 2
                x2 = c2 * patch_width + patch_width // 2
                y2 = r2 * patch_height + patch_height // 2
                cv2.line(output_frame, (x1, y1), (x2, y2), (0, 165, 255), 1)
        
        # Draw temporal edges (orange lines, BGR format)
        for (f0, r1, c1, f1, r2, c2) in temporal_edges:
            if f0 == frame_idx:
                x1 = c1 * patch_width + patch_width // 2
                y1 = r1 * patch_height + patch_height // 2
                x2 = c2 * patch_width + patch_width // 2
                y2 = r2 * patch_height + patch_height // 2
                cv2.line(output_frame, (x1, y1), (x2, y2), (0, 165, 255), 1)
        
        output_frames.append(output_frame)
    
    # Write video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    video_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (original_width, original_height))
    
    for frame in output_frames:
        video_writer.write(frame)
    
    video_writer.release()
    return str(out_path)

def main():
    parser = argparse.ArgumentParser(description="VideoMAE mutual‑kNN → Blender OBJ")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--start", type=int, required=True, help="Start frame index i")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames (default: 16)")
    parser.add_argument("--model_id", type=str, default="MCG-NJU/videomae-base",
                        help="HF model id (default: MCG-NJU/videomae-base)")
    parser.add_argument("--layer", type=int, default=6, help="Encoder layer index to read attentions from (0-based)")
    parser.add_argument("--k_spatial", type=int, default=8, help="Mutual‑kNN k for spatial edges")
    parser.add_argument("--k_temporal", type=int, default=4, help="Mutual‑kNN k for temporal edges")
    parser.add_argument("--win", type=int, default=2, help="Chebyshev window radius around a patch")
    parser.add_argument("--frame_spacing", type=float, default=6.0, help="Spacing between frames in OBJ units")
    parser.add_argument("--unit", type=float, default=1.0, help="Patch cell size in OBJ units")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run the model")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--out", type=str, required=True, help="Output path (OBJ/MTL files or MP4 video)")
    args = parser.parse_args()

    # Load model/processor
    model, processor = load_model_and_processor(args.model_id, device=args.device, dtype_str=args.dtype)

    # Decode frames [i, i+8)
    # Resize to model's expected image_size
    image_size = getattr(model.config, "image_size", 224)
    frames = read_frames(args.video, args.start, args.num_frames, resize_to=(image_size, image_size))

    # Preprocess
    from transformers import AutoImageProcessor
    iproc = processor if isinstance(processor, AutoImageProcessor.__mro__[0]) else processor
    pixel_values = iproc(frames, return_tensors="pt").pixel_values  # shape as required by model
    # Model expects (B, C, T, H, W) or (B, T, C, H, W) depending on version; the processor arranges it correctly.

    import torch
    pixel_values = pixel_values.to(args.device)
    with torch.no_grad():
        outputs = model(pixel_values, output_attentions=True)

    # --------------
    assert outputs.attentions, "Output Attentions are null!"
    # --------------
    
    attentions = outputs.attentions  # tuple(len = num_layers) of (B, heads, L, L)
    if args.layer < 0 or args.layer >= len(attentions):
        print(f"[!] Layer index {args.layer} out of range (0..{len(attentions)-1}).", file=sys.stderr)
        sys.exit(1)
    attn = attentions[args.layer][0]  # (heads, L, L)
    attn_mean = attn.mean(dim=0)      # (L, L)

    # Token geometry
    H_p = W_p = image_size // model.config.patch_size
    tubelet = model.config.tubelet_size
    # Build mutual‑kNN edges
    spatial_edges, temporal_edges, has_cls = build_edges_from_attn(
        attn_mean, T_frames=args.num_frames, H_p=H_p, W_p=W_p, tubelet_size=tubelet,
        k_spatial=args.k_spatial, k_temporal=args.k_temporal, win=args.win
    )

    # Determine output format and write accordingly
    if args.out.lower().endswith('.mp4'):
        # Generate video output
        video_path = generate_video_output(
            args.out, H_p, W_p, frames_rgb=frames,
            spatial_edges=spatial_edges, temporal_edges=temporal_edges,
            frame_spacing=args.frame_spacing, unit=args.unit
        )
        print(f"[OK] Wrote video: {video_path}")
    else:
        # Write OBJ/MTL files
        obj_path, mtl_path = write_obj_with_edges(
            args.out, H_p, W_p, frames_rgb=frames,
            spatial_edges=spatial_edges, temporal_edges=temporal_edges,
            frame_spacing=args.frame_spacing, unit=args.unit
        )
        print(f"[OK] Wrote: {obj_path}\n[OK] Wrote: {mtl_path}")
        print("Import into Blender: File → Import → Wavefront (.obj)")

if __name__ == "__main__":
    main()
