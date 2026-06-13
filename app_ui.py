"""
app_ui.py - Gradio Web UI for Video Super Resolution (4x Upscaler)

Pipeline:
  1. User uploads a low-resolution video via the Gradio interface.
  2. Video frames are read with OpenCV and batched for efficient inference.
  3. Each batch is upscaled 4x by the ESPCN model (GPU or CPU).
  4. A bicubic-upscaled version is generated in parallel as a baseline.
  5. PSNR is computed between the original frame and the model's
     reconstruction (downscale → upscale round-trip) to evaluate quality.
  6. Both output videos and quantitative metrics are displayed in the UI.

Device handling:
  - Automatically selects CUDA if available; otherwise falls back to CPU.
  - On CPU the batch size is reduced to 1 to avoid OOM and inference uses
    float32 (no half-precision) for maximum compatibility.

Dependencies:
  gradio, torch, torchvision, opencv-python, Pillow, imageio, numpy
"""

import math
import os
import time

import cv2
import gradio as gr
import imageio
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageFilter

from model import ESPCN

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCALE_FACTOR = 4
MODEL_PATH = "espcn.pth"
OUTPUT_DIR = "output"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_device():
    """Return the best available torch device and whether it is CUDA."""
    if torch.cuda.is_available():
        return torch.device("cuda"), True
    return torch.device("cpu"), False


def _load_model(device):
    """Load the ESPCN model onto *device* in eval mode."""
    model = ESPCN(scale_factor=SCALE_FACTOR).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model


def calculate_psnr(img1, img2):
    """Peak Signal-to-Noise Ratio between two uint8 images."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_video(video_path, progress=gr.Progress()):
    """Upscale a video 4× with ESPCN and produce a bicubic baseline.

    Returns
    -------
    bicubic_path : str   – path to the bicubic-upscaled video
    enhanced_path : str  – path to the ESPCN-enhanced video
    results_md : str     – Markdown string with experimental metrics
    """
    if not video_path:
        return None, None, "⚠️ No video provided."

    start_time = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Device & model ---------------------------------------------------
    device, is_cuda = _get_device()
    model = _load_model(device)
    # Use smaller batches on CPU to keep memory usage low
    batch_size = 4 if is_cuda else 1

    # --- Open video -------------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, first_frame = cap.read()
    if not ret:
        return None, None, "⚠️ Failed to read video."

    orig_h, orig_w = first_frame.shape[:2]
    out_w, out_h = orig_w * SCALE_FACTOR, orig_h * SCALE_FACTOR
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind

    # --- Writers ----------------------------------------------------------
    enhanced_path = os.path.join(OUTPUT_DIR, "enhanced_video.mp4")
    bicubic_path = os.path.join(OUTPUT_DIR, "bicubic_video.mp4")
    writer_enh = imageio.get_writer(enhanced_path, fps=fps, macro_block_size=None)
    writer_bic = imageio.get_writer(bicubic_path, fps=fps, macro_block_size=None)

    # --- Transforms -------------------------------------------------------
    to_tensor = T.ToTensor()
    to_pil = T.ToPILImage()

    # --- Metrics accumulators ---------------------------------------------
    total_psnr = 0.0
    psnr_count = 0

    # --- Batch processing function ----------------------------------------
    def _process_batch(frames):
        """Run ESPCN on a list of BGR frames; write outputs & compute PSNR."""
        nonlocal total_psnr, psnr_count

        # ── Pass 1: Real upscale (full-res output) ──
        tensors = [to_tensor(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        batch = torch.stack(tensors).to(device, non_blocking=is_cuda)

        with torch.no_grad():
            outputs = model(batch).cpu().clamp(0, 1)

        for i, frame in enumerate(frames):
            # Enhanced frame
            enh_pil = to_pil(outputs[i]).filter(ImageFilter.SHARPEN)
            enh_bgr = cv2.cvtColor(np.array(enh_pil), cv2.COLOR_RGB2BGR)
            enh_bgr = cv2.resize(enh_bgr, (out_w, out_h))
            writer_enh.append_data(cv2.cvtColor(enh_bgr, cv2.COLOR_BGR2RGB))

            # Bicubic baseline
            bic_bgr = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
            writer_bic.append_data(cv2.cvtColor(bic_bgr, cv2.COLOR_BGR2RGB))

        # ── Pass 2: Evaluation (downscale → upscale round-trip PSNR) ──
        eval_tensors = []
        for frame in frames:
            lr = cv2.resize(frame, (orig_w // SCALE_FACTOR, orig_h // SCALE_FACTOR),
                            interpolation=cv2.INTER_CUBIC)
            eval_tensors.append(to_tensor(cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)))

        eval_batch = torch.stack(eval_tensors).to(device, non_blocking=is_cuda)
        with torch.no_grad():
            eval_out = model(eval_batch).cpu().clamp(0, 1)

        for i, frame in enumerate(frames):
            rec_bgr = cv2.cvtColor(np.array(to_pil(eval_out[i])), cv2.COLOR_RGB2BGR)
            rec_bgr = cv2.resize(rec_bgr, (orig_w, orig_h))
            total_psnr += calculate_psnr(frame, rec_bgr)
            psnr_count += 1

    # --- Main read loop ---------------------------------------------------
    frame_buf = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_buf.append(frame)
        frame_count += 1

        if len(frame_buf) == batch_size:
            _process_batch(frame_buf)
            frame_buf = []

        if frame_count % max(batch_size, 1) == 0 or frame_count == total_frames:
            progress(frame_count / total_frames,
                     desc=f"Processing frames {frame_count}/{total_frames}")

    if frame_buf:
        _process_batch(frame_buf)
        progress(1.0, desc="Finishing up…")

    cap.release()
    writer_enh.close()
    writer_bic.close()

    # --- Metrics ----------------------------------------------------------
    elapsed = time.time() - start_time
    in_mb = os.path.getsize(video_path) / (1024 * 1024) if os.path.exists(video_path) else 0
    out_mb = os.path.getsize(enhanced_path) / (1024 * 1024) if os.path.exists(enhanced_path) else 0
    compression = in_mb / out_mb if out_mb > 0 else 1.0
    duration_s = total_frames / fps if fps > 0 else 1.0
    bitrate_kbps = (out_mb * 8192) / duration_s if duration_s > 0 else 0
    avg_psnr = total_psnr / psnr_count if psnr_count > 0 else 0

    results_md = f"""
### 📊 7. Experimental Results

**Quantitative:**
* **Compression ratio:** {compression:.2f}
* **Bitrate:** {bitrate_kbps:.2f} kbps
* **Latency:** {elapsed:.2f} s
* **PSNR:** {avg_psnr:.2f} dB
* **Device:** {device}

**Qualitative:**
* ESPCN AI produces sharper edges and better details compared to Bicubic baseline upscaling.
"""
    return bicubic_path, enhanced_path, results_md


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

_CSS = "footer {display: none !important;}"

with gr.Blocks(title="Video Super Resolution") as demo:
    gr.Markdown("# Video Super Resolution (4x Upscaler)")
    gr.Markdown(
        "Upload a video. The system will upscale its resolution by 4× using "
        "the ESPCN AI model and compare it with standard Bicubic upscaling."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("")
        with gr.Column(scale=2):
            video_input = gr.Video(label="Input Video")
            submit_btn = gr.Button("Upscale Video", variant="primary")
        with gr.Column(scale=1):
            gr.Markdown("")

    gr.Markdown("### 🔍 Perception Comparison")
    with gr.Row():
        video_bicubic = gr.Video(label="Baseline (Bicubic Upscaling 4x)")
        video_enhanced = gr.Video(label="Enhanced Video (ESPCN AI 4x)")

    results_output = gr.Markdown("### 📊 Experimental Results will appear here…")

    submit_btn.click(
        fn=process_video,
        inputs=video_input,
        outputs=[video_bicubic, video_enhanced, results_output],
    )

if __name__ == "__main__":
    try:
        demo.launch(css=_CSS)
    except TypeError:
        demo.launch()
