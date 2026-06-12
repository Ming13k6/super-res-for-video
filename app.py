import gradio as gr
import cv2
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model import ESPCN
import imageio
import time
import math

# =========================
# DEVICE
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

# =========================
# LOAD MODEL
# =========================

model = ESPCN(scale_factor=4).to(device)

model.load_state_dict(
    torch.load(
        "espcn.pth",
        map_location=device
    )
)

model.eval()

# =========================
# TRANSFORM
# =========================

transform = transforms.ToTensor()

# =========================
# PSNR
# =========================

def calculate_psnr(img1, img2):

    mse = np.mean(
        (
            img1.astype(np.float32)
            - img2.astype(np.float32)
        ) ** 2
    )

    if mse == 0:
        return 100

    return 20 * math.log10(
        255.0 / math.sqrt(mse)
    )

# =========================
# SIMPLE QUALITY RATING
# =========================

def quality_rating(psnr):

    if psnr >= 35:
        return "Excellent"

    elif psnr >= 30:
        return "Good"

    elif psnr >= 25:
        return "Fair"

    else:
        return "Poor"

# =========================
# MAIN PROCESS
# =========================

def process_video(video_path):

    os.makedirs("output", exist_ok=True)

    start_time = time.time()

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    input_size = os.path.getsize(video_path)

    output_enhanced = "output/enhanced_video.mp4"
    output_compare = "output/comparison_video.mp4"

    # =========================
    # VIDEO WRITERS
    # =========================

    writer_enh = imageio.get_writer(
        output_enhanced,
        fps=fps,
        codec="libx264",
        macro_block_size=None
    )

    writer_cmp = imageio.get_writer(
        output_compare,
        fps=fps,
        codec="libx264",
        macro_block_size=None
    )

    psnr_scores = []

    # =========================
    # SETTINGS
    # =========================

    batch_size = 4

    frame_buffer = []

    # =========================
    # PROCESS BATCH
    # =========================

    def process_batch(frames):

        tensors = []

        bicubic_frames = []

        for frame in frames:

            h, w = frame.shape[:2]

            # =========================
            # LIMIT HUGE VIDEOS
            # =========================

            if w > 1920:

                scale = 1920 / w

                new_w = int(w * scale)
                new_h = int(h * scale)

                frame = cv2.resize(
                    frame,
                    (new_w, new_h)
                )

                h, w = frame.shape[:2]

            # =========================
            # CREATE LR FRAME
            # =========================

            lr = cv2.resize(
                frame,
                (w // 4, h // 4),
                interpolation=cv2.INTER_CUBIC
            )

            # =========================
            # BICUBIC UPSCALE
            # =========================

            bicubic = cv2.resize(
                lr,
                (w, h),
                interpolation=cv2.INTER_CUBIC
            )

            bicubic_frames.append(bicubic)

            # =========================
            # PREPARE MODEL INPUT
            # =========================

            rgb = cv2.cvtColor(
                lr,
                cv2.COLOR_BGR2RGB
            )

            pil = Image.fromarray(rgb)

            tensor = transform(pil)

            tensors.append(tensor)

        # =========================
        # STACK BATCH
        # =========================

        batch = torch.stack(
            tensors
        ).to(device)

        # =========================
        # MODEL INFERENCE
        # =========================

        with torch.no_grad():

            outputs = model(batch)

        outputs = outputs.clamp(0, 1)

        # =========================
        # PROCESS OUTPUTS
        # =========================

        for i in range(len(outputs)):

            out_img = (
                outputs[i]
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )

            out_img = (
                out_img * 255
            ).clip(0, 255).astype(np.uint8)

            out_img = cv2.cvtColor(
                out_img,
                cv2.COLOR_RGB2BGR
            )

            bicubic = bicubic_frames[i]

            h, w = bicubic.shape[:2]

            out_img = cv2.resize(
                out_img,
                (w, h)
            )

            # =========================
            # PSNR
            # =========================

            psnr = calculate_psnr(
                bicubic,
                out_img
            )

            psnr_scores.append(psnr)

            # =========================
            # LABELS
            # =========================

            cv2.putText(
                bicubic,
                "Bicubic",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                out_img,
                "ESPCN",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # =========================
            # COMPARISON VIDEO
            # =========================

            compare = np.hstack(
                (bicubic, out_img)
            )

            # =========================
            # WRITE VIDEOS
            # =========================

            writer_enh.append_data(
                cv2.cvtColor(
                    out_img,
                    cv2.COLOR_BGR2RGB
                )
            )

            writer_cmp.append_data(
                cv2.cvtColor(
                    compare,
                    cv2.COLOR_BGR2RGB
                )
            )

            del out_img
            del compare

        del batch
        del outputs

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    # =========================
    # READ VIDEO
    # =========================

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_buffer.append(frame)

        if len(frame_buffer) == batch_size:

            process_batch(frame_buffer)

            frame_buffer = []

    # Process remaining frames

    if len(frame_buffer) > 0:

        process_batch(frame_buffer)

    cap.release()

    writer_enh.close()
    writer_cmp.close()

    # =========================
    # METRICS
    # =========================

    total_time = round(
        time.time() - start_time,
        2
    )

    output_size = os.path.getsize(
        output_enhanced
    )

    duration = total_frames / fps

    input_bitrate = round(
        (input_size * 8) / duration / 1000,
        2
    )

    output_bitrate = round(
        (output_size * 8) / duration / 1000,
        2
    )

    compression_ratio = round(
        input_size / output_size,
        2
    )

    latency = round(
        total_time / total_frames,
        4
    )

    avg_psnr = round(
        sum(psnr_scores) / len(psnr_scores),
        2
    )

    rating = quality_rating(avg_psnr)

    # =========================
    # PERCEPTION COMPARISON
    # =========================

    perception = f"""
Bicubic interpolation produces smoother
but blurrier frames.

ESPCN reconstruction preserves sharper
edges and finer details.

Visual quality rating: {rating}
"""

    # =========================
    # METRICS OUTPUT
    # =========================

    metrics = f"""
=========================
EXPERIMENTAL RESULTS
=========================

Processing Time:
{total_time} sec

Average Latency:
{latency} sec/frame

Input Bitrate:
{input_bitrate} kbps

Output Bitrate:
{output_bitrate} kbps

Compression Ratio:
{compression_ratio}

Average PSNR:
{avg_psnr} dB

Perceptual Rating:
{rating}

Perception Comparison:
{perception}
"""

    return (
        output_enhanced,
        output_compare,
        metrics
    )

# =========================
# UI
# =========================

demo = gr.Interface(
    fn=process_video,

    inputs=gr.Video(
        label="Upload Video"
    ),

    outputs=[
        gr.Video(
            label="Enhanced Video"
        ),

        gr.Video(
            label="Comparison Video"
        ),

        gr.Textbox(
            label="Experimental Results",
            lines=20
        )
    ],

    title="ESPCN Video Super Resolution",

    description="""
Upload a low-resolution video
to enhance it using ESPCN.
"""
)

demo.launch(
    share=False,
    inbrowser=True
)