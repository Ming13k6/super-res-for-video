import cv2
import os
import numpy as np

original_folder = "video_frames"
enhanced_folder = "enhanced_frames"

output_path = "output/comparison.mp4"

fps = 30

original_images = sorted(
    os.listdir(original_folder)
)

enhanced_images = sorted(
    os.listdir(enhanced_folder)
)

first_frame = cv2.imread(
    os.path.join(
        original_folder,
        original_images[0]
    )
)

height, width, _ = first_frame.shape

writer = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width * 2, height)
)

for orig, enh in zip(
        original_images,
        enhanced_images):

    orig_frame = cv2.imread(
        os.path.join(
            original_folder,
            orig
        )
    )

    enh_frame = cv2.imread(
        os.path.join(
            enhanced_folder,
            enh
        )
    )

    # simulate LR
    lr_frame = cv2.resize(
        orig_frame,
        (width // 4, height // 4),
        interpolation=cv2.INTER_CUBIC
    )

    # bicubic upscale baseline
    bicubic_frame = cv2.resize(
        lr_frame,
        (width, height),
        interpolation=cv2.INTER_CUBIC
    )

    enh_frame = cv2.resize(
        enh_frame,
        (width, height)
    )

    cv2.putText(
        bicubic_frame,
        "Bicubic",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.putText(
        enh_frame,
        "ESPCN",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    combined = np.hstack(
        (
            bicubic_frame,
            enh_frame
        )
    )

    writer.write(combined)

writer.release()

print(
    "Comparison video created."
)