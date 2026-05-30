import cv2
import os
import numpy as np

original_folder = "video_frames"
enhanced_folder = "enhanced_frames"

output_path = "output/enhanced_video.mp4"

fps = 30

original_images = sorted(
    os.listdir(original_folder)
)

enhanced_images = sorted(
    os.listdir(enhanced_folder)
)

# original size
first_original = cv2.imread(
    os.path.join(
        original_folder,
        original_images[0]
    )
)

orig_height, orig_width, _ = first_original.shape

writer = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (orig_width, orig_height)
)

print(
    "Writer opened:",
    writer.isOpened()
)

for enh in enhanced_images:

    enh_frame = cv2.imread(
        os.path.join(
            enhanced_folder,
            enh
        )
    )

    if enh_frame is None:

        print(
            f"Skipping {enh}"
        )

        continue

    # resize BACK to playable resolution
    enh_frame = cv2.resize(
        enh_frame,
        (orig_width, orig_height)
    )

    enh_frame = np.uint8(
        enh_frame
    )

    writer.write(
        enh_frame
    )

writer.release()

print(
    "Enhanced video created."
)