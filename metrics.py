import cv2
import os

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

original_folder = "video_frames"
enhanced_folder = "enhanced_frames"

original_images = sorted(
    os.listdir(original_folder)
)

enhanced_images = sorted(
    os.listdir(enhanced_folder)
)

total_psnr = 0
total_ssim = 0

count = 0

for orig, enh in zip(
        original_images,
        enhanced_images):

    orig_img = cv2.imread(
        os.path.join(
            original_folder,
            orig
        )
    )

    enh_img = cv2.imread(
        os.path.join(
            enhanced_folder,
            enh
        )
    )

    enh_img = cv2.resize(
        enh_img,
        (
            orig_img.shape[1],
            orig_img.shape[0]
        )
    )

    psnr = peak_signal_noise_ratio(
        orig_img,
        enh_img
    )

    ssim = structural_similarity(
        orig_img,
        enh_img,
        channel_axis=2
    )

    total_psnr += psnr
    total_ssim += ssim

    count += 1

avg_psnr = total_psnr / count
avg_ssim = total_ssim / count

print(f"Average PSNR: {avg_psnr:.2f}")
print(f"Average SSIM: {avg_ssim:.4f}")