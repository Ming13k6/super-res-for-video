import cv2
import os

hr_folder = "dataset/hr"
lr_folder = "dataset/lr"

scale_factor = 4

os.makedirs(lr_folder, exist_ok=True)

for image_name in os.listdir(hr_folder):

    image_path = os.path.join(
        hr_folder,
        image_name
    )

    image = cv2.imread(image_path)

    if image is None:
        continue

    height, width = image.shape[:2]

    lr_image = cv2.resize(
        image,
        (width // scale_factor,
         height // scale_factor),
        interpolation=cv2.INTER_CUBIC
    )

    save_path = os.path.join(
        lr_folder,
        image_name
    )

    cv2.imwrite(save_path, lr_image)

print("LR images generated.")