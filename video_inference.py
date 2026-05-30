import os
import torch
import cv2

from PIL import Image
import torchvision.transforms as transforms

from model import ESPCN

torch.backends.cudnn.benchmark = True

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(device)

model = ESPCN(scale_factor=4).to(device).half()

model.load_state_dict(
    torch.load(
        "espcn.pth",
        weights_only=True
    )
)

model.eval()

input_folder = "video_frames"
output_folder = "enhanced_frames"

# clean output folder
if os.path.exists(output_folder):

    for file in os.listdir(output_folder):

        os.remove(
            os.path.join(
                output_folder,
                file
            )
        )

os.makedirs(
    output_folder,
    exist_ok=True
)

transform = transforms.ToTensor()

images = sorted(
    os.listdir(input_folder)
)

for image_name in images:

    image_path = os.path.join(
        input_folder,
        image_name
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    input_tensor = transform(
        image
    ).unsqueeze(0).to(device).half()

    with torch.no_grad():

        output = model(input_tensor)

    output_image = (
        output.squeeze(0)
        .float()
        .cpu()
        .clamp(0,1)
        .permute(1,2,0)
        .numpy()
    )

    output_image = (
        output_image * 255
    ).astype("uint8")

    output_image = cv2.cvtColor(
        output_image,
        cv2.COLOR_RGB2BGR
    )

    save_path = os.path.join(
        output_folder,
        image_name.replace(".png", ".jpg")
    )

    cv2.imwrite(
        save_path,
        output_image
    )

print(
    "Video frame upscaling complete."
)