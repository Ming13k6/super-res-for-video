import torch
from PIL import Image
import torchvision.transforms as transforms
from PIL import ImageFilter
from model import ESPCN

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

model = ESPCN(scale_factor=4).to(device)

model.load_state_dict(
    torch.load("espcn.pth")
)

model.eval()

image_path = "dataset/lr/84353.jpg"

image = Image.open(image_path).convert("RGB")

transform = transforms.ToTensor()

input_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():

    output = model(input_tensor)

output_image = output.squeeze(0).cpu().clamp(0, 1)

output_image = transforms.ToPILImage()(output_image)

output_image = output_image.filter(
    ImageFilter.SHARPEN
)

output_image.save("output/output.png")

print("Inference complete.")