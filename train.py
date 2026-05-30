import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

from model import ESPCN
from dataset import SRDataset
from torch.utils.data import DataLoader

os.makedirs(
    "checkpoints",
    exist_ok=True
)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

dataset = SRDataset(
    hr_dir="dataset/hr"
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

model = ESPCN(scale_factor=4).to(device)

def hybrid_loss(output, target):

    l1 = F.l1_loss(output, target)

    mse = F.mse_loss(output, target)

    return l1 + 0.1 * mse

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0005
)

epochs = 200

for epoch in range(epochs):

    total_loss = 0

    for lr_imgs, hr_imgs in loader:

        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)

        outputs = model(lr_imgs)

        loss = hybrid_loss(outputs, hr_imgs)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {avg_loss:.6f}"
    )

    if (epoch + 1) % 20 == 0:

        torch.save(
            model.state_dict(),
            f"checkpoints/espcn_epoch_{epoch+1}.pth"
       )

torch.save(
    model.state_dict(),
    "espcn.pth"
)

print("Model saved.")