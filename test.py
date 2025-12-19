import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image

from models.generator import Generator

# ======================
# Config
# ======================
device = "mps" if torch.backends.mps.is_available() else "cpu"
checkpoint_path = "checkpoints/cartoongan_epoch_50.pth"
input_image_path = "test_253.png"
output_dir = "outputs_test"

os.makedirs(output_dir, exist_ok=True)

print("Using device:", device)

# ======================
# Load Model
# ======================
G = Generator().to(device)

checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)

G.load_state_dict(checkpoint["generator_state_dict"])
G.eval()

# ======================
# Preprocessing (NO RESIZE)
# ======================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])

img = Image.open(input_image_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)

print("Input image size:", img_tensor.shape)

# ======================
# Inference
# ======================
with torch.no_grad():
    fake_cartoon = G(img_tensor)

# ======================
# Save Output
# ======================
save_image(
    (fake_cartoon + 1) / 2,
    os.path.join(output_dir, "cartoon.png")
)

print("Hasil kartun disimpan di outputs_test/cartoon.png")
