import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random

class UnpairedImageDataset(Dataset):
    def __init__(self, root, mode="train"):
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3)
        ])

        self.files_A = sorted(os.listdir(os.path.join(root, f"{mode}A")))
        self.files_B = sorted(os.listdir(os.path.join(root, f"{mode}B")))

        self.root = root
        self.mode = mode

    def __getitem__(self, index):
        img_A = Image.open(
            os.path.join(self.root, f"{self.mode}A",
                         self.files_A[index % len(self.files_A)])
        ).convert("RGB")

        img_B = Image.open(
            os.path.join(self.root, f"{self.mode}B",
                         random.choice(self.files_B))
        ).convert("RGB")

        return self.transform(img_A), self.transform(img_B)

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
