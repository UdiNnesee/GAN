import torch
import os
from torch.utils.data import DataLoader

from models.generator import Generator
from models.discriminator import Discriminator
from utils.dataloader import UnpairedImageDataset
from utils.losses import GANLoss, ContentLoss

# ======================
# Device
# ======================
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

# ======================
# Dataset & DataLoader
# ======================
dataset = UnpairedImageDataset(
    root="datasets/cartoon_dataset",
    mode="train"
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

# ======================
# Model
# ======================
G = Generator().to(device)
D = Discriminator().to(device)

# ======================
# Loss Function
# ======================
gan_loss = GANLoss().to(device)
content_loss = ContentLoss().to(device)

# ======================
# Optimizer
# ======================
optimizer_G = torch.optim.Adam(
    G.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)

optimizer_D = torch.optim.Adam(
    D.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)

# ======================
# Hyperparameter
# ======================
num_epochs = 50
lambda_content = 10.0

checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# ======================
# Training Loop
# ======================
for epoch in range(1, num_epochs + 1):
    for i, (real_A, real_B) in enumerate(loader):

        real_A = real_A.to(device)
        real_B = real_B.to(device)

        # ==========================
        # 1. Train Discriminator
        # ==========================
        optimizer_D.zero_grad()

        # Real cartoon
        pred_real = D(real_B)
        loss_D_real = gan_loss(pred_real, True)

        # Fake cartoon
        fake_B = G(real_A).detach()
        pred_fake = D(fake_B)
        loss_D_fake = gan_loss(pred_fake, False)

        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        optimizer_D.step()

        # ======================
        # 2. Train Generator
        # ======================
        optimizer_G.zero_grad()

        fake_B = G(real_A)
        pred_fake = D(fake_B)

        loss_G_adv = gan_loss(pred_fake, True)
        loss_G_content = content_loss(fake_B, real_A)

        loss_G = loss_G_adv + lambda_content * loss_G_content
        loss_G.backward()
        optimizer_G.step()

        # ======================
        # Logging
        # ======================
        if i % 50 == 0:
            print(
                f"[Epoch {epoch}/{num_epochs}] "
                f"[Batch {i}/{len(loader)}] "
                f"[D loss: {loss_D.item():.4f}] "
                f"[G adv: {loss_G_adv.item():.4f}] "
                f"[G content: {loss_G_content.item():.4f}]"
            )

    # ======================
    # Save checkpoint
    # ======================
    if epoch == num_epochs:
        torch.save({
            "epoch": epoch,
            "generator_state_dict": G.state_dict(),
            "discriminator_state_dict": D.state_dict(),
            "optimizer_G_state_dict": optimizer_G.state_dict(),
            "optimizer_D_state_dict": optimizer_D.state_dict()
        }, f"{checkpoint_dir}/cartoongan_epoch_{epoch}.pth")

        print(f"Checkpoint disimpan di {checkpoint_dir}/cartoongan_epoch_{epoch}.pth\n")


print("Training selesai.")
