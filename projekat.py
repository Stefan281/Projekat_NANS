import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 8
num_epochs = 10
learning_rate = 0.001
image_size = 128

# ------------------------------
# Transformacija slika
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

# ------------------------------
# Funkcija za izbor nasumičnih slika
# ------------------------------
def sample_images_from_folder(folder_path, num_samples):
    all_images = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                  if f.endswith(('.png', '.jpg', '.jpeg'))]
    sampled_images = random.sample(all_images, min(num_samples, len(all_images)))
    return sampled_images

# ------------------------------
# Custom dataset (čiste slike)
# ------------------------------
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0  # labela nije bitna

# ------------------------------
# Dataset koji dodaje šum
# ------------------------------
class NoisyDataset(Dataset):
    def __init__(self, clean_dataset, noise_level=0.1):
        self.clean_dataset = clean_dataset
        self.noise_level = noise_level

    def __len__(self):
        return len(self.clean_dataset)

    def __getitem__(self, idx):
        clean_img, _ = self.clean_dataset[idx]
        noise = torch.randn_like(clean_img) * self.noise_level
        noisy_img = torch.clamp(clean_img + noise, 0., 1.)
        return noisy_img, clean_img

# ------------------------------
# Autoencoder model
# ------------------------------
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ------------------------------
# DnCNN model
# ------------------------------
class DnCNN(nn.Module):
    def __init__(self, depth=10, n_channels=32, image_channels=3):
        super(DnCNN, self).__init__()
        layers = []
        # Prvi sloj
        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels,
                                kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        # Srednji slojevi
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels,
                                    kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(n_channels))
            layers.append(nn.ReLU(inplace=True))
        # Poslednji sloj
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels,
                                kernel_size=3, padding=1, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.dncnn(x)
        return x - noise  # čista slika = ulaz - procenjeni šum

# ------------------------------
# Funkcije za trening i evaluaciju
# ------------------------------
def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for noisy_imgs, clean_imgs in dataloader:
        noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
        optimizer.zero_grad()
        outputs = model(noisy_imgs)
        loss = criterion(outputs, clean_imgs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * noisy_imgs.size(0)
    return running_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for noisy_imgs, clean_imgs in dataloader:
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
            outputs = model(noisy_imgs)
            loss = criterion(outputs, clean_imgs)
            total_loss += loss.item() * noisy_imgs.size(0)
    return total_loss / len(dataloader.dataset)

# ------------------------------
# Učitavanje podataka
# ------------------------------
train_folder = 'cifar10-128/train'
test_folder = 'cifar10-128/test'

train_image_paths = []
val_image_paths = []

# Train slike
for class_name in sorted(os.listdir(train_folder)):
    class_path = os.path.join(train_folder, class_name)
    train_image_paths.extend(sample_images_from_folder(class_path, 50))

# Validation slike
for class_name in sorted(os.listdir(test_folder)):
    class_path = os.path.join(test_folder, class_name)
    val_image_paths.extend(sample_images_from_folder(class_path, 10))

# Dataseti
train_dataset = CustomImageDataset(train_image_paths, transform=transform)
val_dataset = CustomImageDataset(val_image_paths, transform=transform)
test_dataset = CustomImageDataset(
    [os.path.join(dp, f) for dp, dn, fn in os.walk(test_folder)
     for f in fn if f.endswith(('.png', '.jpg', '.jpeg'))],
    transform=transform
)

# Dodavanje šuma
train_noisy_dataset = NoisyDataset(train_dataset)
val_noisy_dataset = NoisyDataset(val_dataset)
test_noisy_dataset = NoisyDataset(test_dataset)

# DataLoaders
train_loader = DataLoader(train_noisy_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_noisy_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_noisy_dataset, batch_size=batch_size, shuffle=False)

# ------------------------------
# Inicijalizacija modela
# ------------------------------
autoencoder = DenoisingAutoencoder().to(device)
dncnn = DnCNN().to(device)

criterion = nn.MSELoss()
optimizer_auto = optim.Adam(autoencoder.parameters(), lr=learning_rate)
optimizer_dncnn = optim.Adam(dncnn.parameters(), lr=learning_rate)

# ------------------------------
# Trening oba modela
# ------------------------------
for epoch in range(num_epochs):
    # Autoencoder
    train_loss_auto = train_one_epoch(autoencoder, train_loader, criterion, optimizer_auto)
    val_loss_auto = evaluate(autoencoder, val_loader, criterion)

    # DnCNN
    train_loss_dncnn = train_one_epoch(dncnn, train_loader, criterion, optimizer_dncnn)
    val_loss_dncnn = evaluate(dncnn, val_loader, criterion)

    print(f"Epoch {epoch+1}/{num_epochs} "
          f"- AE Train {train_loss_auto:.4f}, Val {val_loss_auto:.4f} "
          f"- DnCNN Train {train_loss_dncnn:.4f}, Val {val_loss_dncnn:.4f}")

# ------------------------------
# Funkcija za prikaz rezultata
# ------------------------------
def show_denoised_images(autoencoder, dncnn, test_folder):
    autoencoder.eval()
    dncnn.eval()
    class_folders = sorted([d for d in os.listdir(test_folder)
                            if os.path.isdir(os.path.join(test_folder, d))])
    num_classes = len(class_folders)

    fig, axs = plt.subplots(num_classes, 4, figsize=(12, num_classes))
    axs = np.atleast_2d(axs)

    for i, class_name in enumerate(class_folders):
        class_path = os.path.join(test_folder, class_name)
        img_files = sorted([f for f in os.listdir(class_path)
                            if f.endswith(('.png', '.jpg', '.jpeg'))])
        if len(img_files) == 0:
            continue
        img_path = os.path.join(class_path, img_files[0])

        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        noise = torch.randn_like(img_tensor) * 0.1
        noisy_img = torch.clamp(img_tensor + noise, 0., 1.)

        with torch.no_grad():
            denoised_auto = autoencoder(noisy_img).squeeze(0).cpu()
            denoised_dncnn = dncnn(noisy_img).squeeze(0).cpu()

        # Pretvaranje u numpy
        noisy_np = (np.clip(noisy_img.squeeze(0).cpu().numpy().transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
        denoised_auto_np = (np.clip(denoised_auto.numpy().transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
        denoised_dncnn_np = (np.clip(denoised_dncnn.numpy().transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
        original_np = (np.clip(img_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)

        axs[i, 0].imshow(noisy_np)
        axs[i, 1].imshow(denoised_auto_np)
        axs[i, 2].imshow(denoised_dncnn_np)
        axs[i, 3].imshow(original_np)

        if i == 0:
            axs[i, 0].set_title("Noisy")
            axs[i, 1].set_title("Denoised (Autoencoder)")
            axs[i, 2].set_title("Denoised (DnCNN)")
            axs[i, 3].set_title("Original")

        for j in range(4):
            axs[i, j].axis("off")

    plt.tight_layout()
    plt.show()

# ------------------------------
# Poziv funkcije za prikaz
# ------------------------------
show_denoised_images(autoencoder, dncnn, test_folder)