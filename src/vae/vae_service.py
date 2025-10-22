import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image

# -------------------------------
# تعریف مدل ساده VAE
# -------------------------------
class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),  # 64→32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 32→16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # 16→8
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

# -------------------------------
# آماده‌سازی ورودی
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = VAE(latent_dim=128).to(_device)
_model.eval()

# -------------------------------
# encode_image برای pipeline
# -------------------------------
def encode_image(np_img):
    """
    np_img: numpy array [H,W] یا [H,W,C]
    خروجی: feature vector از اندازه (latent_dim,)
    """
    if np_img is None:
        raise ValueError("Input image is None")
    if not isinstance(np_img, np.ndarray):
        raise ValueError("Input must be numpy array")

    if np_img.ndim == 2:
        img = Image.fromarray(np_img.astype(np.uint8), mode="L")
    else:
        img = Image.fromarray(np_img.astype(np.uint8))

    x = transform(img).unsqueeze(0).to(_device)
    with torch.no_grad():
        mu, logvar = _model.encode(x)
        z = _model.reparameterize(mu, logvar)
    return z.cpu().numpy().flatten()
