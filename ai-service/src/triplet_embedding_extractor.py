import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

EMBEDDING_DIM = 512

class MobileNetTriplet(nn.Module):
    def __init__(self):
        super().__init__()

        base = models.mobilenet_v2(weights=None)

        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.embedding = nn.Sequential(
            nn.Linear(1280, EMBEDDING_DIM),
            nn.BatchNorm1d(EMBEDDING_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.embedding(x)
        x = F.normalize(x, p=2, dim=1)
        return x

class TripletEmbeddingExtractor:
    def __init__(self, model_path, device="cpu"):
        self.device = device

        self.model = MobileNetTriplet()

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def extract(self, image_path):
        image = Image.open(image_path).convert("L")
        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(image)

        embedding = embedding.cpu().numpy()[0]
        embedding = embedding / (np.linalg.norm(embedding) + 1e-12)

        return embedding