import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

class EmbeddingExtractor:
    def __init__(self, device="cpu"):
        self.device = device

        model = models.mobilenet_v2(pretrained=True)
        self.model = nn.Sequential(*list(model.features), nn.AdaptiveAvgPool2d(1))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])

    def extract(self, image_path):
        image = Image.open(image_path).convert("L")
        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(image)
            features = features.view(features.size(0), -1)

        embedding = features.cpu().numpy()[0]
        embedding = embedding / np.linalg.norm(embedding)

        return embedding