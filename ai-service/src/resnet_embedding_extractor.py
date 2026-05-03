import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class ResNetEmbeddingExtractor:
    def __init__(self, model_path, device="cpu", num_classes=600):
        self.device = device

        self.model = models.resnet18(weights=None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()

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
            features = self.feature_extractor(image)
            features = features.view(features.size(0), -1)

        embedding = features.cpu().numpy()[0]
        embedding = embedding / (np.linalg.norm(embedding) + 1e-12)

        return embedding