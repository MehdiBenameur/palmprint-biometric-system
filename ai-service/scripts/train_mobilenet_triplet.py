from pathlib import Path
import sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AI_ROOT = PROJECT_ROOT / "ai-service"
sys.path.append(str(AI_ROOT / "src"))

ROI_ROOT = AI_ROOT / "outputs" / "roi_dataset_v3"
MODEL_DIR = AI_ROOT / "outputs" / "models"

NUM_CLASSES = 600
EMBEDDING_DIM = 512
BATCH_SIZE = 32
EPOCHS = 25
LR = 1e-4
MARGIN = 0.3


class TripletPalmDataset(Dataset):
    def __init__(self, roi_root, transform=None):
        self.roi_root = Path(roi_root)
        self.transform = transform

        self.session1_dir = self.roi_root / "session1"
        self.session2_dir = self.roi_root / "session2"

        self.palm_to_session1 = {}
        self.palm_to_session2 = {}

        for palm_dir in sorted(self.session1_dir.glob("palm_*")):
            palm_id = int(palm_dir.name.split("_")[1])
            self.palm_to_session1[palm_id] = sorted(palm_dir.glob("*.bmp"))

        for palm_dir in sorted(self.session2_dir.glob("palm_*")):
            palm_id = int(palm_dir.name.split("_")[1])
            self.palm_to_session2[palm_id] = sorted(palm_dir.glob("*.bmp"))

        self.palm_ids = sorted(
            list(set(self.palm_to_session1.keys()) & set(self.palm_to_session2.keys()))
        )

        self.samples = []

        for palm_id in self.palm_ids:
            for img_path in self.palm_to_session1[palm_id]:
                self.samples.append((img_path, palm_id))

    def __len__(self):
        return len(self.samples)

    def load_image(self, image_path):
        image = Image.open(image_path).convert("L")

        if self.transform:
            image = self.transform(image)

        return image

    def __getitem__(self, idx):
        anchor_path, palm_id = self.samples[idx]

        positive_path = random.choice(self.palm_to_session2[palm_id])

        negative_palm_id = random.choice(self.palm_ids)
        while negative_palm_id == palm_id:
            negative_palm_id = random.choice(self.palm_ids)

        negative_path = random.choice(self.palm_to_session2[negative_palm_id])

        anchor = self.load_image(anchor_path)
        positive = self.load_image(positive_path)
        negative = self.load_image(negative_path)

        return anchor, positive, negative, palm_id


class MobileNetTriplet(nn.Module):
    def __init__(self):
        super().__init__()

        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

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


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total = 0

    for anchor, positive, negative, _ in tqdm(loader, desc="Train"):
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        optimizer.zero_grad()

        anchor_emb = model(anchor)
        positive_emb = model(positive)
        negative_emb = model(negative)

        loss = criterion(anchor_emb, positive_emb, negative_emb)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * anchor.size(0)
        total += anchor.size(0)

    return total_loss / total


def quick_identification_eval(model, roi_root, transform, device, max_queries=1000):
    model.eval()

    gallery_embeddings = []
    gallery_ids = []

    query_embeddings = []
    query_ids = []

    session1_dir = Path(roi_root) / "session1"
    session2_dir = Path(roi_root) / "session2"

    with torch.no_grad():
        for palm_dir in tqdm(sorted(session1_dir.glob("palm_*")), desc="Gallery eval"):
            palm_id = int(palm_dir.name.split("_")[1])

            for img_path in sorted(palm_dir.glob("*.bmp")):
                image = Image.open(img_path).convert("L")
                image = transform(image).unsqueeze(0).to(device)

                embedding = model(image).cpu()

                gallery_embeddings.append(embedding)
                gallery_ids.append(palm_id)

        count = 0

        for palm_dir in tqdm(sorted(session2_dir.glob("palm_*")), desc="Query eval"):
            palm_id = int(palm_dir.name.split("_")[1])

            for img_path in sorted(palm_dir.glob("*.bmp")):
                if count >= max_queries:
                    break

                image = Image.open(img_path).convert("L")
                image = transform(image).unsqueeze(0).to(device)

                embedding = model(image).cpu()

                query_embeddings.append(embedding)
                query_ids.append(palm_id)

                count += 1

            if count >= max_queries:
                break

    gallery_embeddings = torch.cat(gallery_embeddings, dim=0)
    query_embeddings = torch.cat(query_embeddings, dim=0)

    gallery_ids = torch.tensor(gallery_ids)
    query_ids = torch.tensor(query_ids)

    scores = query_embeddings @ gallery_embeddings.T

    top1_indices = torch.argmax(scores, dim=1)
    top1_ids = gallery_ids[top1_indices]

    top1_acc = (top1_ids == query_ids).float().mean().item()

    return top1_acc


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomApply([
            transforms.RandomRotation(degrees=6)
        ], p=0.5),
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=0,
                translate=(0.03, 0.03),
                scale=(0.95, 1.05),
                shear=2
            )
        ], p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.15, contrast=0.15)
        ], p=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = TripletPalmDataset(ROI_ROOT, transform=train_transform)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    print(f"Triplet samples: {len(dataset)}")

    model = MobileNetTriplet().to(device)

    criterion = nn.TripletMarginLoss(margin=MARGIN, p=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_top1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss = train_one_epoch(model, loader, criterion, optimizer, device)

        print(f"Train Triplet Loss: {train_loss:.4f}")

        if epoch % 5 == 0 or epoch == EPOCHS:
            top1_acc = quick_identification_eval(
                model,
                ROI_ROOT,
                eval_transform,
                device,
                max_queries=1000
            )

            print(f"Quick Identification Top-1: {top1_acc:.4f}")

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "embedding_dim": EMBEDDING_DIM,
                "top1_acc": top1_acc,
            }

            torch.save(checkpoint, MODEL_DIR / "mobilenet_triplet_last.pth")

            if top1_acc > best_top1:
                best_top1 = top1_acc
                torch.save(checkpoint, MODEL_DIR / "mobilenet_triplet_best.pth")
                print(f"New best triplet model saved: {best_top1:.4f}")

    print(f"\nTraining finished. Best quick Top-1: {best_top1:.4f}")


if __name__ == "__main__":
    main()