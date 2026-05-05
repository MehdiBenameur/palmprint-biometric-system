from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AI_ROOT = PROJECT_ROOT / "ai-service"
sys.path.append(str(AI_ROOT / "src"))

from palm_dataset import PalmRoiDataset

ROI_ROOT = AI_ROOT / "outputs" / "roi_dataset_v3"
MODEL_DIR = AI_ROOT / "outputs" / "models"

NUM_CLASSES = 600
EMBEDDING_DIM = 512
BATCH_SIZE = 32
EPOCHS = 25
LR = 1e-4

# ---------- ArcFace Layer ----------
class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.s = s
        self.m = m

    def forward(self, x, labels):
        x = F.normalize(x)
        W = F.normalize(self.weight)

        cosine = F.linear(x, W)
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.m)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1,1), 1.0)

        output = (one_hot * target_logits) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output

# ---------- Model ----------
class MobileNetArcFace(nn.Module):
    def __init__(self):
        super().__init__()

        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.embedding = nn.Linear(1280, EMBEDDING_DIM)
        self.arcface = ArcFace(EMBEDDING_DIM, NUM_CLASSES)

    def forward(self, x, labels=None):
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)

        emb = self.embedding(x)

        if labels is not None:
            logits = self.arcface(emb, labels)
            return logits, emb

        return emb

# ---------- Training ----------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels, _ in tqdm(loader, desc="Train"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits, _ = model(images, labels)
        loss = F.cross_entropy(logits, labels)

        loss.backward()
        optimizer.step()

        preds = logits.argmax(1)

        total_loss += loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits, _ = model(images, labels)

            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

# ---------- Main ----------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_dataset = PalmRoiDataset(ROI_ROOT, "session1", transform)
    test_dataset = PalmRoiDataset(ROI_ROOT, "session2", transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = MobileNetArcFace().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_acc = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        test_acc = evaluate(model, test_loader, device)

        print(f"Train Acc: {train_acc:.4f}")
        print(f"Test Acc:  {test_acc:.4f}")

        checkpoint = {
            "model_state_dict": model.state_dict()
        }

        torch.save(checkpoint, MODEL_DIR / "mobilenet_arcface_last.pth")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(checkpoint, MODEL_DIR / "mobilenet_arcface_best.pth")
            print("New best model saved")

    print("Best accuracy:", best_acc)

if __name__ == "__main__":
    main()