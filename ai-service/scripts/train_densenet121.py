from pathlib import Path
import sys
import torch
import torch.nn as nn
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
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def build_model():
    weights = models.DenseNet121_Weights.IMAGENET1K_V1
    model = models.densenet121(weights=weights)

    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, NUM_CLASSES)

    return model

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels, _ in tqdm(loader, desc="Train"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="Eval"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print("Device:", device)

    train_transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.Grayscale(3),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_dataset = PalmRoiDataset(ROI_ROOT, "session1", transform=train_transform)
    test_dataset = PalmRoiDataset(ROI_ROOT, "session2", transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = build_model().to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_acc = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(f"Train Acc: {train_acc:.4f}")
        print(f"Test Acc:  {test_acc:.4f}")

        checkpoint = {
            "model_state_dict": model.state_dict(),
        }

        torch.save(checkpoint, MODEL_DIR / "densenet_last.pth")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(checkpoint, MODEL_DIR / "densenet_best.pth")
            print("New best model saved")

    print("Best accuracy:", best_acc)

if __name__ == "__main__":
    main()