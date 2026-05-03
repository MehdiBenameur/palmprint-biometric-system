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

ROI_ROOT = AI_ROOT / "outputs" / "roi_dataset_v1"
MODEL_DIR = AI_ROOT / "outputs" / "models"

NUM_CLASSES = 600
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def build_model():
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    model = models.mobilenet_v2(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

    return model

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in tqdm(loader, desc="Train"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="Eval"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_dataset = PalmRoiDataset(ROI_ROOT, "session1", transform=train_transform)
    test_dataset = PalmRoiDataset(ROI_ROOT, "session2", transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    model = build_model().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.4f}")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "test_acc": test_acc,
            "num_classes": NUM_CLASSES,
        }

        torch.save(checkpoint, MODEL_DIR / "mobilenet_last.pth")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(checkpoint, MODEL_DIR / "mobilenet_best.pth")
            print(f"New best model saved with accuracy: {best_acc:.4f}")

    print(f"\nTraining finished. Best test accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()