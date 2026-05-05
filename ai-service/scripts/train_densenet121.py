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
LEARNING_RATE = 1e-4

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def build_model():
    weights = models.DenseNet121_Weights.IMAGENET1K_V1
    model = models.densenet121(weights=weights)

    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, NUM_CLASSES)
    )

    return model

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels, _ in tqdm(loader, desc="Train"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="Eval"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomApply([transforms.RandomRotation(degrees=8)], p=0.7),
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.92, 1.08),
                shear=3
            )
        ], p=0.7),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.25, contrast=0.25)
        ], p=0.6),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
        ], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    train_dataset = PalmRoiDataset(ROI_ROOT, "session1", transform=train_transform)
    test_dataset = PalmRoiDataset(ROI_ROOT, "session2", transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    model = build_model().to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.8f}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.4f}")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "test_acc": test_acc,
            "num_classes": NUM_CLASSES,
            "backbone": "densenet121",
        }

        torch.save(checkpoint, MODEL_DIR / "densenet121_v3_last.pth")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(checkpoint, MODEL_DIR / "densenet121_v3_best.pth")
            print(f"New best model saved with accuracy: {best_acc:.4f}")

    print(f"\nTraining finished. Best test accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()