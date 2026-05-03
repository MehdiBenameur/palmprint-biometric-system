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
EPOCHS = 20
LR = 1e-4

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def build_model():
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, NUM_CLASSES)
    )
    return model

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for imgs, labels, _ in tqdm(loader, desc="Train" if train else "Eval"):
            imgs, labels = imgs.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            out = model(imgs)
            loss = criterion(out, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            pred = out.argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    return total_loss/total, correct/total

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print("Device:", device)

    train_tf = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.Grayscale(3),
        transforms.RandomApply([transforms.RandomRotation(8)], p=0.7),
        transforms.RandomApply([transforms.RandomAffine(0, translate=(0.05,0.05), scale=(0.92,1.08), shear=3)], p=0.7),
        transforms.RandomApply([transforms.ColorJitter(0.25,0.25)], p=0.6),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3,[0.5]*3),
    ])

    test_tf = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3,[0.5]*3),
    ])

    train_ds = PalmRoiDataset(ROI_ROOT, "session1", train_tf)
    test_ds  = PalmRoiDataset(ROI_ROOT, "session2", test_tf)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = build_model().to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)

    best = 0.0
    for ep in range(1, EPOCHS+1):
        print(f"\nEpoch {ep}/{EPOCHS} | LR {optim.param_groups[0]['lr']:.6f}")
        tr_l, tr_a = run_epoch(model, train_dl, criterion, optim, device, True)
        te_l, te_a = run_epoch(model, test_dl,  criterion, optim, device, False)
        sched.step()

        print(f"Train: loss {tr_l:.4f} acc {tr_a:.4f}")
        print(f"Test : loss {te_l:.4f} acc {te_a:.4f}")

        ckpt = {"model_state_dict": model.state_dict(), "test_acc": te_a}
        torch.save(ckpt, MODEL_DIR/"resnet18_last.pth")

        if te_a > best:
            best = te_a
            torch.save(ckpt, MODEL_DIR/"resnet18_best.pth")
            print("New best:", best)

    print("Best test acc:", best)

if __name__ == "__main__":
    main()