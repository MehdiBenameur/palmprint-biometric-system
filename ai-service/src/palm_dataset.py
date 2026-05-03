from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class PalmRoiDataset(Dataset):
    def __init__(self, roi_root, session, transform=None):
        self.roi_root = Path(roi_root)
        self.session = session
        self.transform = transform
        self.samples = []

        session_dir = self.roi_root / session

        if not session_dir.exists():
            raise FileNotFoundError(f"Session folder not found: {session_dir}")

        palm_dirs = sorted(session_dir.glob("palm_*"))

        for palm_dir in palm_dirs:
            palm_id = int(palm_dir.name.split("_")[1])
            label = palm_id - 1

            for img_path in sorted(palm_dir.glob("*.bmp")):
                self.samples.append((img_path, label, palm_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, palm_id = self.samples[idx]

        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, label, palm_id