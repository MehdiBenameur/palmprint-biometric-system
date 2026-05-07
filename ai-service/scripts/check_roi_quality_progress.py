from pathlib import Path
import cv2
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data" / "Tongji"
ROI_ROOT = PROJECT_ROOT / "ai-service" / "outputs" / "roi_dataset_v1"
INDEX_CSV = DATASET_ROOT / "tongji_index.csv"
OUT = PROJECT_ROOT / "ai-service" / "outputs" / "roi_quality_check.png"

def quality(gray):
    return {
        "blur": round(cv2.Laplacian(gray, cv2.CV_64F).var(), 2),
        "brightness": round(gray.mean(), 2),
        "contrast": round(gray.std(), 2),
    }

df = pd.read_csv(INDEX_CSV)
samples = df[(df["session"] == "session1") & (df["sample_index"] == 1)].iloc[0:12]

plt.figure(figsize=(14, 8))

for i, row in enumerate(samples.itertuples(), start=1):
    raw = cv2.imread(row.image_path, cv2.IMREAD_GRAYSCALE)
    roi_path = ROI_ROOT / "session1" / f"palm_{row.palm_id:03d}" / "01.bmp"
    roi = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)

    q = quality(roi)

    plt.subplot(3, 4, i)
    plt.imshow(roi, cmap="gray")
    plt.title(f"Palm {row.palm_id}\nB:{q['blur']} C:{q['contrast']}")
    plt.axis("off")

plt.tight_layout()
plt.savefig(OUT, dpi=200)
print(f"Saved: {OUT}")