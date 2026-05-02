from pathlib import Path
import sys
import cv2
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AI_ROOT = PROJECT_ROOT / "ai-service"
sys.path.append(str(AI_ROOT / "src"))

from roi_extractor import PalmRoiExtractor

DATASET_ROOT = PROJECT_ROOT / "data" / "Tongji"
INDEX_CSV = DATASET_ROOT / "tongji_index.csv"
OUTPUT_DIR = AI_ROOT / "outputs" / "roi_v1"

def draw_debug(image_bgr, result):
    debug = result["debug"]
    display = image_bgr.copy()

    x, y, w, h = debug["hand_bbox"]
    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 3)

    px, py, pw, ph = debug["palm_bbox_inside_hand"]
    cv2.rectangle(display, (x + px, y + py), (x + px + pw, y + py + ph), (0, 0, 255), 3)

    return display

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INDEX_CSV)
    samples = df[df["session"] == "session1"].head(12)

    extractor = PalmRoiExtractor(output_size=128)

    plt.figure(figsize=(12, 8))

    for i, row in enumerate(samples.itertuples(), start=1):
        image = cv2.imread(row.image_path)
        result = extractor.extract(image)

        roi_path = OUTPUT_DIR / f"roi_{row.file_name.rsplit('.', 1)[0]}.bmp"
        cv2.imwrite(str(roi_path), result["roi"])

        debug_img = draw_debug(image, result)
        debug_rgb = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)

        plt.subplot(3, 4, i)
        plt.imshow(debug_rgb)
        plt.title(f"Palm {row.palm_id} / sample {row.sample_index}")
        plt.axis("off")

    debug_grid_path = OUTPUT_DIR / "roi_debug_grid.png"
    plt.tight_layout()
    plt.savefig(debug_grid_path, dpi=200)
    plt.close()

    print(f"ROI outputs saved in: {OUTPUT_DIR}")
    print(f"Debug grid saved: {debug_grid_path}")
    print("Generated 12 ROI images.")

if __name__ == "__main__":
    main()