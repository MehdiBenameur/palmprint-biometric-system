from pathlib import Path
import sys
import cv2
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AI_ROOT = PROJECT_ROOT / "ai-service"
sys.path.append(str(AI_ROOT / "src"))

from roi_extractor import PalmRoiExtractor

DATASET_ROOT = PROJECT_ROOT / "data" / "Tongji"
INDEX_CSV = DATASET_ROOT / "tongji_index.csv"

OUTPUT_ROOT = AI_ROOT / "outputs" / "roi_dataset_v3"

def main():
    df = pd.read_csv(INDEX_CSV)

    extractor = PalmRoiExtractor(output_size=128)

    success = 0
    failed = 0

    for row in tqdm(df.itertuples(), total=len(df)):
        image = cv2.imread(row.image_path)

        try:
            result = extractor.extract(image)

            save_dir = OUTPUT_ROOT / row.session / f"palm_{row.palm_id:03d}"
            save_dir.mkdir(parents=True, exist_ok=True)

            save_path = save_dir / f"{row.sample_index:02d}.bmp"
            cv2.imwrite(str(save_path), result["roi"])

            success += 1

        except Exception as e:
            failed += 1

    print("\nROI dataset generation finished")
    print(f"Success: {success}")
    print(f"Failed: {failed}")

if __name__ == "__main__":
    main()