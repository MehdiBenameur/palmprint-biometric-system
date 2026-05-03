from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AI_ROOT = PROJECT_ROOT / "ai-service"
sys.path.append(str(AI_ROOT / "src"))

from embedding_extractor import EmbeddingExtractor

ROI_ROOT = AI_ROOT / "outputs" / "roi_dataset_v1"
OUTPUT_DIR = AI_ROOT / "outputs" / "embeddings"

def main():
    extractor = EmbeddingExtractor(device="cpu")

    gallery_embeddings = []
    query_embeddings = []

    for session in ["session1", "session2"]:
        session_path = ROI_ROOT / session

        for palm_dir in tqdm(list(session_path.iterdir()), desc=session):
            palm_id = int(palm_dir.name.split("_")[1])

            for img_path in palm_dir.iterdir():
                emb = extractor.extract(str(img_path))

                record = {
                    "palm_id": palm_id,
                    "embedding": emb
                }

                if session == "session1":
                    gallery_embeddings.append(record)
                else:
                    query_embeddings.append(record)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_DIR / "gallery.npy", gallery_embeddings)
    np.save(OUTPUT_DIR / "query.npy", query_embeddings)

    print("Embeddings saved.")

if __name__ == "__main__":
    main()