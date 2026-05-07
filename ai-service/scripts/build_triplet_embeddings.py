from pathlib import Path
import sys
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AI_ROOT = PROJECT_ROOT / "ai-service"
sys.path.append(str(AI_ROOT / "src"))

from triplet_embedding_extractor import TripletEmbeddingExtractor

ROI_ROOT = AI_ROOT / "outputs" / "roi_dataset_v3"
MODEL_PATH = AI_ROOT / "outputs" / "models" / "mobilenet_triplet_best.pth"
OUTPUT_DIR = AI_ROOT / "outputs" / "triplet_embeddings_v3"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    extractor = TripletEmbeddingExtractor(
        model_path=MODEL_PATH,
        device=device
    )

    gallery = []
    query = []

    for session in ["session1", "session2"]:
        session_path = ROI_ROOT / session

        for palm_dir in tqdm(sorted(session_path.iterdir()), desc=session):
            if not palm_dir.is_dir():
                continue

            palm_id = int(palm_dir.name.split("_")[1])

            for img_path in sorted(palm_dir.glob("*.bmp")):
                embedding = extractor.extract(img_path)

                record = {
                    "palm_id": palm_id,
                    "image_path": str(img_path),
                    "embedding": embedding,
                }

                if session == "session1":
                    gallery.append(record)
                else:
                    query.append(record)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_DIR / "gallery.npy", gallery)
    np.save(OUTPUT_DIR / "query.npy", query)

    print("Triplet embeddings saved.")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Gallery: {len(gallery)}")
    print(f"Query: {len(query)}")

if __name__ == "__main__":
    main()