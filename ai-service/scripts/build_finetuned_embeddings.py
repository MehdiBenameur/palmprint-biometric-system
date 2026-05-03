from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AI_ROOT = PROJECT_ROOT / "ai-service"
sys.path.append(str(AI_ROOT / "src"))

from finetuned_embedding_extractor import FineTunedEmbeddingExtractor

ROI_ROOT = AI_ROOT / "outputs" / "roi_dataset_v1"
MODEL_PATH = AI_ROOT / "outputs" / "models" / "mobilenet_improved_best.pth"
OUTPUT_DIR = AI_ROOT / "outputs" / "finetuned_embeddings"

def main():
    device = "cuda" if False else "cpu"
    extractor = FineTunedEmbeddingExtractor(MODEL_PATH, device=device)

    gallery = []
    query = []

    for session in ["session1", "session2"]:
        session_path = ROI_ROOT / session

        for palm_dir in tqdm(sorted(session_path.iterdir()), desc=session):
            palm_id = int(palm_dir.name.split("_")[1])

            for img_path in sorted(palm_dir.glob("*.bmp")):
                emb = extractor.extract(str(img_path))

                record = {
                    "palm_id": palm_id,
                    "image_path": str(img_path),
                    "embedding": emb,
                }

                if session == "session1":
                    gallery.append(record)
                else:
                    query.append(record)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "gallery.npy", gallery)
    np.save(OUTPUT_DIR / "query.npy", query)

    print("Fine-tuned embeddings saved.")
    print(f"Gallery: {len(gallery)}")
    print(f"Query: {len(query)}")

if __name__ == "__main__":
    main()