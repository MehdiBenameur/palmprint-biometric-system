from pathlib import Path
import sys
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AI_ROOT = PROJECT_ROOT / "ai-service"
sys.path.append(str(AI_ROOT / "src"))

from finetuned_embedding_extractor import FineTunedEmbeddingExtractor
from triplet_embedding_extractor import TripletEmbeddingExtractor

CNN_MODEL_PATH = AI_ROOT / "outputs" / "models" / "mobilenet_v3_best.pth"
TRIPLET_MODEL_PATH = AI_ROOT / "outputs" / "models" / "mobilenet_triplet_best.pth"

CNN_GALLERY_PATH = AI_ROOT / "outputs" / "finetuned_embeddings_v3" / "gallery.npy"
TRIPLET_GALLERY_PATH = AI_ROOT / "outputs" / "triplet_embeddings_v3" / "gallery.npy"

ALPHA = 0.42
BETA = 0.58
TOP_K = 5
THRESHOLD = 0.65


def l2_normalize(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


class FinalPalmIdentifier:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.cnn_extractor = FineTunedEmbeddingExtractor(
            CNN_MODEL_PATH,
            device=self.device,
            num_classes=600
        )

        self.triplet_extractor = TripletEmbeddingExtractor(
            TRIPLET_MODEL_PATH,
            device=self.device
        )

        self.cnn_gallery = np.load(CNN_GALLERY_PATH, allow_pickle=True)
        self.triplet_gallery = np.load(TRIPLET_GALLERY_PATH, allow_pickle=True)

        self.gallery_ids = np.array([int(g["palm_id"]) for g in self.cnn_gallery])
        triplet_ids = np.array([int(g["palm_id"]) for g in self.triplet_gallery])

        if not np.array_equal(self.gallery_ids, triplet_ids):
            raise ValueError("CNN and Triplet gallery IDs are not aligned.")

        self.cnn_gallery_embeddings = l2_normalize(
            np.array([g["embedding"] for g in self.cnn_gallery])
        )

        self.triplet_gallery_embeddings = l2_normalize(
            np.array([g["embedding"] for g in self.triplet_gallery])
        )

    def identify(self, image_path):
        cnn_embedding = self.cnn_extractor.extract(image_path)
        triplet_embedding = self.triplet_extractor.extract(image_path)

        cnn_embedding = cnn_embedding / (np.linalg.norm(cnn_embedding) + 1e-12)
        triplet_embedding = triplet_embedding / (np.linalg.norm(triplet_embedding) + 1e-12)

        cnn_scores = self.cnn_gallery_embeddings @ cnn_embedding
        triplet_scores = self.triplet_gallery_embeddings @ triplet_embedding

        final_scores = ALPHA * cnn_scores + BETA * triplet_scores

        top_indices = np.argsort(-final_scores)[:TOP_K]

        top_candidates = []
        for idx in top_indices:
            top_candidates.append({
                "palm_id": int(self.gallery_ids[idx]),
                "score": float(final_scores[idx]),
                "cnn_score": float(cnn_scores[idx]),
                "triplet_score": float(triplet_scores[idx]),
            })

        best_candidate = top_candidates[0]
        accepted = best_candidate["score"] >= THRESHOLD

        return {
            "accepted": bool(accepted),
            "predicted_palm_id": best_candidate["palm_id"] if accepted else None,
            "final_score": best_candidate["score"],
            "threshold": THRESHOLD,
            "method": "CNN MobileNet + Triplet deep fusion",
            "alpha": ALPHA,
            "beta": BETA,
            "top_k": top_candidates,
        }