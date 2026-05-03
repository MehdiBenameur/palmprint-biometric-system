from pathlib import Path
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AI_ROOT = PROJECT_ROOT / "ai-service"
EMB_DIR = AI_ROOT / "outputs" / "embeddings"

def l2_normalize(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

def main():
    gallery = np.load(EMB_DIR / "gallery.npy", allow_pickle=True)
    query = np.load(EMB_DIR / "query.npy", allow_pickle=True)

    palm_ids = sorted(set(int(g["palm_id"]) for g in gallery))

    prototypes = []
    prototype_ids = []

    for palm_id in palm_ids:
        embs = np.array([g["embedding"] for g in gallery if int(g["palm_id"]) == palm_id])
        proto = embs.mean(axis=0)
        prototypes.append(proto)
        prototype_ids.append(palm_id)

    prototypes = l2_normalize(np.array(prototypes))
    prototype_ids = np.array(prototype_ids)

    query_embeddings = l2_normalize(np.array([q["embedding"] for q in query]))
    query_ids = np.array([int(q["palm_id"]) for q in query])

    scores = query_embeddings @ prototypes.T
    best_indices = np.argmax(scores, axis=1)
    predicted_ids = prototype_ids[best_indices]

    correct = predicted_ids == query_ids
    accuracy = correct.mean()

    print(f"Total queries: {len(query_ids)}")
    print(f"Correct: {correct.sum()}")
    print(f"Accuracy: {accuracy:.4f}")

    top5_indices = np.argsort(scores, axis=1)[:, -5:]
    top5_ids = prototype_ids[top5_indices]
    top5_correct = np.array([
        query_ids[i] in top5_ids[i]
        for i in range(len(query_ids))
    ])

    print(f"Top-5 Accuracy: {top5_correct.mean():.4f}")

if __name__ == "__main__":
    main()