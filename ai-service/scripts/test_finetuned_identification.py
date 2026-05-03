from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AI_ROOT = PROJECT_ROOT / "ai-service"
EMB_DIR = AI_ROOT / "outputs" / "finetuned_embeddings"

def l2_normalize(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

def main():
    gallery = np.load(EMB_DIR / "gallery.npy", allow_pickle=True)
    query = np.load(EMB_DIR / "query.npy", allow_pickle=True)

    gallery_embeddings = l2_normalize(np.array([g["embedding"] for g in gallery]))
    gallery_ids = np.array([int(g["palm_id"]) for g in gallery])

    query_embeddings = l2_normalize(np.array([q["embedding"] for q in query]))
    query_ids = np.array([int(q["palm_id"]) for q in query])

    scores = query_embeddings @ gallery_embeddings.T

    top1_indices = np.argmax(scores, axis=1)
    top1_ids = gallery_ids[top1_indices]
    top1_scores = scores[np.arange(len(query_ids)), top1_indices]

    top1_acc = (top1_ids == query_ids).mean()

    top3_indices = np.argsort(scores, axis=1)[:, -3:]
    top5_indices = np.argsort(scores, axis=1)[:, -5:]
    top10_indices = np.argsort(scores, axis=1)[:, -10:]

    top3_acc = np.mean([query_ids[i] in gallery_ids[top3_indices[i]] for i in range(len(query_ids))])
    top5_acc = np.mean([query_ids[i] in gallery_ids[top5_indices[i]] for i in range(len(query_ids))])
    top10_acc = np.mean([query_ids[i] in gallery_ids[top10_indices[i]] for i in range(len(query_ids))])

    print(f"Total queries: {len(query_ids)}")
    print(f"Top-1 Accuracy:  {top1_acc:.4f}")
    print(f"Top-3 Accuracy:  {top3_acc:.4f}")
    print(f"Top-5 Accuracy:  {top5_acc:.4f}")
    print(f"Top-10 Accuracy: {top10_acc:.4f}")
    print()
    print(f"Mean Top-1 Score: {top1_scores.mean():.4f}")
    print(f"Min Top-1 Score:  {top1_scores.min():.4f}")
    print(f"Max Top-1 Score:  {top1_scores.max():.4f}")

if __name__ == "__main__":
    main()