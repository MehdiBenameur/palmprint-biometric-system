from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AI_ROOT = PROJECT_ROOT / "ai-service"

CNN_EMB_DIR = AI_ROOT / "outputs" / "finetuned_embeddings_v3"
TRIPLET_EMB_DIR = AI_ROOT / "outputs" / "triplet_embeddings_v3"

ALPHA = 0.42  # CNN score
BETA = 0.58   # Triplet score

def l2_normalize(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

def compute_topk(scores, gallery_ids, query_ids):
    top1_indices = np.argmax(scores, axis=1)
    top1_ids = gallery_ids[top1_indices]
    top1_scores = scores[np.arange(len(query_ids)), top1_indices]

    top3_indices = np.argsort(-scores, axis=1)[:, :3]
    top5_indices = np.argsort(-scores, axis=1)[:, :5]
    top10_indices = np.argsort(-scores, axis=1)[:, :10]

    top1_acc = (top1_ids == query_ids).mean()

    top3_acc = np.mean([
        query_ids[i] in gallery_ids[top3_indices[i]]
        for i in range(len(query_ids))
    ])

    top5_acc = np.mean([
        query_ids[i] in gallery_ids[top5_indices[i]]
        for i in range(len(query_ids))
    ])

    top10_acc = np.mean([
        query_ids[i] in gallery_ids[top10_indices[i]]
        for i in range(len(query_ids))
    ])

    return top1_acc, top3_acc, top5_acc, top10_acc, top1_scores

def main():
    cnn_gallery = np.load(CNN_EMB_DIR / "gallery.npy", allow_pickle=True)
    cnn_query = np.load(CNN_EMB_DIR / "query.npy", allow_pickle=True)

    triplet_gallery = np.load(TRIPLET_EMB_DIR / "gallery.npy", allow_pickle=True)
    triplet_query = np.load(TRIPLET_EMB_DIR / "query.npy", allow_pickle=True)

    cnn_gallery_ids = np.array([int(g["palm_id"]) for g in cnn_gallery])
    cnn_query_ids = np.array([int(q["palm_id"]) for q in cnn_query])

    triplet_gallery_ids = np.array([int(g["palm_id"]) for g in triplet_gallery])
    triplet_query_ids = np.array([int(q["palm_id"]) for q in triplet_query])

    if not np.array_equal(cnn_gallery_ids, triplet_gallery_ids):
        raise ValueError("Gallery IDs mismatch between CNN and Triplet embeddings.")

    if not np.array_equal(cnn_query_ids, triplet_query_ids):
        raise ValueError("Query IDs mismatch between CNN and Triplet embeddings.")

    gallery_ids = cnn_gallery_ids
    query_ids = cnn_query_ids

    cnn_gallery_embeddings = l2_normalize(np.array([g["embedding"] for g in cnn_gallery]))
    cnn_query_embeddings = l2_normalize(np.array([q["embedding"] for q in cnn_query]))

    triplet_gallery_embeddings = l2_normalize(np.array([g["embedding"] for g in triplet_gallery]))
    triplet_query_embeddings = l2_normalize(np.array([q["embedding"] for q in triplet_query]))

    cnn_scores = cnn_query_embeddings @ cnn_gallery_embeddings.T
    triplet_scores = triplet_query_embeddings @ triplet_gallery_embeddings.T

    fused_scores = ALPHA * cnn_scores + BETA * triplet_scores

    top1, top3, top5, top10, top1_scores = compute_topk(
        fused_scores,
        gallery_ids,
        query_ids
    )

    print("=== CNN + Triplet Fusion ===")
    print(f"ALPHA CNN: {ALPHA}")
    print(f"BETA Triplet: {BETA}")
    print(f"Total gallery: {len(gallery_ids)}")
    print(f"Total queries: {len(query_ids)}")
    print(f"Top-1 Accuracy:  {top1:.4f}")
    print(f"Top-3 Accuracy:  {top3:.4f}")
    print(f"Top-5 Accuracy:  {top5:.4f}")
    print(f"Top-10 Accuracy: {top10:.4f}")
    print()
    print(f"Mean Top-1 Score: {top1_scores.mean():.4f}")
    print(f"Min Top-1 Score:  {top1_scores.min():.4f}")
    print(f"Max Top-1 Score:  {top1_scores.max():.4f}")

if __name__ == "__main__":
    main()