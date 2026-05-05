from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AI_ROOT = PROJECT_ROOT / "ai-service"
EMB_DIR = AI_ROOT / "outputs" / "finetuned_embeddings_v3"

def l2_normalize(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

def topk_accuracy(scores, gallery_ids, query_ids, ks=(1, 3, 5, 10), higher_is_better=True):
    results = {}

    if higher_is_better:
        sorted_idx = np.argsort(scores, axis=1)
    else:
        sorted_idx = np.argsort(-scores, axis=1)

    for k in ks:
        topk_idx = sorted_idx[:, -k:]
        acc = np.mean([
            query_ids[i] in gallery_ids[topk_idx[i]]
            for i in range(len(query_ids))
        ])
        results[k] = acc

    return results

def print_results(title, results):
    print(f"\n=== {title} ===")
    print(f"Top-1 : {results[1]:.4f}")
    print(f"Top-3 : {results[3]:.4f}")
    print(f"Top-5 : {results[5]:.4f}")
    print(f"Top-10: {results[10]:.4f}")

def build_prototypes(gallery_embeddings, gallery_ids):
    palm_ids = sorted(set(gallery_ids))
    prototypes = []
    prototype_ids = []

    for pid in palm_ids:
        embs = gallery_embeddings[gallery_ids == pid]
        proto = embs.mean(axis=0)
        prototypes.append(proto)
        prototype_ids.append(pid)

    return np.array(prototypes), np.array(prototype_ids)

def main():
    gallery = np.load(EMB_DIR / "gallery.npy", allow_pickle=True)
    query = np.load(EMB_DIR / "query.npy", allow_pickle=True)

    gallery_embeddings = np.array([g["embedding"] for g in gallery])
    gallery_ids = np.array([int(g["palm_id"]) for g in gallery])

    query_embeddings = np.array([q["embedding"] for q in query])
    query_ids = np.array([int(q["palm_id"]) for q in query])

    gallery_embeddings = l2_normalize(gallery_embeddings)
    query_embeddings = l2_normalize(query_embeddings)

    print(f"Gallery: {len(gallery_ids)}")
    print(f"Query: {len(query_ids)}")
    print(f"Embedding dim: {gallery_embeddings.shape[1]}")

    cosine_scores = query_embeddings @ gallery_embeddings.T
    cosine_results = topk_accuracy(cosine_scores, gallery_ids, query_ids)
    print_results("Image-to-image Cosine", cosine_results)

    manhattan_dist = np.sum(np.abs(query_embeddings[:, None, :] - gallery_embeddings[None, :, :]), axis=2)
    manhattan_scores = -manhattan_dist
    manhattan_results = topk_accuracy(manhattan_scores, gallery_ids, query_ids)
    print_results("Image-to-image Manhattan", manhattan_results)

    prototypes, prototype_ids = build_prototypes(gallery_embeddings, gallery_ids)
    prototypes = l2_normalize(prototypes)

    proto_cosine_scores = query_embeddings @ prototypes.T
    proto_cosine_results = topk_accuracy(proto_cosine_scores, prototype_ids, query_ids)
    print_results("Prototype Cosine", proto_cosine_results)

    proto_manhattan_dist = np.sum(np.abs(query_embeddings[:, None, :] - prototypes[None, :, :]), axis=2)
    proto_manhattan_scores = -proto_manhattan_dist
    proto_manhattan_results = topk_accuracy(proto_manhattan_scores, prototype_ids, query_ids)
    print_results("Prototype Manhattan", proto_manhattan_results)

if __name__ == "__main__":
    main()