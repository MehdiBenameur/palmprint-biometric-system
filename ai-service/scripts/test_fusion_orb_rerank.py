from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AI_ROOT = PROJECT_ROOT / "ai-service"

CNN_EMB_DIR = AI_ROOT / "outputs" / "finetuned_embeddings_v3"
TRIPLET_EMB_DIR = AI_ROOT / "outputs" / "triplet_embeddings_v3"

ALPHA = 0.42
BETA = 0.58
TOP_K = 10
ORB_WEIGHT = 0.1

def l2_normalize(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

def orb_score(img1_path, img2_path, orb):
    img1 = cv2.imread(str(img1_path), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(img2_path), cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        return 0.0

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        return 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.knnMatch(des1, des2, k=2)

    good = []

    for pair in matches:
        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) == 0:
        return 0.0

    match_ratio = len(good) / max(len(kp1), len(kp2), 1)
    mean_distance = np.mean([m.distance for m in good])
    distance_score = 1.0 - min(mean_distance / 100.0, 1.0)

    return float(0.7 * match_ratio + 0.3 * distance_score)

def main():
    cnn_gallery = np.load(CNN_EMB_DIR / "gallery.npy", allow_pickle=True)
    cnn_query = np.load(CNN_EMB_DIR / "query.npy", allow_pickle=True)

    triplet_gallery = np.load(TRIPLET_EMB_DIR / "gallery.npy", allow_pickle=True)
    triplet_query = np.load(TRIPLET_EMB_DIR / "query.npy", allow_pickle=True)

    gallery_ids = np.array([int(g["palm_id"]) for g in cnn_gallery])
    query_ids = np.array([int(q["palm_id"]) for q in cnn_query])

    gallery_paths = np.array([g["image_path"] for g in cnn_gallery])
    query_paths = np.array([q["image_path"] for q in cnn_query])

    cnn_g = l2_normalize(np.array([g["embedding"] for g in cnn_gallery]))
    cnn_q = l2_normalize(np.array([q["embedding"] for q in cnn_query]))

    tri_g = l2_normalize(np.array([g["embedding"] for g in triplet_gallery]))
    tri_q = l2_normalize(np.array([q["embedding"] for q in triplet_query]))

    cnn_scores = cnn_q @ cnn_g.T
    tri_scores = tri_q @ tri_g.T

    fused_scores = ALPHA * cnn_scores + BETA * tri_scores

    orb = cv2.ORB_create(nfeatures=700)

    correct = 0

    for i in tqdm(range(len(query_ids)), desc="Fusion + ORB reranking"):
        scores = fused_scores[i]
        topk_idx = np.argsort(-scores)[:TOP_K]

        best_score = -1.0
        best_id = None

        for idx in topk_idx:
            base_score = float(scores[idx])
            orb_s = orb_score(query_paths[i], gallery_paths[idx], orb)
            final_score = base_score + ORB_WEIGHT * orb_s

            if final_score > best_score:
                best_score = final_score
                best_id = gallery_ids[idx]

        if best_id == query_ids[i]:
            correct += 1

    acc = correct / len(query_ids)

    print("=== FINAL SYSTEM ===")
    print(f"ALPHA CNN: {ALPHA}")
    print(f"BETA Triplet: {BETA}")
    print(f"TOP_K: {TOP_K}")
    print(f"ORB_WEIGHT: {ORB_WEIGHT}")
    print(f"Top-1 Accuracy (Fusion + ORB): {acc:.4f}")

if __name__ == "__main__":
    main()