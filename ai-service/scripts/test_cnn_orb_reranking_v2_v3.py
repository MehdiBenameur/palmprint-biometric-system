from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AI_ROOT = PROJECT_ROOT / "ai-service"
EMB_DIR = AI_ROOT / "outputs" / "finetuned_embeddings_v3"

TOP_K = 10
ALPHA = 0.5
BETA = 0.5
ORB_FEATURES = 700

def l2_normalize(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

def extract_orb_features(image_path, orb):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        return [], None

    keypoints, descriptors = orb.detectAndCompute(image, None)

    if descriptors is None or len(keypoints) == 0:
        return [], None

    return keypoints, descriptors

def orb_score_from_descriptors(kp1, des1, kp2, des2):
    if des1 is None or des2 is None:
        return 0.0

    if len(kp1) == 0 or len(kp2) == 0:
        return 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = matcher.knnMatch(des1, des2, k=2)

    good_matches = []

    for pair in knn_matches:
        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) == 0:
        return 0.0

    match_ratio = len(good_matches) / max(len(kp1), len(kp2), 1)

    mean_distance = np.mean([m.distance for m in good_matches])
    distance_score = 1.0 - min(mean_distance / 100.0, 1.0)

    score = 0.7 * match_ratio + 0.3 * distance_score

    return float(score)

def main():
    gallery = np.load(EMB_DIR / "gallery.npy", allow_pickle=True)
    query = np.load(EMB_DIR / "query.npy", allow_pickle=True)

    gallery_embeddings = l2_normalize(np.array([g["embedding"] for g in gallery]))
    gallery_ids = np.array([int(g["palm_id"]) for g in gallery])
    gallery_paths = np.array([g["image_path"] for g in gallery])

    query_embeddings = l2_normalize(np.array([q["embedding"] for q in query]))
    query_ids = np.array([int(q["palm_id"]) for q in query])
    query_paths = np.array([q["image_path"] for q in query])

    cnn_scores = query_embeddings @ gallery_embeddings.T

    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)

    print("Caching ORB features for gallery...")
    gallery_orb_cache = []
    for path in tqdm(gallery_paths, desc="Gallery ORB cache"):
        gallery_orb_cache.append(extract_orb_features(path, orb))

    print("Caching ORB features for queries...")
    query_orb_cache = []
    for path in tqdm(query_paths, desc="Query ORB cache"):
        query_orb_cache.append(extract_orb_features(path, orb))

    correct = 0

    for i in tqdm(range(len(query_ids)), desc="CNN + ORB reranking final"):
        q_scores = cnn_scores[i]

        topk_indices = np.argsort(-q_scores)[:TOP_K]

        q_kp, q_des = query_orb_cache[i]

        best_final_score = -1.0
        best_id = None

        for idx in topk_indices:
            cnn_score = float(q_scores[idx])

            g_kp, g_des = gallery_orb_cache[idx]
            orb_score = orb_score_from_descriptors(q_kp, q_des, g_kp, g_des)

            final_score = ALPHA * cnn_score + BETA * orb_score

            if final_score > best_final_score:
                best_final_score = final_score
                best_id = gallery_ids[idx]

        if best_id == query_ids[i]:
            correct += 1

    acc = correct / len(query_ids)

    print(f"Total queries: {len(query_ids)}")
    print(f"TOP_K: {TOP_K}")
    print(f"ALPHA: {ALPHA}")
    print(f"BETA: {BETA}")
    print(f"ORB_FEATURES: {ORB_FEATURES}")
    print(f"CNN + ORB Re-ranking Final Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()