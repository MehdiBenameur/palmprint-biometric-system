from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AI_ROOT = PROJECT_ROOT / "ai-service"
EMB_DIR = AI_ROOT / "outputs" / "finetuned_embeddings_v3"

TOP_K = 5
ALPHA = 0.6  # CNN score
BETA = 0.4   # ORB score

def l2_normalize(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

def orb_match_score(img1_path, img2_path):
    img1 = cv2.imread(str(img1_path), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(img2_path), cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        return 0.0

    orb = cv2.ORB_create(nfeatures=500)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)

    if len(matches) == 0:
        return 0.0

    good_matches = [m for m in matches if m.distance < 60]

    score = len(good_matches) / max(len(kp1), len(kp2), 1)
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

    correct = 0

    for i in tqdm(range(len(query_ids)), desc="CNN + ORB reranking"):
        q_scores = cnn_scores[i]

        topk_indices = np.argsort(q_scores)[-TOP_K:]
        q_path = query_paths[i]

        best_final_score = -1
        best_id = None

        for idx in topk_indices:
            cnn_score = float(q_scores[idx])
            orb_score = orb_match_score(q_path, gallery_paths[idx])

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
    print(f"CNN + ORB Re-ranking Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()