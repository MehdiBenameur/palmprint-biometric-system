from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AI_ROOT = PROJECT_ROOT / "ai-service"
sys.path.append(str(AI_ROOT / "src"))

from final_identifier import FinalPalmIdentifier, ALPHA, BETA

QUERY_ROOT = AI_ROOT / "outputs" / "roi_dataset_v3" / "session2"

KNOWN_MAX_ID = 500
THRESHOLDS = [0.70, 0.75, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90]
POOLING = "top3_mean"


def get_query_images():
    known_samples = []
    unknown_samples = []

    for palm_dir in sorted([p for p in QUERY_ROOT.iterdir() if p.is_dir()]):
        palm_id = int(palm_dir.name.split("_")[1])

        for img_path in sorted(palm_dir.glob("*.bmp")):
            if palm_id <= KNOWN_MAX_ID:
                known_samples.append((img_path, palm_id))
            else:
                unknown_samples.append((img_path, palm_id))

    return known_samples, unknown_samples


def restrict_gallery(identifier):
    mask = identifier.gallery_ids <= KNOWN_MAX_ID

    identifier.gallery_ids = identifier.gallery_ids[mask]
    identifier.cnn_gallery_embeddings = identifier.cnn_gallery_embeddings[mask]
    identifier.triplet_gallery_embeddings = identifier.triplet_gallery_embeddings[mask]

    print(f"Restricted gallery templates: {len(identifier.gallery_ids)}")
    print(f"Restricted gallery identities: {len(set(identifier.gallery_ids))}")


def identity_score(values):
    if POOLING == "max":
        return float(np.max(values))

    if POOLING == "mean":
        return float(np.mean(values))

    if POOLING == "top3_mean":
        top_values = np.sort(values)[-3:]
        return float(np.mean(top_values))

    raise ValueError(f"Unknown pooling: {POOLING}")


def identify_identity_pooling(identifier, image_path, threshold):
    cnn_embedding = identifier.cnn_extractor.extract(image_path)
    triplet_embedding = identifier.triplet_extractor.extract(image_path)

    cnn_embedding = cnn_embedding / (np.linalg.norm(cnn_embedding) + 1e-12)
    triplet_embedding = triplet_embedding / (np.linalg.norm(triplet_embedding) + 1e-12)

    cnn_scores = identifier.cnn_gallery_embeddings @ cnn_embedding
    triplet_scores = identifier.triplet_gallery_embeddings @ triplet_embedding
    fused_scores = ALPHA * cnn_scores + BETA * triplet_scores

    unique_ids = np.array(sorted(set(identifier.gallery_ids)))

    identity_scores = []

    for palm_id in unique_ids:
        indices = np.where(identifier.gallery_ids == palm_id)[0]
        score = identity_score(fused_scores[indices])
        identity_scores.append(score)

    identity_scores = np.array(identity_scores)

    best_idx = int(np.argmax(identity_scores))
    best_id = int(unique_ids[best_idx])
    best_score = float(identity_scores[best_idx])

    accepted = best_score >= threshold

    return accepted, best_id, best_score


def evaluate_threshold(identifier, known_samples, unknown_samples, threshold):
    known_correct = 0
    known_accepted = 0
    unknown_rejected = 0
    unknown_accepted = 0

    known_scores = []
    unknown_scores = []

    for image_path, true_id in tqdm(known_samples, desc=f"Known TH={threshold}"):
        accepted, pred_id, score = identify_identity_pooling(identifier, image_path, threshold)

        known_scores.append(score)

        if accepted:
            known_accepted += 1

        if accepted and pred_id == true_id:
            known_correct += 1

    for image_path, _ in tqdm(unknown_samples, desc=f"Unknown TH={threshold}"):
        accepted, _, score = identify_identity_pooling(identifier, image_path, threshold)

        unknown_scores.append(score)

        if accepted:
            unknown_accepted += 1
        else:
            unknown_rejected += 1

    return {
        "threshold": threshold,
        "known_accuracy": known_correct / len(known_samples),
        "known_acceptance": known_accepted / len(known_samples),
        "unknown_rejection": unknown_rejected / len(unknown_samples),
        "false_accept_rate": unknown_accepted / len(unknown_samples),
        "known_mean_score": float(np.mean(known_scores)),
        "known_min_score": float(np.min(known_scores)),
        "unknown_mean_score": float(np.mean(unknown_scores)),
        "unknown_max_score": float(np.max(unknown_scores)),
    }


def main():
    identifier = FinalPalmIdentifier()
    restrict_gallery(identifier)

    known_samples, unknown_samples = get_query_images()

    print(f"Known queries: {len(known_samples)}")
    print(f"Unknown queries: {len(unknown_samples)}")
    print(f"POOLING: {POOLING}")

    results = []

    for threshold in THRESHOLDS:
        result = evaluate_threshold(identifier, known_samples, unknown_samples, threshold)
        results.append(result)

        print("\n=== Threshold Evaluation Identity Pooling ===")
        print(f"Threshold: {result['threshold']}")
        print(f"Known accuracy:         {result['known_accuracy']:.4f}")
        print(f"Known acceptance:       {result['known_acceptance']:.4f}")
        print(f"Unknown rejection:      {result['unknown_rejection']:.4f}")
        print(f"False accept rate:      {result['false_accept_rate']:.4f}")
        print(f"Known mean score:       {result['known_mean_score']:.4f}")
        print(f"Known min score:        {result['known_min_score']:.4f}")
        print(f"Unknown mean score:     {result['unknown_mean_score']:.4f}")
        print(f"Unknown max score:      {result['unknown_max_score']:.4f}")

    print("\n=== Summary ===")
    for r in results:
        print(
            f"TH={r['threshold']:.2f} | "
            f"KnownAcc={r['known_accuracy']:.4f} | "
            f"KnownAccept={r['known_acceptance']:.4f} | "
            f"UnknownReject={r['unknown_rejection']:.4f} | "
            f"FAR={r['false_accept_rate']:.4f}"
        )


if __name__ == "__main__":
    main()