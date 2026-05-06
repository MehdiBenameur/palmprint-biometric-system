from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AI_ROOT = PROJECT_ROOT / "ai-service"
sys.path.append(str(AI_ROOT / "src"))

from final_identifier import (
    FinalPalmIdentifier,
    l2_normalize,
    ALPHA,
    BETA,
)

QUERY_ROOT = AI_ROOT / "outputs" / "roi_dataset_v3" / "session2"

KNOWN_MAX_ID = 500
THRESHOLDS = [0.65, 0.70, 0.75, 0.80, 0.85]


def get_query_images():
    known_samples = []
    unknown_samples = []

    palm_dirs = sorted([p for p in QUERY_ROOT.iterdir() if p.is_dir()])

    for palm_dir in palm_dirs:
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


def identify_with_threshold(identifier, image_path, threshold):
    cnn_embedding = identifier.cnn_extractor.extract(image_path)
    triplet_embedding = identifier.triplet_extractor.extract(image_path)

    cnn_embedding = cnn_embedding / (np.linalg.norm(cnn_embedding) + 1e-12)
    triplet_embedding = triplet_embedding / (np.linalg.norm(triplet_embedding) + 1e-12)

    cnn_scores = identifier.cnn_gallery_embeddings @ cnn_embedding
    triplet_scores = identifier.triplet_gallery_embeddings @ triplet_embedding

    final_scores = ALPHA * cnn_scores + BETA * triplet_scores

    best_idx = int(np.argmax(final_scores))
    best_score = float(final_scores[best_idx])
    best_id = int(identifier.gallery_ids[best_idx])

    accepted = best_score >= threshold

    return accepted, best_id, best_score


def evaluate_threshold(identifier, known_samples, unknown_samples, threshold):
    known_correct = 0
    known_accepted = 0

    unknown_rejected = 0
    unknown_accepted = 0

    known_scores = []
    unknown_scores = []

    for image_path, true_id in tqdm(known_samples, desc=f"Known threshold {threshold}"):
        accepted, predicted_id, score = identify_with_threshold(identifier, image_path, threshold)

        known_scores.append(score)

        if accepted:
            known_accepted += 1

        if accepted and predicted_id == true_id:
            known_correct += 1

    for image_path, _ in tqdm(unknown_samples, desc=f"Unknown threshold {threshold}"):
        accepted, predicted_id, score = identify_with_threshold(identifier, image_path, threshold)

        unknown_scores.append(score)

        if accepted:
            unknown_accepted += 1
        else:
            unknown_rejected += 1

    known_total = len(known_samples)
    unknown_total = len(unknown_samples)

    known_accuracy = known_correct / known_total
    known_acceptance_rate = known_accepted / known_total

    unknown_rejection_rate = unknown_rejected / unknown_total
    false_accept_rate = unknown_accepted / unknown_total

    return {
        "threshold": threshold,
        "known_total": known_total,
        "unknown_total": unknown_total,
        "known_accuracy": known_accuracy,
        "known_acceptance_rate": known_acceptance_rate,
        "unknown_rejection_rate": unknown_rejection_rate,
        "false_accept_rate": false_accept_rate,
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
    print()

    results = []

    for threshold in THRESHOLDS:
        result = evaluate_threshold(identifier, known_samples, unknown_samples, threshold)
        results.append(result)

        print("\n=== Threshold Evaluation ===")
        print(f"Threshold: {result['threshold']}")
        print(f"Known accuracy:          {result['known_accuracy']:.4f}")
        print(f"Known acceptance rate:   {result['known_acceptance_rate']:.4f}")
        print(f"Unknown rejection rate:  {result['unknown_rejection_rate']:.4f}")
        print(f"False accept rate:       {result['false_accept_rate']:.4f}")
        print(f"Known mean score:        {result['known_mean_score']:.4f}")
        print(f"Known min score:         {result['known_min_score']:.4f}")
        print(f"Unknown mean score:      {result['unknown_mean_score']:.4f}")
        print(f"Unknown max score:       {result['unknown_max_score']:.4f}")

    print("\n=== Summary ===")
    for r in results:
        print(
            f"TH={r['threshold']:.2f} | "
            f"Known Acc={r['known_accuracy']:.4f} | "
            f"Unknown Reject={r['unknown_rejection_rate']:.4f} | "
            f"FAR={r['false_accept_rate']:.4f}"
        )


if __name__ == "__main__":
    main()