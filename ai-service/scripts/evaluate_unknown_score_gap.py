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

SCORE_THRESHOLDS = [0.70, 0.75, 0.80, 0.85]
GAP_THRESHOLDS = [0.01, 0.02, 0.03, 0.05, 0.08]


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


def get_scores(identifier, image_path):
    cnn_embedding = identifier.cnn_extractor.extract(image_path)
    triplet_embedding = identifier.triplet_extractor.extract(image_path)

    cnn_embedding = cnn_embedding / (np.linalg.norm(cnn_embedding) + 1e-12)
    triplet_embedding = triplet_embedding / (np.linalg.norm(triplet_embedding) + 1e-12)

    cnn_scores = identifier.cnn_gallery_embeddings @ cnn_embedding
    triplet_scores = identifier.triplet_gallery_embeddings @ triplet_embedding

    final_scores = ALPHA * cnn_scores + BETA * triplet_scores

    sorted_indices = np.argsort(-final_scores)

    best_idx = int(sorted_indices[0])
    second_idx = int(sorted_indices[1])

    best_score = float(final_scores[best_idx])
    second_score = float(final_scores[second_idx])
    gap = best_score - second_score

    best_id = int(identifier.gallery_ids[best_idx])
    second_id = int(identifier.gallery_ids[second_idx])

    return best_id, second_id, best_score, second_score, gap


def precompute_results(identifier, samples, desc):
    rows = []

    for image_path, true_id in tqdm(samples, desc=desc):
        best_id, second_id, best_score, second_score, gap = get_scores(identifier, image_path)

        rows.append({
            "true_id": true_id,
            "best_id": best_id,
            "second_id": second_id,
            "best_score": best_score,
            "second_score": second_score,
            "gap": gap,
            "correct": best_id == true_id,
        })

    return rows


def evaluate_combo(known_rows, unknown_rows, score_threshold, gap_threshold):
    known_correct_accepted = 0
    known_accepted = 0

    unknown_rejected = 0
    unknown_accepted = 0

    for row in known_rows:
        accepted = (
            row["best_score"] >= score_threshold
            and row["gap"] >= gap_threshold
        )

        if accepted:
            known_accepted += 1

        if accepted and row["correct"]:
            known_correct_accepted += 1

    for row in unknown_rows:
        accepted = (
            row["best_score"] >= score_threshold
            and row["gap"] >= gap_threshold
        )

        if accepted:
            unknown_accepted += 1
        else:
            unknown_rejected += 1

    known_total = len(known_rows)
    unknown_total = len(unknown_rows)

    return {
        "score_threshold": score_threshold,
        "gap_threshold": gap_threshold,
        "known_accuracy": known_correct_accepted / known_total,
        "known_acceptance_rate": known_accepted / known_total,
        "unknown_rejection_rate": unknown_rejected / unknown_total,
        "false_accept_rate": unknown_accepted / unknown_total,
    }


def main():
    identifier = FinalPalmIdentifier()
    restrict_gallery(identifier)

    known_samples, unknown_samples = get_query_images()

    print(f"Known queries: {len(known_samples)}")
    print(f"Unknown queries: {len(unknown_samples)}")

    known_rows = precompute_results(identifier, known_samples, "Precompute known scores")
    unknown_rows = precompute_results(identifier, unknown_samples, "Precompute unknown scores")

    print("\n=== Score / Gap Statistics ===")
    print(f"Known mean best score:   {np.mean([r['best_score'] for r in known_rows]):.4f}")
    print(f"Unknown mean best score: {np.mean([r['best_score'] for r in unknown_rows]):.4f}")
    print(f"Known mean gap:          {np.mean([r['gap'] for r in known_rows]):.4f}")
    print(f"Unknown mean gap:        {np.mean([r['gap'] for r in unknown_rows]):.4f}")

    all_results = []

    for score_threshold in SCORE_THRESHOLDS:
        for gap_threshold in GAP_THRESHOLDS:
            result = evaluate_combo(
                known_rows,
                unknown_rows,
                score_threshold,
                gap_threshold
            )
            all_results.append(result)

    print("\n=== Summary Score + Gap ===")
    for r in all_results:
        print(
            f"S={r['score_threshold']:.2f} | "
            f"G={r['gap_threshold']:.2f} | "
            f"KnownAcc={r['known_accuracy']:.4f} | "
            f"KnownAccept={r['known_acceptance_rate']:.4f} | "
            f"UnknownReject={r['unknown_rejection_rate']:.4f} | "
            f"FAR={r['false_accept_rate']:.4f}"
        )


if __name__ == "__main__":
    main()