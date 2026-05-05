from pathlib import Path
import sys
from tqdm import tqdm
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AI_ROOT = PROJECT_ROOT / "ai-service"
sys.path.append(str(AI_ROOT / "src"))

from final_identifier import FinalPalmIdentifier

QUERY_ROOT = AI_ROOT / "outputs" / "roi_dataset_v3" / "session2"


def get_all_query_images():
    samples = []

    palm_dirs = sorted([p for p in QUERY_ROOT.iterdir() if p.is_dir()])

    for palm_dir in palm_dirs:
        palm_id = int(palm_dir.name.split("_")[1])

        for img_path in sorted(palm_dir.glob("*.bmp")):
            samples.append((img_path, palm_id))

    return samples


def main():
    identifier = FinalPalmIdentifier()

    samples = get_all_query_images()

    total = 0
    correct = 0

    accepted_count = 0
    rejected_count = 0

    scores = []

    for image_path, true_palm_id in tqdm(samples, desc="Evaluating"):
        result = identifier.identify(image_path)

        total += 1

        if result["accepted"]:
            accepted_count += 1
        else:
            rejected_count += 1

        predicted_id = result["predicted_palm_id"]

        if predicted_id == true_palm_id:
            correct += 1

        scores.append(result["final_score"])

    accuracy = correct / total

    print("\n=== Final Identifier Evaluation ===")
    print(f"Total queries: {total}")
    print(f"Correct identifications: {correct}")
    print(f"Accuracy: {accuracy:.4f}")
    print()
    print(f"Accepted: {accepted_count}")
    print(f"Rejected: {rejected_count}")
    print()
    print(f"Mean score: {np.mean(scores):.4f}")
    print(f"Min score:  {np.min(scores):.4f}")
    print(f"Max score:  {np.max(scores):.4f}")


if __name__ == "__main__":
    main()