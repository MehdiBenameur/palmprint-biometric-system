from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AI_ROOT = PROJECT_ROOT / "ai-service"
sys.path.append(str(AI_ROOT / "src"))

from final_identifier import FinalPalmIdentifier

TEST_IMAGE = AI_ROOT / "outputs" / "roi_dataset_v3" / "session2" / "palm_001" / "01.bmp"


def main():
    identifier = FinalPalmIdentifier()

    result = identifier.identify(TEST_IMAGE)

    print("=== Final Identifier Test ===")
    print(f"Test image: {TEST_IMAGE}")
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()