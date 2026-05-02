from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data" / "Tongji"
OUTPUT_CSV = DATASET_ROOT / "tongji_index.csv"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".bmp", ".png", ".tif", ".tiff"}

def parse_image_info(image_path: Path, session_name: str) -> dict:
    image_number = int(image_path.stem)

    palm_id = ((image_number - 1) // 10) + 1
    sample_index = ((image_number - 1) % 10) + 1
    subject_id = ((palm_id - 1) // 2) + 1
    palm_index_for_subject = ((palm_id - 1) % 2) + 1

    return {
        "image_path": str(image_path),
        "session": session_name,
        "file_name": image_path.name,
        "image_number": image_number,
        "palm_id": palm_id,
        "subject_id": subject_id,
        "palm_index_for_subject": palm_index_for_subject,
        "sample_index": sample_index,
    }

def collect_session(session_name: str) -> list[dict]:
    session_path = DATASET_ROOT / session_name

    if not session_path.exists():
        raise FileNotFoundError(f"Missing folder: {session_path}")

    rows = []
    for image_path in sorted(session_path.iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in VALID_EXTENSIONS:
            rows.append(parse_image_info(image_path, session_name))

    return rows

def main() -> None:
    rows = []
    rows.extend(collect_session("session1"))
    rows.extend(collect_session("session2"))

    df = pd.DataFrame(rows)
    df = df.sort_values(["session", "image_number"]).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print("Tongji index created.")
    print(f"CSV: {OUTPUT_CSV}")
    print(f"Total images: {len(df)}")
    print(f"Unique palms: {df['palm_id'].nunique()}")
    print(f"Unique subjects: {df['subject_id'].nunique()}")
    print()
    print(df["session"].value_counts())
    print()
    print(df.head(10))

if __name__ == "__main__":
    main()