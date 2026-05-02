from pathlib import Path
import cv2
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data" / "Tongji"
INDEX_CSV = DATASET_ROOT / "tongji_index.csv"
OUTPUT_DIR = PROJECT_ROOT / "ai-service" / "outputs" / "dataset_inspection"

def compute_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = gray.mean()
    contrast = gray.std()

    return {
        "blur": round(float(blur), 2),
        "brightness": round(float(brightness), 2),
        "contrast": round(float(contrast), 2),
    }

def save_sample_grid(df):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    samples = pd.concat([
        df[df["session"] == "session1"].head(8),
        df[df["session"] == "session2"].head(8),
    ])

    plt.figure(figsize=(12, 6))

    for i, row in enumerate(samples.itertuples(), start=1):
        image = cv2.imread(row.image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.subplot(2, 8, i)
        plt.imshow(image_rgb)
        plt.title(f"{row.session}\nID {row.palm_id}")
        plt.axis("off")

    output_path = OUTPUT_DIR / "tongji_samples_grid.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Sample grid saved: {output_path}")

def analyze_quality(df):
    rows = []

    for row in df.sample(100, random_state=42).itertuples():
        image = cv2.imread(row.image_path)

        if image is None:
            continue

        q = compute_quality(image)
        rows.append({
            "session": row.session,
            "file_name": row.file_name,
            "palm_id": row.palm_id,
            **q
        })

    quality_df = pd.DataFrame(rows)
    output_csv = OUTPUT_DIR / "quality_sample_100.csv"
    quality_df.to_csv(output_csv, index=False)

    print(f"Quality CSV saved: {output_csv}")
    print()
    print("Quality summary:")
    print(quality_df[["blur", "brightness", "contrast"]].describe())

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INDEX_CSV)

    print("Dataset loaded.")
    print(f"Total images: {len(df)}")
    print(f"Sessions: {df['session'].unique()}")
    print(f"Unique palms: {df['palm_id'].nunique()}")
    print(f"Unique subjects: {df['subject_id'].nunique()}")
    print()

    print("Checking first image...")
    first_path = df.iloc[0]["image_path"]
    image = cv2.imread(first_path)

    if image is None:
        raise ValueError(f"Cannot read image: {first_path}")

    print(f"First image path: {first_path}")
    print(f"First image shape: {image.shape}")
    print(f"First image quality: {compute_quality(image)}")
    print()

    save_sample_grid(df)
    analyze_quality(df)

if __name__ == "__main__":
    main()