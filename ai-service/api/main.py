from fastapi import FastAPI, UploadFile, File, HTTPException
import tempfile
import shutil
import os

from src.final_identifier import FinalPalmIdentifier
from src.image_quality import compute_image_quality

app = FastAPI(title="Palmprint AI Service")

identifier = FinalPalmIdentifier()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Palmprint AI Service"
    }


@app.post("/generate-embedding")
async def generate_embedding(image: UploadFile = File(...)):
    suffix = os.path.splitext(image.filename or "")[1].lower()

    if suffix not in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]:
        raise HTTPException(status_code=400, detail="Invalid image format.")

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            shutil.copyfileobj(image.file, temp_file)
            temp_path = temp_file.name

        quality = compute_image_quality(temp_path)

        if not quality["acceptable"]:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Image quality is too low.",
                    "quality": quality
                }
            )

        cnn_embedding = identifier.cnn_extractor.extract(temp_path)
        triplet_embedding = identifier.triplet_extractor.extract(temp_path)

        return {
            "cnn_embedding": cnn_embedding.tolist(),
            "triplet_embedding": triplet_embedding.tolist(),
            "quality_score": quality["quality_score"],
            "quality_details": quality,
            "model_version": "cnn-triplet-fusion-v1"
        }

    except HTTPException:
        raise

    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)