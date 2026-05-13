import cv2
import numpy as np


def compute_image_quality(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        return {
            "quality_score": 0.0,
            "blur": 0.0,
            "brightness": 0.0,
            "contrast": 0.0,
            "acceptable": False
        }

    blur = cv2.Laplacian(image, cv2.CV_64F).var()
    brightness = float(np.mean(image))
    contrast = float(np.std(image))

    blur_score = min(blur / 50.0, 1.0)

    brightness_score = 1.0 - abs(brightness - 100.0) / 100.0
    brightness_score = max(0.0, min(brightness_score, 1.0))

    contrast_score = min(contrast / 50.0, 1.0)

    quality_score = (
        0.4 * blur_score +
        0.3 * brightness_score +
        0.3 * contrast_score
    )

    acceptable = quality_score >= 0.30

    return {
        "quality_score": float(quality_score),
        "blur": float(blur),
        "brightness": float(brightness),
        "contrast": float(contrast),
        "acceptable": bool(acceptable)
    }