from pathlib import Path
import cv2
import numpy as np


class PalmRoiExtractor:
    def __init__(self, output_size: int = 128):
        self.output_size = output_size

    def extract(self, image_bgr: np.ndarray) -> dict:
        if image_bgr is None:
            raise ValueError("Input image is None")

        # 1. Convert to grayscale
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # 2. Blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 3. Threshold (Otsu)
        _, binary = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Invert if needed
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)

        # 4. Morphological cleaning
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # 5. Find contours
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            raise ValueError("No hand contour found")

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # 6. Crop hand region
        hand_roi = gray[y:y + h, x:x + w]

        # =========================
        # ROI V3 (FINAL)
        # =========================
        roi_size = int(min(w, h) * 0.50)

        center_x = int(w * 0.48)
        center_y = int(h * 0.64)

        x1 = max(center_x - roi_size // 2, 0)
        y1 = max(center_y - roi_size // 2, 0)
        x2 = min(x1 + roi_size, w)
        y2 = min(y1 + roi_size, h)

        # Ensure correct size
        if x2 - x1 < roi_size:
            x1 = max(x2 - roi_size, 0)

        if y2 - y1 < roi_size:
            y1 = max(y2 - roi_size, 0)

        palm_roi = hand_roi[y1:y2, x1:x2]

        if palm_roi.size == 0:
            raise ValueError("Empty palm ROI")

        # 7. Resize
        palm_roi = cv2.resize(
            palm_roi,
            (self.output_size, self.output_size),
            interpolation=cv2.INTER_AREA
        )

        # 8. CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        palm_roi = clahe.apply(palm_roi)

        # Debug info
        debug = {
            "gray": gray,
            "mask": mask,
            "hand_bbox": (x, y, w, h),
            "palm_bbox_inside_hand": (x1, y1, x2 - x1, y2 - y1),
        }

        return {
            "roi": palm_roi,
            "debug": debug,
        }


def save_roi(image_path: str, output_path: str) -> None:
    extractor = PalmRoiExtractor()
    image = cv2.imread(image_path)
    result = extractor.extract(image)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, result["roi"])