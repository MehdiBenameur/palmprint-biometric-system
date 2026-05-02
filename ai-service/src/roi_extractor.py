from pathlib import Path
import cv2
import numpy as np

class PalmRoiExtractor:
    def __init__(self, output_size: int = 128):
        self.output_size = output_size

    def extract(self, image_bgr: np.ndarray) -> dict:
        if image_bgr is None:
            raise ValueError("Input image is None")

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        _, binary = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)

        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            raise ValueError("No hand contour found")

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        hand_roi = gray[y:y+h, x:x+w]

        palm_side = int(min(w, h) * 0.72)

        center_x = w // 2
        center_y = int(h * 0.65)

        x1 = max(center_x - palm_side // 2, 0)
        y1 = max(center_y - palm_side // 2, 0)
        x2 = min(x1 + palm_side, w)
        y2 = min(y1 + palm_side, h)

        palm_roi = hand_roi[y1:y2, x1:x2]

        if palm_roi.size == 0:
            raise ValueError("Empty palm ROI")

        palm_roi = cv2.resize(palm_roi, (self.output_size, self.output_size))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        palm_roi = clahe.apply(palm_roi)

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