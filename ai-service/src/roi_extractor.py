from pathlib import Path
from turtle import distance
import cv2
import numpy as np


class PalmRoiExtractor:
    def __init__(self, output_size: int = 128):
        self.output_size = output_size

    def _segment_hand(self, image_bgr: np.ndarray):
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

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            raise ValueError("No hand contour found")

        contour = max(contours, key=cv2.contourArea)
        return gray, mask, contour

    def _detect_valley_candidates(self, contour, image_shape):
        hull_indices = cv2.convexHull(contour, returnPoints=False)

        if hull_indices is None or len(hull_indices) < 4:
            return []

        defects = cv2.convexityDefects(contour, hull_indices)

        if defects is None:
            return []

        h, w = image_shape[:2]
        candidates = []

        for i in range(defects.shape[0]):
            start_idx, end_idx, far_idx, depth = defects[i, 0]
            far = tuple(contour[far_idx][0])
            fx, fy = far
            depth_float = depth / 256.0

            if depth_float < 20:
                continue

            if fy > h * 0.58:
                continue

            if fx < w * 0.08 or fx > w * 0.92:
                continue

            candidates.append({
                "point": far,
                "depth": depth_float,
            })

        return candidates

    def _select_two_valleys(self, candidates, image_shape):
        if len(candidates) < 2:
            return []

        # Trier par X (gauche → droite)
        candidates = sorted(candidates, key=lambda c: c["point"][0])

        # Si on a au moins 4 → prendre les 2 du milieu (robuste)
        if len(candidates) >= 4:
            selected = candidates[1:3]

        # Si 3 → prendre les 2 centraux
        elif len(candidates) == 3:
            selected = candidates[0:2]

        else:
            selected = candidates[:2]

        return selected


    def _fallback_roi(self, gray, contour):
        x, y, w, h = cv2.boundingRect(contour)
        hand_roi = gray[y:y + h, x:x + w]

        roi_size = int(min(w, h) * 0.50)
        center_x = int(w * 0.48)
        center_y = int(h * 0.64)

        x1 = max(center_x - roi_size // 2, 0)
        y1 = max(center_y - roi_size // 2, 0)
        x2 = min(x1 + roi_size, w)
        y2 = min(y1 + roi_size, h)

        if x2 - x1 < roi_size:
            x1 = max(x2 - roi_size, 0)
        if y2 - y1 < roi_size:
            y1 = max(y2 - roi_size, 0)

        palm_roi = hand_roi[y1:y2, x1:x2]

        debug = {
            "method": "fallback_v3",
            "hand_bbox": (x, y, w, h),
            "palm_bbox_inside_hand": (x1, y1, x2 - x1, y2 - y1),
        }

        return palm_roi, debug

    def _crop_with_valleys(self, gray, selected):
        p1 = np.array(selected[0]["point"], dtype=np.float32)
        p2 = np.array(selected[1]["point"], dtype=np.float32)

        midpoint = (p1 + p2) / 2.0
        distance = np.linalg.norm(p2 - p1)

        if distance < 30:
            raise ValueError("Selected valleys too close")

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        angle = np.degrees(np.arctan2(dy, dx))

        h, w = gray.shape[:2]
        rot_matrix = cv2.getRotationMatrix2D(tuple(midpoint), angle, 1.0)
        rotated = cv2.warpAffine(
            gray,
            rot_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        midpoint_h = np.array([midpoint[0], midpoint[1], 1.0], dtype=np.float32)
        rotated_midpoint = rot_matrix @ midpoint_h

        roi_size = int(distance * 2.8)
        offset_y = int(distance * 1.8)

        cx = int(rotated_midpoint[0])
        cy = int(rotated_midpoint[1] + offset_y)

        x1 = cx - roi_size // 2
        y1 = cy - roi_size // 2
        x2 = x1 + roi_size
        y2 = y1 + roi_size

        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            raise ValueError("Valley ROI outside image")

        palm_roi = rotated[y1:y2, x1:x2]

        debug = {
            "method": "valley_v6",
            "selected_valleys": (tuple(p1.astype(int)), tuple(p2.astype(int))),
            "midpoint": tuple(midpoint.astype(int)),
            "distance": float(distance),
            "angle": float(angle),
            "roi_bbox": (x1, y1, x2 - x1, y2 - y1),
        }

        return palm_roi, debug

    def extract(self, image_bgr: np.ndarray) -> dict:
        if image_bgr is None:
            raise ValueError("Input image is None")

        gray, mask, contour = self._segment_hand(image_bgr)
        x, y, w, h = cv2.boundingRect(contour)

        candidates = self._detect_valley_candidates(contour, image_bgr.shape)
        selected = self._select_two_valleys(candidates, image_bgr.shape)

        try:
            if len(selected) == 2:
                palm_roi, debug = self._crop_with_valleys(gray, selected)
            else:
                palm_roi, debug = self._fallback_roi(gray, contour)
        except Exception:
            palm_roi, debug = self._fallback_roi(gray, contour)

        if palm_roi.size == 0:
            raise ValueError("Empty palm ROI")

        palm_roi = cv2.resize(
            palm_roi,
            (self.output_size, self.output_size),
            interpolation=cv2.INTER_AREA
        )

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        palm_roi = clahe.apply(palm_roi)

        debug["hand_bbox"] = (x, y, w, h)

        if "palm_bbox_inside_hand" not in debug:
            debug["palm_bbox_inside_hand"] = debug.get("roi_bbox", (0, 0, 0, 0))
        debug["gray"] = gray
        debug["mask"] = mask

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