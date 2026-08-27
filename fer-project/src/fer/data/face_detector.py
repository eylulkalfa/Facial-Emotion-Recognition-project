"""State-of-the-art Face Detection module using OpenCV YuNet DNN and Haar Cascade fallbacks."""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Model file paths
CURRENT_DIR = Path(__file__).parent.resolve()
YUNET_MODEL_PATH = CURRENT_DIR / "face_detection_yunet.onnx"
HAAR_MODEL_PATH = CURRENT_DIR / "haarcascade_frontalface_default.xml"


@dataclass
class FaceDetection:
    """Dataclass holding face detection result.

    Attributes:
        bbox: Tuple of (x_min, y_min, x_max, y_max) in absolute pixel coordinates.
        confidence: Detection confidence score in range [0, 1].
        face_image: Cropped face image as RGB numpy array (uint8).
    """

    bbox: Tuple[int, int, int, int]
    confidence: float
    face_image: np.ndarray


class FaceDetector:
    """OpenCV YuNet DNN + Haar Cascade face detector with tight cropping for FER."""

    def __init__(
        self,
        bypass: bool = False,
        min_confidence: float = 0.3,
        margin_ratio: float = 0.15,
    ):
        """Initialize FaceDetector.

        Args:
            bypass: If True, skip detection and return original image unchanged.
            min_confidence: Minimum detection confidence threshold.
            margin_ratio: Padding expansion ratio added around detected face bounding box (default 0.15 = 15%).
        """
        self.bypass = bypass
        self.min_confidence = min_confidence
        self.margin_ratio = margin_ratio
        self.logger = logging.getLogger(__name__)

        self._yunet_detector = None
        self._haar_cascade = None

        if not self.bypass:
            if YUNET_MODEL_PATH.exists():
                try:
                    self._yunet_detector = cv2.FaceDetectorYN.create(
                        str(YUNET_MODEL_PATH),
                        "",
                        (300, 300),
                        score_threshold=min_confidence,
                        nms_threshold=0.3,
                        top_k=5000,
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to initialize YuNet face detector: {e}")

            if HAAR_MODEL_PATH.exists():
                try:
                    cascade = cv2.CascadeClassifier(str(HAAR_MODEL_PATH))
                    if not cascade.empty():
                        self._haar_cascade = cascade
                except Exception as e:
                    self.logger.warning(f"Failed to initialize Haar Cascade classifier: {e}")

    def _crop_with_margin(
        self, image: np.ndarray, x_min_raw: float, y_min_raw: float, box_w: float, box_h: float
    ) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
        h, w = image.shape[:2]

        # Add 15% margin around face to include forehead wrinkles and chin gestures
        margin_w = box_w * self.margin_ratio
        margin_h = box_h * self.margin_ratio

        x_min = int(max(0, x_min_raw - margin_w))
        y_min = int(max(0, y_min_raw - margin_h))
        x_max = int(min(w, x_min_raw + box_w + margin_w))
        y_max = int(min(h, y_min_raw + box_h + margin_h))

        # Make crop square to prevent distortion when resizing to 224x224
        crop_w = x_max - x_min
        crop_h = y_max - y_min
        if crop_w != crop_h:
            diff = abs(crop_w - crop_h) // 2
            if crop_w < crop_h:
                x_min = max(0, x_min - diff)
                x_max = min(w, x_max + diff)
            else:
                y_min = max(0, y_min - diff)
                y_max = min(h, y_max + diff)

        face_crop = image[y_min:y_max, x_min:x_max].copy()
        return (x_min, y_min, x_max, y_max), face_crop

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect faces in an RGB numpy image using YuNet DNN and Haar Cascade.

        Args:
            image: RGB numpy array of shape (H, W, 3) and dtype uint8.

        Returns:
            List of FaceDetection objects sorted by confidence descending.
        """
        if self.bypass:
            return []

        h, w = image.shape[:2]
        detections: List[FaceDetection] = []

        # 1. Try OpenCV YuNet DNN Face Detector
        if self._yunet_detector is not None:
            try:
                # YuNet expects BGR input image
                bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                self._yunet_detector.setInputSize((w, h))
                _, faces = self._yunet_detector.detect(bgr_img)

                if faces is not None:
                    for face in faces:
                        box = face[:4]
                        score = float(face[-1])
                        fx, fy, fw, fh = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                        bbox, crop = self._crop_with_margin(image, fx, fy, fw, fh)
                        if crop.size > 0:
                            detections.append(FaceDetection(bbox=bbox, confidence=score, face_image=crop))
            except Exception as e:
                self.logger.warning(f"YuNet face detection error: {e}")

        # 2. Try OpenCV Haar Cascade if YuNet found nothing
        if not detections and self._haar_cascade is not None:
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                faces = self._haar_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                for (x, y, fw, fh) in faces:
                    bbox, crop = self._crop_with_margin(image, x, y, fh, fh)
                    if crop.size > 0:
                        detections.append(FaceDetection(bbox=bbox, confidence=0.8, face_image=crop))
            except Exception as e:
                self.logger.warning(f"Haar cascade face detection error: {e}")

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def detect_and_crop(self, image: np.ndarray) -> np.ndarray:
        """Detect the primary face and return tightly cropped face image.

        If no face is detected, crops central 80% square region.

        Args:
            image: RGB numpy array of shape (H, W, 3) and dtype uint8.

        Returns:
            Tightly cropped face as RGB numpy array.
        """
        if self.bypass:
            return image

        detections = self.detect(image)
        if detections:
            return detections[0].face_image

        # Fallback: Crop central 80% square region
        self.logger.warning("No face detected by detector, applying central face crop fallback")
        h, w = image.shape[:2]
        crop_size = int(min(h, w) * 0.8)
        start_y = (h - crop_size) // 2
        start_x = (w - crop_size) // 2
        return image[start_y : start_y + crop_size, start_x : start_x + crop_size].copy()

    def close(self) -> None:
        """Release resources."""
        self._yunet_detector = None
        self._haar_cascade = None

    def __del__(self) -> None:
        self.close()
