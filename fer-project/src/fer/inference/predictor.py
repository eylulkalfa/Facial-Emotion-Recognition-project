"""End-to-end emotion predictor using ONNX Runtime."""

from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import onnxruntime as ort
from PIL import Image

from fer.data.face_detector import FaceDetector
from fer.data.label_mapping import EMOTION_NAMES, IDX_TO_LABEL
from fer.data.transforms import get_val_transforms


class FERPredictor:
    """End-to-end emotion predictor handling preprocessing, face detection, and ONNX inference."""

    def __init__(self, model_path: str, bypass_face_detection: bool = False):
        """Initialize FERPredictor.

        Args:
            model_path: Path to the exported .onnx model file.
            bypass_face_detection: If True, skip face detection (for pre-cropped images).
        """
        self.model_path = str(Path(model_path).resolve())
        self.session = ort.InferenceSession(
            self.model_path, providers=["CPUExecutionProvider"]
        )
        self.face_detector = FaceDetector(bypass=bypass_face_detection)
        self.transform = get_val_transforms(224)

    def predict(
        self, image: Union[str, Path, np.ndarray, Image.Image]
    ) -> Dict[str, Any]:
        """Predict emotion from an input image.

        Args:
            image: Image file path string/Path, RGB numpy array, or PIL Image.

        Returns:
            Dict containing:
                - "emotion": top predicted emotion label string
                - "confidence": top emotion probability float
                - "probabilities": dict mapping all 7 emotion names to probabilities
                - "cropped_face": cropped RGB numpy array
        """
        # Load / convert input image to RGB uint8 numpy array
        if isinstance(image, (str, Path)):
            img_pil = Image.open(str(image)).convert("RGB")
            img_np = np.array(img_pil, dtype=np.uint8)
        elif isinstance(image, Image.Image):
            img_pil = image.convert("RGB")
            img_np = np.array(img_pil, dtype=np.uint8)
        elif isinstance(image, np.ndarray):
            img_np = image.copy()
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Face detection and crop
        cropped_face = self.face_detector.detect_and_crop(img_np)

        # Resize to 224x224
        face_pil = Image.fromarray(cropped_face)
        resized_pil = face_pil.resize((224, 224), Image.LANCZOS)
        resized_np = np.array(resized_pil, dtype=np.uint8)

        # Apply validation transform (Normalize + ToTensor)
        transformed = self.transform(image=resized_np)
        tensor = transformed["image"]  # shape [3, 224, 224]

        # Add batch dimension and convert to numpy float32
        input_array = tensor.unsqueeze(0).numpy()  # shape [1, 3, 224, 224]

        # ONNX Runtime inference
        outputs = self.session.run(None, {"input": input_array})
        probs = outputs[0][0]  # shape [7]

        top_idx = int(np.argmax(probs))
        top_emotion = IDX_TO_LABEL.get(top_idx, EMOTION_NAMES[top_idx])
        top_confidence = float(probs[top_idx])

        probabilities_dict = {
            EMOTION_NAMES[i]: float(probs[i]) for i in range(len(probs))
        }

        return {
            "emotion": top_emotion,
            "confidence": top_confidence,
            "probabilities": probabilities_dict,
            "cropped_face": cropped_face,
        }

    def predict_batch(self, images: List[Any]) -> List[Dict[str, Any]]:
        """Predict emotion for a list of images."""
        return [self.predict(img) for img in images]

    def close(self) -> None:
        """Release face detector resources."""
        self.face_detector.close()
