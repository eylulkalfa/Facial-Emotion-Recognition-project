"""Gradio web application demo for Facial Emotion Recognition."""

import gradio as gr

from fer.inference.predictor import FERPredictor


def create_demo(
    model_path: str, bypass_face_detection: bool = False
) -> gr.Blocks:
    """Build Gradio web demo application.

    Args:
        model_path: Path to the exported .onnx model file.
        bypass_face_detection: If True, skip face detection.

    Returns:
        gr.Blocks instance.
    """
    predictor = FERPredictor(
        model_path, bypass_face_detection=bypass_face_detection
    )

    def predict_emotion(image):
        if image is None:
            return None, None
        res = predictor.predict(image)
        return res["cropped_face"], res["probabilities"]

    with gr.Blocks(title="Facial Emotion Recognition") as demo:
        gr.Markdown("# 🎭 Facial Emotion Recognition")
        gr.Markdown(
            "Upload an image or capture from webcam to detect facial emotions."
        )
        gr.Markdown(
            "**Supported emotions:** anger, disgust, fear, happiness, sadness, surprise, neutral"
        )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    label="Input Image",
                    type="numpy",
                    sources=["upload", "webcam"],
                )
                submit_btn = gr.Button("Predict Emotion", variant="primary")

            with gr.Column():
                cropped_output = gr.Image(
                    label="Detected Face Crop (Model Input)",
                    type="numpy",
                )
                label_output = gr.Label(
                    label="Predicted Emotion Probabilities",
                    num_top_classes=7,
                )

        submit_btn.click(
            fn=predict_emotion,
            inputs=image_input,
            outputs=[cropped_output, label_output],
        )

        image_input.change(
            fn=predict_emotion,
            inputs=image_input,
            outputs=[cropped_output, label_output],
        )

    return demo
