"""Flask application entry point for the house segmentation API."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request
from PIL import Image
from werkzeug.utils import secure_filename


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "best_model.pth"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "inference"

app = Flask(__name__)


@app.get("/health")
def health() -> tuple[object, int]:
    """Return a simple JSON health payload."""
    return (
        jsonify(
            {
                "status": "ok",
                "service": "house-segmentation-api",
                "model_path": str(MODEL_PATH),
                "model_exists": MODEL_PATH.exists(),
            }
        ),
        200,
    )


@app.post("/predict-mask")
def predict_mask() -> tuple[object, int]:
    """Run segmentation on an uploaded image and save the predicted mask."""
    upload = request.files.get("image")
    if upload is None or upload.filename == "":
        return jsonify({"error": "Missing image upload. Send a file field named 'image'."}), 400

    try:
        image = Image.open(upload.stream).convert("RGB")
    except Exception as exc:
        return jsonify({"error": f"Failed to read uploaded image: {exc}"}), 400

    from .inference import predict

    mask = predict(image)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    original_name = secure_filename(upload.filename) or "uploaded_image.png"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    prediction_name = f"{Path(original_name).stem}_mask_{timestamp}.png"
    prediction_path = OUTPUT_DIR / prediction_name
    Image.fromarray(mask, mode="L").save(prediction_path)

    return (
        jsonify(
            {
                "status": "ok",
                "prediction_path": str(prediction_path),
            }
        ),
        200,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
