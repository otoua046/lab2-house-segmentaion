from io import BytesIO

from PIL import Image

from app.app import app


def make_test_image_bytes() -> BytesIO:
    image = Image.new("RGB", (256, 256), color="white")
    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_health_endpoint():
    client = app.test_client()
    response = client.get("/health")

    assert response.status_code == 200
    data = response.get_json()

    assert data is not None
    assert data["status"] == "ok"
    assert "model_path" in data


def test_predict_mask_missing_file():
    client = app.test_client()
    response = client.post("/predict-mask", data={}, content_type="multipart/form-data")

    assert response.status_code == 400
    data = response.get_json()

    assert data is not None
    assert "error" in data


def test_predict_mask_valid_image():
    client = app.test_client()

    data = {
        "image": (make_test_image_bytes(), "test.png"),
    }

    response = client.post("/predict-mask", data=data, content_type="multipart/form-data")

    assert response.status_code == 200
    payload = response.get_json()

    assert payload is not None
    assert payload["status"] == "ok"
    assert "prediction_path" in payload