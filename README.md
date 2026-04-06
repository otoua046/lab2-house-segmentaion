# House Segmentation API (SEG4180 - Lab 2)

This project implements an end-to-end machine learning pipeline for **house segmentation from aerial images**, including:

- Dataset preparation (mask generation)
- Model training (U-Net)
- Evaluation (IoU, Dice)
- Inference API (Flask)
- Dockerized deployment
- CI/CD with GitHub Actions

---

## 🚀 Quick Start 

### 1. Pull Docker image

```bash
docker pull otoua046/lab2-house-segmentation:latest
```

### 2. Run container

```bash
docker run --rm -p 5000:5000 otoua046/lab2-house-segmentation:latest
```

### 3. Test API

```bash
curl -X POST http://localhost:5000/predict-mask \
  -F "image=@data/test/images/000014.png"
```

You should receive a JSON response and a segmentation mask will be generated.

---

## 📡 API Endpoints

### GET /health

Check if the service is running.

**Response:**
```json
{
  "status": "ok"
}
```

---

### POST /predict-mask

Upload an image and receive a predicted segmentation mask.

**Request:**
- `multipart/form-data`
- field name: `image`

**Response:**
```json
{
  "status": "ok",
  "prediction_path": "outputs/inference/xxx_mask.png"
}
```

---

## 🧠 Model Details

- Architecture: U-Net
- Input size: 256x256
- Task: Binary segmentation (house vs background)
- Loss: BCE + Dice

### Performance

| Metric | Value |
|------|------|
| IoU  | ~0.44 |
| Dice | ~0.56 |

---

## 📊 Dataset

Dataset: Satellite building segmentation dataset

Processing steps:
- Bounding boxes → pixel masks
- Dataset split into train / validation / test
- Mask generation inspired by SAM (Segment Anything Model)

---

## 🐳 Docker

Build locally:

```bash
docker build -t lab2-house-segmentation:latest .
```

Run with output persistence:

```bash
docker run --rm -p 5000:5000 \
  -v $(pwd)/outputs:/app/outputs \
  lab2-house-segmentation:latest
```

---

## 🔁 CI/CD Pipeline

GitHub Actions automatically:

- Runs tests (pytest)
- Builds Docker image
- Pushes to Docker Hub

Workflow file:

```
.github/workflows/ci-cd.yml
```

---

## 🧪 Testing

Run tests locally:

```bash
pytest tests/test_api.py -v
```

Smoke test:

```bash
./tests/smoke.sh
```

---

## 📁 Project Structure

```
app/            # API + inference
training/       # dataset + training pipeline
model/          # trained weights
tests/          # unit tests
outputs/        # predictions
```

---

## ⚠️ Notes

- Predictions are saved in `outputs/inference/`
- Model is included in the repository for reproducibility
- CI/CD requires Docker Hub credentials

---


## 🧾 Summary

This project demonstrates a complete ML lifecycle:

- Data → Model → API → Deployment → CI/CD

Focus is on **engineering reliability and reproducibility**, not just model accuracy.
