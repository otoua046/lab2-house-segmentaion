# Docker image definition.
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000 \
    MODEL_PATH=model/best_model.pth \
    APP_SECRET_KEY=4jhfedk44fdfd \
    INFERENCE_OUTPUT_DIR=outputs/inference

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY model ./model
COPY pytest.ini .
COPY tests ./tests

RUN mkdir -p outputs/inference

EXPOSE 5000

CMD ["python", "-m", "waitress", "--host=0.0.0.0", "--port=5000", "app.app:app"]