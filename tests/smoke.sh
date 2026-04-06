#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://localhost:5000}"
TEST_IMAGE="${2:-data/test/images/000014.png}"

echo "==> Health check"
curl -s "$BASE_URL/health"

echo
echo "==> Predict mask"
curl -s -X POST "$BASE_URL/predict-mask" \
  -F "image=@${TEST_IMAGE}"
echo