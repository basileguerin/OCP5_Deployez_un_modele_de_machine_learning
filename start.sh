#!/usr/bin/env bash
set -e

echo "Init DB (create tables if needed)..."
python -c "from db.create_db import main; main()"

echo "Starting FastAPI..."
uvicorn api.main:app --host 127.0.0.1 --port 8000 &

echo "Starting Streamlit..."
exec streamlit run app.py \
  --server.address 0.0.0.0 \
  --server.port "${PORT:-7860}"