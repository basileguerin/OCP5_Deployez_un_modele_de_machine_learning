#!/usr/bin/env bash
set -e

echo "Init DB (create tables if needed)..."
python -c "from db.create_db import main; main()" || true

echo "Starting FastAPI..."
uvicorn api.main:app --host 127.0.0.1 --port 8000 &

echo "Starting Streamlit..."
streamlit run app.py \
    --server.port 7860 \
    --server.address 0.0.0.0