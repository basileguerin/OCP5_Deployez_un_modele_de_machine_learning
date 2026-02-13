from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import JSONB
import json

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "db" / "schema.sql"
CSV_PATH = ROOT / "data" / "dataset_clean.csv"

DB_URL = "postgresql+psycopg2://hruser:hrpassword@localhost:5432/hrpredict"


engine = create_engine(DB_URL)

def run_schema():
    with engine.begin() as conn:
        conn.execute(text(SCHEMA_PATH.read_text(encoding="utf-8")))

def load_dataset():
    df = pd.read_csv(CSV_PATH)
    rows = [{"features": json.dumps(row)} for row in df.to_dict(orient="records")]

    with engine.begin() as conn:
        # Vider la table avant insertion
        conn.execute(text("TRUNCATE TABLE employees RESTART IDENTITY CASCADE"))

        # Insert
        conn.execute(
            text("INSERT INTO employees (features) VALUES (CAST(:features AS JSONB))"),
            rows
        )

if __name__ == "__main__":
    run_schema()
    load_dataset()
    print("DB created + dataset inserted.")
