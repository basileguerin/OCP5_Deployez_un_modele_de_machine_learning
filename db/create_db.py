from pathlib import Path
import os
import json

import pandas as pd
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "db" / "schema.sql"
CSV_PATH = ROOT / "data" / "dataset_clean.csv"

DEFAULT_LOCAL_PG = "postgresql+psycopg2://hruser:hrpassword@localhost:5432/hrpredict"
DB_URL = os.getenv("DATABASE_URL", DEFAULT_LOCAL_PG)

engine = create_engine(DB_URL)


def run_schema() -> None:
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with engine.begin() as conn:
        conn.execute(text(sql))
    print("Schema dropped & recreated.")


def load_dataset() -> None:
    if not CSV_PATH.exists():
        print("No dataset file found -> skip dataset load.")
        return

    df = pd.read_csv(CSV_PATH)
    rows = [{"features": json.dumps(row, ensure_ascii=False)} for row in df.to_dict(orient="records")]

    with engine.begin() as conn:
        # Postgres-only (TRUNCATE + JSONB)
        conn.execute(text("TRUNCATE TABLE employees RESTART IDENTITY CASCADE"))
        conn.execute(
            text("INSERT INTO employees (features) VALUES (CAST(:features AS JSONB))"),
            rows,
        )

    print(f"Dataset inserted: {len(rows)} rows into employees.")


def main():
    run_schema()
    load_dataset()
    print("DB created + dataset inserted.")


if __name__ == "__main__":
    main()