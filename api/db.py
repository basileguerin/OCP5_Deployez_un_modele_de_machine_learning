import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DEFAULT_LOCAL_PG = "postgresql+psycopg2://hruser:hrpassword@localhost:5432/hrpredict"
DB_URL = os.getenv("DATABASE_URL", DEFAULT_LOCAL_PG)

engine = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)