import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

URL = os.getenv("BAGVISION_DATABASE_URL", "sqlite:///./backend/data/bagvision.db")
if URL.startswith("sqlite") and URL.endswith("bagvision.db"):
    Path("backend/data").mkdir(parents=True, exist_ok=True)
engine = create_engine(URL, connect_args={"check_same_thread": False} if URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

class Base(DeclarativeBase):
    pass

def init_db() -> None:
    from backend.database.models import Image, Model, Product, SearchHistory  # noqa: F401
    Base.metadata.create_all(bind=engine)

def get_db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
