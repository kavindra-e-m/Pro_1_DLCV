from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

from backend.api.routes import router
from backend.database.db import init_db

@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield

app = FastAPI(title="BagVision AI", version="1.0.0", description="Product-level handbag image retrieval and comparison", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])
app.mount("/media", StaticFiles(directory="data/processed/images", check_dir=False), name="media")
app.mount("/metrics", StaticFiles(directory="artifacts/metrics", check_dir=False), name="metrics")
app.mount("/dataset-reports", StaticFiles(directory="data/reports", check_dir=False), name="dataset-reports")
app.include_router(router, prefix="/api")

@app.get("/")
def root() -> dict[str, str]:
    return {"name": "BagVision AI", "health": "/api/health"}
