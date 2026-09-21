from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.assist import router as assist_router
from app.api.image import router as image_router

app = FastAPI(
    title="Orange API",
    version="1.0.0",
    description="Backend API for Orange AI"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(assist_router)
app.include_router(image_router)

@app.get("/")
def root():
    return {
        "message": "Welcome to Orange API"
    }

@app.get("/health")
def health():
    return {
        "status": "online"
    }