from fastapi import FastAPI
from app.api.assist import router as assist_router

app = FastAPI(
    title="Orange API",
    version="1.0.0",
    description="Backend API for Orange AI"
)

app.include_router(assist_router)

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