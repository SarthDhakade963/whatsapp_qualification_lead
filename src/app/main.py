# FastAPI / webhook entry point
# For now, just a placeholder

from fastapi import FastAPI
from .settings import Settings

settings = Settings()
app = FastAPI(title=settings.app_name)

@app.get("/")
def root():
    return {"message": "WhatsApp Lead Qualification API"}

