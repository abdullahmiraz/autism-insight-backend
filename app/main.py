from fastapi import FastAPI
from app.routers import detection, progress, therapy

app = FastAPI()

# Include routers
app.include_router(detection.router,   tags=["Detection"])
app.include_router(progress.router,   tags=["Progress"])
app.include_router(therapy.router,   tags=["Therapy"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the Autism Detection API"}
