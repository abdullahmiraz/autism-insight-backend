from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# from app.routers import detection, progress, therapy
from app.routers import detection

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace * with specific frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(detection.router, tags=["Detection"])
# app.include_router(progress.router,   tags=["Progress"])
# app.include_router(therapy.router,   tags=["Therapy"])


@app.get("/")
def read_root():
    return {"message": "Welcome to the Autism Detection API"}
