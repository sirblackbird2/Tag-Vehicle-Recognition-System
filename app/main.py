from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import predict

app = FastAPI(
    title="Vehicle Recognition API",
    description="Detect vehicles and read license plates",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(predict.router)

@app.get("/")
async def root():
    return {"message": "Vehicle Recognition API is running!"}