from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from app.routes import predict
from app.config import BASE_DIR

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

# Serve the live camera HTML page
@app.get("/live")
async def live_camera():
    html_path = BASE_DIR / "frontend" / "live_camera.html"
    if html_path.exists():
        return FileResponse(html_path)
    return {"error": "live_camera.html not found"}