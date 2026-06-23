import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.database.session import engine, Base
from src.api.routes import router as api_router
from src.utils.config import logger

# Initialize Database Tables
try:
    Base.metadata.create_all(bind=engine)
    logger.info("Successfully created database tables.")
except Exception as e:
    logger.critical(f"Failed to initialize database tables: {e}")
    raise

app = FastAPI(
    title="AI Resume Analyzer & Interview Coach API",
    description="Backend services for parsing resumes, calculating ATS compatibility, and generating coaching metrics.",
    version="1.0.0"
)

# CORS Middleware configurations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development. Restrict in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(api_router, prefix="/api")

@app.get("/")
def read_root():
    return {
        "app": "AI Resume Analyzer & Interview Coach API",
        "status": "online",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
