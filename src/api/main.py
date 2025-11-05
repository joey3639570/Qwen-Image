"""
Main entry point for the FastAPI application
"""
import os
import sys
import logging
import uvicorn
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

if __name__ == "__main__":
    # Get configuration from environment variables
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    workers = int(os.environ.get("WORKERS", 1))
    reload = os.environ.get("RELOAD", "false").lower() == "true"
    
    # Start the FastAPI application
    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level="info"
    )

