"""
FastAPI application for Qwen-Image-Edit service
"""
import os
import random
import base64
import io
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np

from .models import EditRequest, EditResponse, HealthResponse, StatusResponse
from .gpu_manager import EditMultiGPUManager

app = FastAPI(
    title="Qwen-Image-Edit API",
    description="FastAPI service for Qwen-Image-Edit with multi-GPU support",
    version="1.0.0"
)

# Global GPU manager instance
gpu_manager: Optional[EditMultiGPUManager] = None

MAX_SEED = np.iinfo(np.int32).max


def initialize_gpu_manager():
    """Initialize the global GPU manager"""
    import torch
    global gpu_manager
    if gpu_manager is None:
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            
            num_gpus = int(os.environ.get("NUM_GPUS_TO_USE", torch.cuda.device_count()))
            task_queue_size = int(os.environ.get("TASK_QUEUE_SIZE", 100))
            model_repo_id = os.environ.get("MODEL_REPO_ID", "Qwen/Qwen-Image-Edit")
            
            gpu_manager = EditMultiGPUManager(
                model_repo_id=model_repo_id,
                num_gpus=num_gpus,
                task_queue_size=task_queue_size
            )
            gpu_manager.start_workers()
            print("GPU Manager initialized successfully")
        except Exception as e:
            print(f"GPU Manager initialization failed: {e}")
            raise


def pil_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def validate_image(image: Image.Image) -> Image.Image:
    """Validate and convert image to RGB"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image


@app.on_event("startup")
async def startup_event():
    """Initialize GPU manager on startup"""
    try:
        initialize_gpu_manager()
    except Exception as e:
        print(f"Failed to initialize GPU manager: {e}")
        # Don't raise here, let health check handle it


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global gpu_manager
    if gpu_manager:
        gpu_manager.stop()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if gpu_manager is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "version": "1.0.0", "error": "GPU manager not initialized"}
        )
    
    return HealthResponse(status="healthy", version="1.0.0")


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get system status"""
    if gpu_manager is None:
        raise HTTPException(status_code=503, detail="GPU manager not initialized")
    
    status = gpu_manager.get_queue_status()
    return StatusResponse(**status)


@app.post("/edit", response_model=EditResponse)
async def edit_image(
    image: UploadFile = File(..., description="Input image file"),
    prompt: str = Form(..., description="Edit instruction prompt"),
    seed: Optional[int] = Form(None, description="Random seed for reproducibility"),
    randomize_seed: Optional[bool] = Form(False, description="Whether to randomize the seed"),
    true_guidance_scale: Optional[float] = Form(4.0, description="True guidance scale"),
    num_inference_steps: Optional[int] = Form(50, description="Number of inference steps"),
    num_images_per_prompt: Optional[int] = Form(1, description="Number of images to generate per prompt")
):
    """
    Edit an image based on the provided prompt.
    
    Supports both single image editing with various parameters.
    """
    if gpu_manager is None:
        raise HTTPException(status_code=503, detail="GPU manager not initialized")
    
    try:
        # Read and validate image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data))
        pil_image = validate_image(pil_image)
        
        # Determine seed
        if randomize_seed:
            seed = random.randint(0, MAX_SEED)
        elif seed is None:
            seed = 42
        
        # Validate parameters
        true_guidance_scale = max(1.0, min(10.0, true_guidance_scale))
        num_inference_steps = max(1, min(50, num_inference_steps))
        num_images_per_prompt = max(1, min(4, num_images_per_prompt))
        
        # Submit task to GPU manager
        result = gpu_manager.submit_task(
            image=pil_image,
            prompt=prompt,
            negative_prompt=" ",
            seed=seed,
            true_guidance_scale=true_guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            timeout=int(os.environ.get("TASK_TIMEOUT", 300))
        )
        
        if result['success']:
            # Convert images to base64
            images_base64 = [pil_to_base64(img) for img in result['images']]
            
            return EditResponse(
                success=True,
                images=images_base64,
                seed=seed,
                gpu_id=result.get('gpu_id')
            )
        else:
            return EditResponse(
                success=False,
                error=result.get('error', 'Unknown error')
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/edit/json", response_model=EditResponse)
async def edit_image_json(request: EditRequest, image_base64: str = Form(...)):
    """
    Edit an image based on the provided prompt (JSON format with base64 image).
    
    Alternative endpoint that accepts base64 encoded image.
    """
    if gpu_manager is None:
        raise HTTPException(status_code=503, detail="GPU manager not initialized")
    
    try:
        # Decode base64 image
        if image_base64.startswith("data:image"):
            image_base64 = image_base64.split(",")[1]
        
        image_data = base64.b64decode(image_base64)
        pil_image = Image.open(io.BytesIO(image_data))
        pil_image = validate_image(pil_image)
        
        # Determine seed
        seed = request.seed
        if request.randomize_seed:
            seed = random.randint(0, MAX_SEED)
        elif seed is None:
            seed = 42
        
        # Use request parameters with defaults
        true_guidance_scale = request.true_guidance_scale or 4.0
        num_inference_steps = request.num_inference_steps or 50
        num_images_per_prompt = request.num_images_per_prompt or 1
        
        # Validate parameters
        true_guidance_scale = max(1.0, min(10.0, true_guidance_scale))
        num_inference_steps = max(1, min(50, num_inference_steps))
        num_images_per_prompt = max(1, min(4, num_images_per_prompt))
        
        # Submit task to GPU manager
        result = gpu_manager.submit_task(
            image=pil_image,
            prompt=request.prompt,
            negative_prompt=" ",
            seed=seed,
            true_guidance_scale=true_guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            timeout=int(os.environ.get("TASK_TIMEOUT", 300))
        )
        
        if result['success']:
            # Convert images to base64
            images_base64 = [pil_to_base64(img) for img in result['images']]
            
            return EditResponse(
                success=True,
                images=images_base64,
                seed=seed,
                gpu_id=result.get('gpu_id')
            )
        else:
            return EditResponse(
                success=False,
                error=result.get('error', 'Unknown error')
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

