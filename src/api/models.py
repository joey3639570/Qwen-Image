"""
Pydantic models for FastAPI request/response schemas
"""
from typing import Optional, List
from pydantic import BaseModel, Field
import numpy as np

MAX_SEED = np.iinfo(np.int32).max


class EditRequest(BaseModel):
    """Request model for image editing"""
    prompt: str = Field(..., description="Edit instruction prompt")
    seed: Optional[int] = Field(None, ge=0, le=MAX_SEED, description="Random seed for reproducibility")
    randomize_seed: Optional[bool] = Field(False, description="Whether to randomize the seed")
    true_guidance_scale: Optional[float] = Field(4.0, ge=1.0, le=10.0, description="True guidance scale")
    num_inference_steps: Optional[int] = Field(50, ge=1, le=50, description="Number of inference steps")
    num_images_per_prompt: Optional[int] = Field(1, ge=1, le=4, description="Number of images to generate per prompt")


class EditResponse(BaseModel):
    """Response model for image editing"""
    success: bool = Field(..., description="Whether the request was successful")
    images: Optional[List[str]] = Field(None, description="Base64 encoded edited images")
    seed: Optional[int] = Field(None, description="Seed used for generation")
    gpu_id: Optional[int] = Field(None, description="GPU ID that processed the task")
    error: Optional[str] = Field(None, description="Error message if request failed")


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")


class StatusResponse(BaseModel):
    """Response model for system status"""
    active_workers: int = Field(..., description="Number of active GPU workers")
    task_queue_size: int = Field(..., description="Current task queue size")
    result_queue_size: int = Field(..., description="Current result queue size")
    pending_tasks: int = Field(..., description="Number of pending tasks")
    total_gpus: int = Field(..., description="Total number of GPUs")
    gpu_info: Optional[List[dict]] = Field(None, description="GPU information")

