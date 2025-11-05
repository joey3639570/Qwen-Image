"""
Pydantic models for FastAPI request/response schemas
"""
from typing import Optional, List, Dict, Any
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


class MultiImageEditRequest(BaseModel):
    """Request model for multi-image editing"""
    mode: int = Field(..., ge=1, le=16, description="編輯模式 (1-16)")
    prompt: str = Field(..., description="編輯指令提示")
    seed: Optional[int] = Field(None, ge=0, le=MAX_SEED, description="Random seed")
    randomize_seed: Optional[bool] = Field(False, description="Whether to randomize the seed")
    true_guidance_scale: Optional[float] = Field(None, ge=1.0, le=10.0, description="True guidance scale")
    guidance_scale: Optional[float] = Field(None, ge=1.0, le=10.0, description="Guidance scale")
    num_inference_steps: Optional[int] = Field(None, ge=1, le=50, description="Number of inference steps")
    num_images_per_prompt: Optional[int] = Field(None, ge=1, le=4, description="Number of images to generate")
    text_params: Optional[Dict[str, Any]] = Field(None, description="文字編輯參數 (模式 9-12 需要)")
    additional_params: Optional[Dict[str, Any]] = Field(None, description="額外參數")


class MultiImageEditResponse(BaseModel):
    """Response model for multi-image editing"""
    success: bool = Field(..., description="Whether the request was successful")
    images: Optional[List[str]] = Field(None, description="Base64 encoded edited images")
    seed: Optional[int] = Field(None, description="Seed used for generation")
    gpu_id: Optional[int] = Field(None, description="GPU ID that processed the task")
    error: Optional[str] = Field(None, description="Error message if request failed")
    prompt: Optional[str] = Field(None, description="Generated/enhanced prompt")
    mode: Optional[int] = Field(None, description="Edit mode used")


class EditModeInfoResponse(BaseModel):
    """Response model for edit mode information"""
    mode: int = Field(..., description="Edit mode")
    name: str = Field(..., description="Mode name")
    description: str = Field(..., description="Mode description")
    requires_multiple_images: bool = Field(..., description="Whether multiple images are required")
    requires_controlnet: bool = Field(..., description="Whether ControlNet is required")
    requires_text_params: bool = Field(..., description="Whether text parameters are required")
    min_images: int = Field(..., description="Minimum number of images")
    max_images: int = Field(..., description="Maximum number of images")

