"""
FastAPI application for Qwen-Image-Edit service
"""
import os
import random
import base64
import io
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np

from .models import (
    EditRequest, EditResponse, HealthResponse, StatusResponse,
    MultiImageEditRequest, MultiImageEditResponse, EditModeInfoResponse
)
from .gpu_manager import EditMultiGPUManager
from .services.qwen_image_edit import MultiImageEditService

app = FastAPI(
    title="Qwen-Image-Edit API",
    description="FastAPI service for Qwen-Image-Edit with multi-GPU support",
    version="1.0.0"
)

# Global GPU manager instance
gpu_manager: Optional[EditMultiGPUManager] = None

# Global service instance
edit_service: Optional[MultiImageEditService] = None

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
            use_plus_pipeline = os.environ.get("USE_PLUS_PIPELINE", "false").lower() == "true"
            
            gpu_manager = EditMultiGPUManager(
                model_repo_id=model_repo_id,
                num_gpus=num_gpus,
                task_queue_size=task_queue_size,
                use_plus_pipeline=use_plus_pipeline
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
    """Initialize GPU manager and services on startup"""
    global edit_service
    try:
        initialize_gpu_manager()
        edit_service = MultiImageEditService()
        print("Services initialized successfully")
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
            images=pil_image,
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
            images=pil_image,
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


@app.post("/edit/multi", response_model=EditResponse)
async def edit_multi_images(
    images: List[UploadFile] = File(..., description="Input image files (1-3 images for multi-image editing)"),
    prompt: str = Form(..., description="Edit instruction prompt"),
    seed: Optional[int] = Form(None, description="Random seed for reproducibility"),
    randomize_seed: Optional[bool] = Form(False, description="Whether to randomize the seed"),
    true_guidance_scale: Optional[float] = Form(4.0, description="True guidance scale"),
    guidance_scale: Optional[float] = Form(1.0, description="Guidance scale (for plus pipeline)"),
    num_inference_steps: Optional[int] = Form(40, description="Number of inference steps"),
    num_images_per_prompt: Optional[int] = Form(1, description="Number of images to generate per prompt")
):
    """
    Edit multiple images (1-3) based on the provided prompt.
    
    This endpoint supports multi-image editing using Qwen-Image-Edit-2509.
    Requires USE_PLUS_PIPELINE=true and MODEL_REPO_ID=Qwen/Qwen-Image-Edit-2509.
    """
    if gpu_manager is None:
        raise HTTPException(status_code=503, detail="GPU manager not initialized")
    
    if not gpu_manager.use_plus_pipeline:
        raise HTTPException(
            status_code=400, 
            detail="Multi-image editing requires USE_PLUS_PIPELINE=true and Qwen-Image-Edit-2509 model"
        )
    
    if len(images) < 1 or len(images) > 3:
        raise HTTPException(
            status_code=400,
            detail="Multi-image editing supports 1-3 input images"
        )
    
    try:
        # Read and validate all images
        pil_images = []
        for img_file in images:
            image_data = await img_file.read()
            pil_image = Image.open(io.BytesIO(image_data))
            pil_image = validate_image(pil_image)
            pil_images.append(pil_image)
        
        # Determine seed
        if randomize_seed:
            seed = random.randint(0, MAX_SEED)
        elif seed is None:
            seed = 42
        
        # Validate parameters
        true_guidance_scale = max(1.0, min(10.0, true_guidance_scale))
        guidance_scale = max(1.0, min(10.0, guidance_scale))
        num_inference_steps = max(1, min(50, num_inference_steps))
        num_images_per_prompt = max(1, min(4, num_images_per_prompt))
        
        # Submit task to GPU manager
        result = gpu_manager.submit_task(
            images=pil_images,
            prompt=prompt,
            negative_prompt=" ",
            seed=seed,
            true_guidance_scale=true_guidance_scale,
            guidance_scale=guidance_scale,
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
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")


@app.post("/multi-image-editing", response_model=MultiImageEditResponse)
async def multi_image_editing(
    images: List[UploadFile] = File(..., description="Input image files (1-5 images)"),
    mode: int = Form(..., ge=1, le=16, description="編輯模式 (1-16)"),
    prompt: str = Form(..., description="Edit instruction prompt"),
    controlnet_image: Optional[UploadFile] = File(None, description="ControlNet condition image (for modes 13-16)"),
    seed: Optional[int] = Form(None, description="Random seed"),
    randomize_seed: Optional[bool] = Form(False, description="Whether to randomize the seed"),
    true_guidance_scale: Optional[float] = Form(None, description="True guidance scale"),
    guidance_scale: Optional[float] = Form(None, description="Guidance scale"),
    num_inference_steps: Optional[int] = Form(None, description="Number of inference steps"),
    num_images_per_prompt: Optional[int] = Form(None, description="Number of images to generate"),
    text_params_json: Optional[str] = Form(None, description="Text editing parameters as JSON (for modes 9-12)"),
    additional_params_json: Optional[str] = Form(None, description="Additional parameters as JSON")
):
    """
    多圖片編輯端點 - 支援 16 種編輯模式
    
    - 模式 1-5: 多圖編輯（需要 2-5 張圖片）
    - 模式 6-8: 單圖一致性編輯
    - 模式 9-12: 文字編輯（需要 text_params）
    - 模式 13-16: ControlNet 控制（需要 controlnet_image）
    """
    import json
    
    if gpu_manager is None:
        raise HTTPException(status_code=503, detail="GPU manager not initialized")
    
    if edit_service is None:
        raise HTTPException(status_code=503, detail="Edit service not initialized")
    
    # 檢查是否需要 Plus Pipeline
    if mode <= 5 and not gpu_manager.use_plus_pipeline:
        raise HTTPException(
            status_code=400,
            detail="多圖編輯模式 (1-5) 需要啟用 USE_PLUS_PIPELINE=true 並使用 Qwen-Image-Edit-2509 模型"
        )
    
    try:
        # 讀取並驗證圖片
        pil_images = []
        for img_file in images:
            image_data = await img_file.read()
            pil_image = Image.open(io.BytesIO(image_data))
            pil_image = validate_image(pil_image)
            pil_images.append(pil_image)
        
        # 讀取 ControlNet 條件圖（如果提供）
        controlnet_pil_image = None
        if controlnet_image:
            controlnet_data = await controlnet_image.read()
            controlnet_pil_image = Image.open(io.BytesIO(controlnet_data))
            controlnet_pil_image = validate_image(controlnet_pil_image)
        
        # 解析 JSON 參數
        text_params = None
        if text_params_json:
            try:
                text_params = json.loads(text_params_json)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid text_params JSON format")
        
        additional_params = None
        if additional_params_json:
            try:
                additional_params = json.loads(additional_params_json)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid additional_params JSON format")
        
        # 使用服務處理請求
        service_result = edit_service.process_multi_image_edit(
            mode=mode,
            images=pil_images,
            user_prompt=prompt,
            controlnet_image=controlnet_pil_image,
            text_params=text_params,
            additional_params=additional_params
        )
        
        if not service_result.get('success'):
            return MultiImageEditResponse(
                success=False,
                error=service_result.get('error', 'Unknown error')
            )
        
        # 獲取預設參數
        default_params = edit_service.get_default_parameters(mode)
        
        # 確定參數值（使用請求參數或預設值）
        if randomize_seed:
            seed = random.randint(0, MAX_SEED)
        elif seed is None:
            seed = default_params.get('seed', 42)
        
        true_guidance_scale = true_guidance_scale or default_params.get('true_guidance_scale', 4.0)
        guidance_scale = guidance_scale or default_params.get('guidance_scale', 1.0)
        num_inference_steps = num_inference_steps or default_params.get('num_inference_steps', 40)
        num_images_per_prompt = num_images_per_prompt or default_params.get('num_images_per_prompt', 1)
        
        # 驗證參數範圍
        true_guidance_scale = max(1.0, min(10.0, true_guidance_scale))
        guidance_scale = max(1.0, min(10.0, guidance_scale))
        num_inference_steps = max(1, min(50, num_inference_steps))
        num_images_per_prompt = max(1, min(4, num_images_per_prompt))
        
        # 生成增強後的提示詞
        enhanced_prompt = service_result.get('prompt', prompt)
        
        # TODO: 實作實際的模型 API 呼叫
        # 目前先返回模擬回應，待模型端點完成後替換為實際處理
        # 以下是實際處理的示例代碼（註解掉）：
        """
        # 提交任務到 GPU 管理器
        result = gpu_manager.submit_task(
            images=pil_images,
            prompt=enhanced_prompt,
            negative_prompt=" ",
            seed=seed,
            true_guidance_scale=true_guidance_scale,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            timeout=int(os.environ.get("TASK_TIMEOUT", 300))
        )
        
        if result['success']:
            images_base64 = [pil_to_base64(img) for img in result['images']]
            return MultiImageEditResponse(
                success=True,
                images=images_base64,
                seed=seed,
                gpu_id=result.get('gpu_id'),
                prompt=enhanced_prompt,
                mode=mode
            )
        else:
            return MultiImageEditResponse(
                success=False,
                error=result.get('error', 'Unknown error')
            )
        """
        
        # 模擬回應（待模型端點完成後移除）
        return MultiImageEditResponse(
            success=True,
            images=None,  # 實際處理時會填充圖片
            seed=seed,
            gpu_id=None,
            prompt=enhanced_prompt,
            mode=mode,
            error=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing multi-image editing: {str(e)}")


@app.get("/edit-mode/{mode}", response_model=EditModeInfoResponse)
async def get_edit_mode_info(mode: int):
    """獲取編輯模式資訊"""
    if edit_service is None:
        raise HTTPException(status_code=503, detail="Edit service not initialized")
    
    if mode < 1 or mode > 16:
        raise HTTPException(status_code=400, detail="編輯模式必須在 1-16 之間")
    
    mode_info = edit_service.get_mode_info(mode)
    if mode_info is None:
        raise HTTPException(status_code=404, detail=f"編輯模式 {mode} 不存在")
    
    return EditModeInfoResponse(**mode_info)


@app.get("/edit-modes", response_model=List[EditModeInfoResponse])
async def get_all_edit_modes():
    """獲取所有編輯模式資訊"""
    if edit_service is None:
        raise HTTPException(status_code=503, detail="Edit service not initialized")
    
    modes = []
    for mode in range(1, 17):
        mode_info = edit_service.get_mode_info(mode)
        if mode_info:
            modes.append(EditModeInfoResponse(**mode_info))
    
    return modes
