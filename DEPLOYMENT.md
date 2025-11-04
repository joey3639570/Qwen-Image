# Qwen-Image-Edit FastAPI Service Deployment Guide

## 概述

本指南說明如何在 Linux pod 上部署 Qwen-Image-Edit FastAPI 服務，支援多 GPU 並行處理。

## 系統要求

- **Python**: 3.12.11
- **CUDA**: 12.8
- **GPU**: NVIDIA GPU(s)，每個至少 16GB 記憶體
- **作業系統**: Linux
- **磁碟空間**: 至少 50GB（用於模型下載和緩存）

## 安裝步驟

### 1. 準備環境

確保 pod 已安裝：
- Python 3.12.11
- CUDA 12.8
- NVIDIA 驅動程式
- Git

檢查 CUDA 版本：
```bash
nvcc --version
```

檢查 GPU 可用性：
```bash
nvidia-smi
```

### 2. 上傳代碼

將項目代碼上傳到 pod 的目標目錄，例如：
```bash
cd /path/to/your/project
```

### 3. 安裝依賴

運行安裝腳本：
```bash
chmod +x install.sh
./install.sh
```

如果需要使用虛擬環境：
```bash
USE_VENV=true ./install.sh
```

安裝腳本會：
- 檢查 Python 和 CUDA 版本
- 安裝 PyTorch（CUDA 12.1 版本，兼容 CUDA 12.8）
- 安裝所有 Python 依賴
- 驗證安裝

### 4. 配置環境變數

複製環境變數示例文件：
```bash
cp .env.example .env
```

編輯 `.env` 文件，根據你的環境調整配置：
```bash
nano .env
```

重要配置項：
- `NUM_GPUS_TO_USE`: 使用的 GPU 數量（預設：所有可用 GPU）
- `TASK_QUEUE_SIZE`: 任務隊列大小（預設：100）
- `TASK_TIMEOUT`: 任務超時時間（秒，預設：300）
- `MODEL_REPO_ID`: 模型倉庫 ID（預設：Qwen/Qwen-Image-Edit）
- `HOST`: 服務監聽地址（預設：0.0.0.0）
- `PORT`: 服務端口（預設：8000）

### 5. 啟動服務

運行啟動腳本：
```bash
chmod +x start.sh
./start.sh
```

或者手動啟動：
```bash
python3 -m src.api.main
```

服務啟動後，你會看到：
- GPU 初始化訊息
- 服務監聽地址和端口
- API 文檔地址

### 6. 驗證服務

訪問健康檢查端點：
```bash
curl http://localhost:8000/health
```

訪問 API 文檔：
```bash
# 在瀏覽器中打開
http://your-pod-ip:8000/docs
```

## API 端點

### 1. 健康檢查

**GET** `/health`

檢查服務健康狀態。

**響應示例**:
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

### 2. 系統狀態

**GET** `/status`

獲取系統狀態，包括 GPU 使用情況和隊列狀態。

**響應示例**:
```json
{
  "active_workers": 4,
  "task_queue_size": 0,
  "result_queue_size": 0,
  "pending_tasks": 0,
  "total_gpus": 4,
  "gpu_info": [
    {
      "gpu_id": 0,
      "name": "NVIDIA A100",
      "memory_allocated": 12.5,
      "memory_reserved": 14.0
    }
  ]
}
```

### 3. 圖片編輯（Multipart Form）

**POST** `/edit`

使用 multipart/form-data 上傳圖片進行編輯。

**請求參數**:
- `image` (file, required): 輸入圖片文件
- `prompt` (string, required): 編輯指令提示
- `seed` (int, optional): 隨機種子
- `randomize_seed` (bool, optional): 是否隨機化種子（預設：false）
- `true_guidance_scale` (float, optional): 引導縮放係數（預設：4.0，範圍：1.0-10.0）
- `num_inference_steps` (int, optional): 推理步數（預設：50，範圍：1-50）
- `num_images_per_prompt` (int, optional): 每個提示生成的圖片數量（預設：1，範圍：1-4）

**使用 curl 示例**:
```bash
curl -X POST "http://localhost:8000/edit" \
  -F "image=@input.jpg" \
  -F "prompt=Change the background to a sunset scene" \
  -F "seed=42" \
  -F "true_guidance_scale=4.0" \
  -F "num_inference_steps=50"
```

**響應示例**:
```json
{
  "success": true,
  "images": [
    "data:image/png;base64,iVBORw0KGgoAAAANS..."
  ],
  "seed": 42,
  "gpu_id": 0
}
```

### 4. 圖片編輯（JSON + Base64）

**POST** `/edit/json`

使用 JSON 格式和 base64 編碼圖片進行編輯。

**請求體**:
```json
{
  "prompt": "Change the background to a sunset scene",
  "seed": 42,
  "randomize_seed": false,
  "true_guidance_scale": 4.0,
  "num_inference_steps": 50,
  "num_images_per_prompt": 1
}
```

**請求參數**:
- `image_base64` (form-data, required): Base64 編碼的圖片

**使用 curl 示例**:
```bash
IMAGE_BASE64=$(base64 -w 0 input.jpg)
curl -X POST "http://localhost:8000/edit/json" \
  -H "Content-Type: application/json" \
  -F "image_base64=data:image/jpeg;base64,$IMAGE_BASE64" \
  -d '{
    "prompt": "Change the background to a sunset scene",
    "seed": 42,
    "true_guidance_scale": 4.0,
    "num_inference_steps": 50
  }'
```

## 故障排除

### 問題 1: CUDA 不可用

**症狀**: 啟動時出現 "CUDA is not available" 錯誤

**解決方案**:
1. 檢查 NVIDIA 驅動程式：
   ```bash
   nvidia-smi
   ```
2. 檢查 CUDA 版本：
   ```bash
   nvcc --version
   ```
3. 重新安裝 PyTorch（CUDA 版本）：
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

### 問題 2: 模型下載失敗

**症狀**: GPU worker 初始化時模型下載失敗

**解決方案**:
1. 檢查網路連接
2. 檢查 HuggingFace 訪問權限
3. 手動設置 HuggingFace token（如需要）：
   ```bash
   export HF_TOKEN=your_token_here
   ```
4. 檢查磁碟空間是否足夠

### 問題 3: GPU 記憶體不足

**症狀**: 處理任務時出現 CUDA out of memory 錯誤

**解決方案**:
1. 減少使用的 GPU 數量（設置 `NUM_GPUS_TO_USE`）
2. 減少任務隊列大小（設置 `TASK_QUEUE_SIZE`）
3. 減少 `num_images_per_prompt` 參數
4. 減少 `num_inference_steps` 參數

### 問題 4: 任務超時

**症狀**: 請求返回超時錯誤

**解決方案**:
1. 增加 `TASK_TIMEOUT` 環境變數
2. 減少 `num_inference_steps` 參數
3. 檢查 GPU 是否正常工作

### 問題 5: 端口被占用

**症狀**: 啟動時出現端口占用錯誤

**解決方案**:
1. 更改 `PORT` 環境變數
2. 或停止占用端口的進程

## 性能優化

### 1. 調整 GPU 數量

根據你的硬體配置調整 `NUM_GPUS_TO_USE`：
- 更多 GPU = 更高的並發處理能力
- 但每個 GPU 需要至少 16GB 記憶體

### 2. 調整隊列大小

根據預期負載調整 `TASK_QUEUE_SIZE`：
- 更大的隊列 = 可以處理更多並發請求
- 但會占用更多記憶體

### 3. 調整推理參數

根據質量要求調整：
- `num_inference_steps`: 更多步數 = 更好質量但更慢
- `num_images_per_prompt`: 生成多張圖片需要更多時間和記憶體

## 監控和日誌

服務日誌會輸出到標準輸出，包括：
- GPU 初始化狀態
- 任務處理狀態
- 錯誤訊息

建議使用系統日誌工具（如 systemd、supervisor）來管理服務和日誌。

## 停止服務

使用 `Ctrl+C` 停止服務，或發送 SIGTERM 信號：
```bash
pkill -f "src.api.main"
```

服務會優雅地停止所有 GPU worker 進程。

## 更新服務

1. 停止當前服務
2. 更新代碼
3. 重新運行 `./install.sh`（如果需要更新依賴）
4. 重新啟動服務

## 安全建議

1. **不要將服務暴露在公網**：使用防火牆或反向代理
2. **使用 HTTPS**：在反向代理層配置 SSL/TLS
3. **限制訪問**：使用 IP 白名單或認證
4. **監控資源使用**：定期檢查 GPU 和記憶體使用情況

## 支援

如有問題，請檢查：
1. 服務日誌
2. GPU 狀態（`nvidia-smi`）
3. 系統資源使用情況

## 版本資訊

- API 版本: 1.0.0
- 支援的模型: Qwen/Qwen-Image-Edit
- Python: 3.12.11
- CUDA: 12.8

