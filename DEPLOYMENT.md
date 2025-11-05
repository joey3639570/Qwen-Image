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
- `USE_PLUS_PIPELINE`: 是否使用 QwenImageEditPlusPipeline（預設：false）
  - 設為 `true` 以啟用多圖片編輯功能（需要 MODEL_REPO_ID=Qwen/Qwen-Image-Edit-2509）
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

## 功能特性

### 單圖片編輯
- 使用 `Qwen/Qwen-Image-Edit` 模型
- 端點：`POST /edit` 和 `POST /edit/json`
- 支援標準圖片編輯功能

### 多圖片編輯
- 使用 `Qwen/Qwen-Image-Edit-2509` 模型
- 端點：`POST /edit/multi`
- 支援 1-3 張圖片同時編輯
- 需要設置 `USE_PLUS_PIPELINE=true`
- 支援多種組合：person + person, person + product, person + scene

## API 端點

### 系統端點

#### 1. 健康檢查

**GET** `/health`

檢查服務健康狀態。

**響應示例**:
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

**狀態碼**:
- `200`: 服務正常
- `503`: 服務未初始化

---

#### 2. 系統狀態

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

---

### 編輯模式資訊端點

#### 3. 獲取編輯模式資訊

**GET** `/edit-mode/{mode}`

獲取指定編輯模式的詳細資訊。

**路徑參數**:
- `mode` (int, required): 編輯模式編號 (1-16)

**響應示例**:
```json
{
  "mode": 1,
  "name": "人物 + 人物",
  "description": "將兩個人物合併到同一個場景中",
  "requires_multiple_images": true,
  "requires_controlnet": false,
  "requires_text_params": false,
  "min_images": 2,
  "max_images": 5
}
```

---

#### 4. 獲取所有編輯模式

**GET** `/edit-modes`

獲取所有 16 種編輯模式的資訊列表。

**響應示例**:
```json
[
  {
    "mode": 1,
    "name": "人物 + 人物",
    "description": "將兩個人物合併到同一個場景中",
    ...
  },
  ...
]
```

---

### 圖片編輯端點

#### 5. 單圖片編輯 (Multipart Form)

**POST** `/edit`

使用 multipart/form-data 上傳單張圖片進行編輯。

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

---

#### 6. 單圖片編輯 (JSON + Base64)

**POST** `/edit/json`

使用 JSON 格式和 base64 編碼圖片進行編輯。

**請求參數**:
- `image_base64` (form-data, required): Base64 編碼的圖片（可包含 data URI 前綴）

**請求體** (JSON):
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

**使用 curl 示例**:
```bash
IMAGE_BASE64=$(base64 -w 0 input.jpg)
curl -X POST "http://localhost:8000/edit/json" \
  -F "image_base64=data:image/jpeg;base64,$IMAGE_BASE64" \
  -F 'prompt=Change the background to a sunset scene' \
  -F 'seed=42' \
  -F 'true_guidance_scale=4.0' \
  -F 'num_inference_steps=50'
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

---

#### 7. 多圖片編輯 (基礎版)

**POST** `/edit/multi`

使用 multipart/form-data 上傳多張圖片（1-3張）進行編輯。需要啟用 `USE_PLUS_PIPELINE=true` 並使用 `Qwen/Qwen-Image-Edit-2509` 模型。

**請求參數**:
- `images` (file[], required): 輸入圖片文件列表（1-3張）
- `prompt` (string, required): 編輯指令提示
- `seed` (int, optional): 隨機種子
- `randomize_seed` (bool, optional): 是否隨機化種子（預設：false）
- `true_guidance_scale` (float, optional): 引導縮放係數（預設：4.0，範圍：1.0-10.0）
- `guidance_scale` (float, optional): 引導縮放係數（用於 plus pipeline，預設：1.0，範圍：1.0-10.0）
- `num_inference_steps` (int, optional): 推理步數（預設：40，範圍：1-50）
- `num_images_per_prompt` (int, optional): 每個提示生成的圖片數量（預設：1，範圍：1-4）

**使用 curl 示例**:
```bash
curl -X POST "http://localhost:8000/edit/multi" \
  -F "images=@image1.jpg" \
  -F "images=@image2.jpg" \
  -F "prompt=The magician bear is on the left, the alchemist bear is on the right" \
  -F "seed=42" \
  -F "true_guidance_scale=4.0" \
  -F "guidance_scale=1.0" \
  -F "num_inference_steps=40"
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

**注意事項**:
- 需要設置環境變數：
  ```bash
  export USE_PLUS_PIPELINE=true
  export MODEL_REPO_ID=Qwen/Qwen-Image-Edit-2509
  ```
- 支援 1-3 張輸入圖片

---

#### 8. 多圖片編輯 (進階版 - 16種模式)

**POST** `/multi-image-editing`

進階多圖片編輯端點，支援 16 種編輯模式，最多 5 張圖片，支援 ControlNet 和文字編輯參數。

**請求參數**:
- `images` (file[], required): 輸入圖片文件列表（1-5張，根據模式調整）
- `mode` (int, required): 編輯模式 (1-16)
  - 模式 1-5: 多圖編輯（需要 2-5 張圖片）
  - 模式 6-8: 單圖一致性編輯（1 張圖片）
  - 模式 9-12: 文字編輯（需要 `text_params_json`）
  - 模式 13-16: ControlNet 控制（需要 `controlnet_image`）
- `prompt` (string, required): 編輯指令提示
- `controlnet_image` (file, optional): ControlNet 條件圖（模式 13-16 需要）
- `seed` (int, optional): 隨機種子
- `randomize_seed` (bool, optional): 是否隨機化種子（預設：false）
- `true_guidance_scale` (float, optional): 引導縮放係數（預設：根據模式自動設定）
- `guidance_scale` (float, optional): 引導縮放係數（預設：根據模式自動設定）
- `num_inference_steps` (int, optional): 推理步數（預設：根據模式自動設定）
- `num_images_per_prompt` (int, optional): 每個提示生成的圖片數量（預設：1）
- `text_params_json` (string, optional): 文字編輯參數 JSON（模式 9-12 需要）
  ```json
  {
    "old_text": "舊文字",
    "new_text": "新文字",
    "position": "center",
    "font_type": "Arial",
    "color": "red"
  }
  ```
- `additional_params_json` (string, optional): 額外參數 JSON
  ```json
  {
    "positions": ["left", "right"],
    "style": "realistic",
    "preserve_features": ["face", "hair"]
  }
  ```

**16 種編輯模式說明**:

| 模式 | 名稱 | 說明 | 最小圖片數 | 需要 ControlNet | 需要文字參數 |
|------|------|------|-----------|----------------|-------------|
| 1 | 人物 + 人物 | 將兩個人物合併到同一個場景中 | 2 | ❌ | ❌ |
| 2 | 人物 + 產品 | 將人物和產品組合在一起 | 2 | ❌ | ❌ |
| 3 | 人物 + 場景 | 將人物放置在場景中 | 2 | ❌ | ❌ |
| 4 | 產品 + 產品 | 將多個產品組合在一起 | 2 | ❌ | ❌ |
| 5 | 多物件組合 | 將多個物件組合在一起 | 2 | ❌ | ❌ |
| 6 | 人物一致性編輯 | 編輯人物時保持身份一致性 | 1 | ❌ | ❌ |
| 7 | 產品一致性編輯 | 編輯產品時保持產品一致性 | 1 | ❌ | ❌ |
| 8 | 風格轉換 | 將圖片轉換為不同風格 | 1 | ❌ | ❌ |
| 9 | 文字替換 | 替換圖片中的文字 | 1 | ❌ | ✅ |
| 10 | 文字添加 | 在圖片中添加文字 | 1 | ❌ | ✅ |
| 11 | 文字字體編輯 | 更改文字的字體 | 1 | ❌ | ✅ |
| 12 | 文字顏色編輯 | 更改文字的顏色 | 1 | ❌ | ✅ |
| 13 | 深度圖控制 | 使用深度圖控制生成結果 | 1 | ✅ | ❌ |
| 14 | 邊緣圖控制 | 使用邊緣圖控制生成結果 | 1 | ✅ | ❌ |
| 15 | 關鍵點控制 | 使用關鍵點控制生成結果 | 1 | ✅ | ❌ |
| 16 | 草圖控制 | 使用草圖控制生成結果 | 1 | ✅ | ❌ |

**使用 curl 示例** (模式 1: 人物 + 人物):
```bash
curl -X POST "http://localhost:8000/multi-image-editing" \
  -F "images=@person1.jpg" \
  -F "images=@person2.jpg" \
  -F "mode=1" \
  -F "prompt=Two people facing each other in a park" \
  -F "seed=42" \
  -F "true_guidance_scale=4.0" \
  -F "guidance_scale=1.0" \
  -F "num_inference_steps=40"
```

**使用 curl 示例** (模式 9: 文字替換):
```bash
curl -X POST "http://localhost:8000/multi-image-editing" \
  -F "images=@image_with_text.jpg" \
  -F "mode=9" \
  -F "prompt=Replace the text" \
  -F 'text_params_json={"old_text": "Hello", "new_text": "Hi"}' \
  -F "seed=42"
```

**使用 curl 示例** (模式 13: 深度圖控制):
```bash
curl -X POST "http://localhost:8000/multi-image-editing" \
  -F "images=@input.jpg" \
  -F "controlnet_image=@depth_map.png" \
  -F "mode=13" \
  -F "prompt=Follow the depth structure" \
  -F "seed=42"
```

**響應示例**:
```json
{
  "success": true,
  "images": [
    "data:image/png;base64,iVBORw0KGgoAAAANS..."
  ],
  "seed": 42,
  "gpu_id": 0,
  "prompt": "增強後的提示詞",
  "mode": 1
}
```

**注意事項**:
- 模式 1-5 需要設置：
  ```bash
  export USE_PLUS_PIPELINE=true
  export MODEL_REPO_ID=Qwen/Qwen-Image-Edit-2509
  ```
- 模式 9-12 必須提供 `text_params_json`
- 模式 13-16 必須提供 `controlnet_image`
- 目前返回模擬回應，待模型端點完成後將實作實際處理

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

