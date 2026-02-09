import torch
from PIL import Image
import numpy as np
import os
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# 1. 確保儲存目錄存在
os.makedirs("pred", exist_ok=True)

# 2. 載入模型 (請確保 checkpoint 路徑正確，若 build_sam3_image_model 有參數請加上)
print("Loading model...")
model = build_sam3_image_model() 
processor = Sam3Processor(model)

# 3. 載入影像
img_path = "images/car.jpg"
if not os.path.exists(img_path):
    raise FileNotFoundError(f"找不到影像檔案: {img_path}")

image = Image.open(img_path).convert("RGB")
inference_state = processor.set_image(image)

# 4. 進行文字提示預測 (建議 prompt 使用簡單的單字)
prompt_text = "car"
print(f"Running inference with prompt: '{prompt_text}'...")
output = processor.set_text_prompt(state=inference_state, prompt=prompt_text)

# 5. 取得輸出結果
masks = output["masks"]   # Shape 通常是 [N, 1, H, W]
boxes = output["boxes"]
scores = output["scores"]

print(f"Masks shape: {masks.shape}")
print(f"Scores: {scores.tolist()}")

# 6. 檢查是否偵測到物體
if masks.shape[0] == 0:
    print("!!! 警告：模型沒有偵測到任何物體，請檢查 Prompt 或模型權重。")
else:
    # 7. 處理與合併 Mask
    # 使用 torch.any 將所有偵測到的物件 (N) 合併成一個 boolean mask
    # dim=0 代表在物件維度做「或 (OR)」運算
    merged, _ = torch.max(masks, dim=0) # 或者用 masks.any(dim=0)
    
    # 轉為 Numpy 並移除多餘維度 (例如從 1, H, W 變成 H, W)
    merged_np = merged.detach().cpu().numpy().squeeze()
    
    # 檢查 merged_np 是否為全零
    if np.max(merged_np) == 0:
        print("!!! 警告：合併後的 Mask 為空（全黑）。")
    else:
        # 8. 轉換為 0-255 格式並儲存
        mask_img = (merged_np * 255).astype(np.uint8)
        save_path = "pred/merged_mask_car.png"
        Image.fromarray(mask_img).save(save_path)
        print(f"成功！Mask 已儲存至: {save_path}")
        print(f"偵測到 {len(scores)} 個候選框。")