import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as T
from sam3.model_builder import build_sam3_image_model
import os
from tqdm import tqdm  # 建議安裝: pip install tqdm，方便看進度

# --- 1. 路徑設定 (Linux 格式) ---
input_dir = "/mnt/d/medsam3/sam3/s0"
output_mask_dir = "pred/masks"
output_overlay_dir = "pred/overlays"
checkpoint_path = "finetune_weight/sam3_liver_full_finetuned_v5_best.pth"

os.makedirs(output_mask_dir, exist_ok=True)
os.makedirs(output_overlay_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. 輔助函式 ---
def preprocess_image(path, size=(1008, 1008)):
    image_pil = Image.open(path).convert("RGB")
    image_resized = image_pil.resize(size, Image.BILINEAR)
    return T.ToTensor()(image_resized).unsqueeze(0).to(device)

def smooth_array(arr, window):
    # 加上 padding 避免邊界縮減，這也是圓潤的關鍵
    padded = np.pad(arr, (window // 2, window // 2), mode='wrap')
    return np.convolve(padded, np.ones(window)/window, mode='valid')

# --- 3. 載入模型 ---
model = build_sam3_image_model().to(device)
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"✅ 已載入權重: {checkpoint_path}")
model.eval()

# --- 4. 取得檔案列表 ---
valid_extensions = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
image_files = [f for f in os.listdir(input_dir) if f.endswith(valid_extensions)]

print(f"🚀 開始處理 {len(image_files)} 張圖片...")

# --- 5. 批次處理迴圈 ---
for filename in tqdm(image_files):
    img_path = os.path.join(input_dir, filename)
    
    # A. 推理
    input_tensor = preprocess_image(img_path)
    with torch.no_grad():
        features = model.backbone(input_tensor, ["liver"])
        real_feat = features['vision_features'] if isinstance(features, dict) else features
        pred_logits = model.segmentation_head.semantic_seg_head(real_feat)
        pred_masks = F.interpolate(pred_logits, size=(1008, 1008), mode='bilinear', align_corners=False)
        probs = torch.sigmoid(pred_masks)[0, 0].cpu().numpy()
        binary_mask = (probs > 0.5).astype(np.uint8) * 255

    # B. 形態學優化
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = cv2.erode(binary_mask, kernel, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(eroded, connectivity=8)
    
    if num_labels > 1:
        max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        main_mask = (labels == max_label).astype(np.uint8) * 255
    else:
        main_mask = eroded
    dilated = cv2.dilate(main_mask, kernel, iterations=1)

    # C. 提取輪廓並執行「點移動平均」圓潤化
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    refined_mask = np.zeros_like(dilated)
    smooth_contours = []
    
    window_size = 25 # 如果覺得不夠圓，可以調大到 35 或 45
    
    for cnt in contours:
        if len(cnt) < window_size: continue
        
        x, y = cnt[:, 0, 0], cnt[:, 0, 1]
        smooth_x = smooth_array(x.astype(float), window_size)
        smooth_y = smooth_array(y.astype(float), window_size)
        
        smoothed_cnt = np.stack([smooth_x, smooth_y], axis=1).astype(np.int32).reshape(-1, 1, 2)
        smooth_contours.append(smoothed_cnt)
        cv2.fillPoly(refined_mask, [smoothed_cnt], 255)

    # D. 疊圖處理
    orig_img = cv2.imread(img_path) 
    if orig_img is None:
        orig_img = np.array(Image.open(img_path).convert("RGB"))
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
    orig_img = cv2.resize(orig_img, (1008, 1008))

    overlay = orig_img.copy()
    red_layer = np.zeros_like(orig_img)
    red_layer[:] = [0, 0, 255] # 紅色遮罩
    
    mask_indices = refined_mask > 0
    overlay[mask_indices] = cv2.addWeighted(orig_img, 0.6, red_layer, 0.4, 0)[mask_indices]
    cv2.drawContours(overlay, smooth_contours, -1, (0, 255, 0), 2)

    # E. 儲存
    base_name = os.path.splitext(filename)[0]
    cv2.imwrite(os.path.join(output_mask_dir, f"{base_name}_mask.png"), refined_mask)
    cv2.imwrite(os.path.join(output_overlay_dir, f"{base_name}_overlay.png"), overlay)

print(f"\n✨ 任務完成！結果儲存在 pred/ 下。")
