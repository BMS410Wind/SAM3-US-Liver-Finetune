#!/usr/bin/env python3
"""
完整診斷工具：檢查數據、模型、推理流程
"""
import os
import torch
import numpy as np
from PIL import Image
import cv2
from sam3.model_builder import build_sam3_image_model
import torch.nn.functional as F

print("="*70)
print("🔍 SAM3 肝臟分割完整診斷")
print("="*70)

# ==================== 1. 檢查訓練數據 ====================
print("\n1️⃣  檢查訓練數據...")

IMG_DIR = "/mnt/d/medsam3/sam3/images"
MASK_DIR = "/mnt/d/medsam3/sam3/masks"

if not os.path.exists(IMG_DIR) or not os.path.exists(MASK_DIR):
    print(f"   ❌ 找不到數據目錄！")
    print(f"   請確認路徑:")
    print(f"      圖像: {IMG_DIR}")
    print(f"      遮罩: {MASK_DIR}")
else:
    images = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.png')])
    masks = sorted([f for f in os.listdir(MASK_DIR) if f.startswith('mask_')])
    
    print(f"   圖像數量: {len(images)}")
    print(f"   遮罩數量: {len(masks)}")
    
    if len(images) != len(masks):
        print(f"   ⚠️  數量不匹配！")
    
    # 檢查前5張
    print(f"\n   檢查前 5 張數據...")
    issues = []
    
    for i in range(min(5, len(images))):
        img_name = images[i]
        mask_name = f"mask_{img_name}"
        
        img_path = os.path.join(IMG_DIR, img_name)
        mask_path = os.path.join(MASK_DIR, mask_name)
        
        if not os.path.exists(mask_path):
            issues.append(f"{img_name}: 找不到對應 mask")
            continue
        
        # 讀取
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        mask_arr = np.array(mask)
        
        # 檢查
        unique_vals = np.unique(mask_arr)
        white_ratio = (mask_arr > 128).sum() / mask_arr.size
        
        status = "✅"
        notes = []
        
        if len(unique_vals) > 3:
            status = "⚠️"
            notes.append(f"不是純二值 ({len(unique_vals)} 個值)")
        
        if white_ratio < 0.05:
            status = "⚠️"
            notes.append(f"肝臟太少 ({white_ratio*100:.1f}%)")
        elif white_ratio > 0.7:
            status = "⚠️"
            notes.append(f"肝臟太多 ({white_ratio*100:.1f}%)")
        
        if mask.mode not in ['L', '1']:
            status = "⚠️"
            notes.append(f"模式異常 ({mask.mode})")
        
        print(f"   {status} {img_name[:30]:30s} | 肝臟: {white_ratio*100:5.1f}% | {' '.join(notes) if notes else 'OK'}")
        
        if notes:
            issues.append(f"{img_name}: {', '.join(notes)}")
    
    if issues:
        print(f"\n   ⚠️  發現 {len(issues)} 個潛在問題")
    else:
        print(f"\n   ✅ 數據檢查通過！")

# ==================== 2. 檢查模型權重 ====================
print(f"\n2️⃣  檢查模型權重...")

CHECKPOINT = "finetune_weight/sam3_liver_full_finetuned_v5_best.pth"

if not os.path.exists(CHECKPOINT):
    print(f"   ❌ 找不到權重: {CHECKPOINT}")
else:
    checkpoint = torch.load(CHECKPOINT, map_location='cpu')
    
    if isinstance(checkpoint, dict):
        keys = list(checkpoint.keys())
        print(f"   格式: 字典 (keys: {keys[:5]}...)")
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"   ✅ 標準格式")
            print(f"      Epoch: {checkpoint.get('epoch', 'N/A')}")
            print(f"      Loss: {checkpoint.get('loss', 'N/A')}")
        else:
            state_dict = checkpoint
            print(f"   格式: 直接 state_dict")
    else:
        state_dict = checkpoint
        print(f"   格式: 直接 state_dict")
    
    # 檢查參數數量
    total_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
    print(f"   總參數: {total_params:,}")
    
    # 檢查是否有 segmentation_head
    has_head = any('segmentation_head' in k for k in state_dict.keys())
    has_backbone = any('backbone' in k for k in state_dict.keys())
    
    print(f"   包含 segmentation_head: {'✅' if has_head else '❌'}")
    print(f"   包含 backbone: {'✅' if has_backbone else '❌'}")

# ==================== 3. 測試模型推理 ====================
print(f"\n3️⃣  測試模型推理...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   設備: {device}")

try:
    model = build_sam3_image_model().to(device)
    model.load_state_dict(state_dict if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint else checkpoint['model_state_dict'])
    model.eval()
    print(f"   ✅ 模型載入成功")
    
    # 測試不同尺寸
    test_sizes = [(1008, 1008), (896, 896)]
    working_size = None
    
    for size in test_sizes:
        print(f"\n   測試 {size}...")
        try:
            dummy = torch.randn(1, 3, size[0], size[1]).to(device)
            
            with torch.no_grad():
                features = model.backbone(dummy, ["liver"])
                feat = features['vision_features'] if isinstance(features, dict) else features
                logits = model.segmentation_head.semantic_seg_head(feat)
                probs = torch.sigmoid(logits)
            
            print(f"      ✅ 推理成功！")
            print(f"         Logits 範圍: [{logits.min():.2f}, {logits.max():.2f}]")
            print(f"         概率範圍: [{probs.min():.4f}, {probs.max():.4f}]")
            print(f"         概率平均: {probs.mean():.4f}")
            
            # 診斷
            if probs.mean() > 0.45 and probs.mean() < 0.55:
                print(f"      ⚠️  概率接近 0.5 → 模型可能沒有學習到有效特徵")
            elif probs.min() > 0.4 or probs.max() < 0.6:
                print(f"      ⚠️  概率範圍太窄 → 模型不確定")
            else:
                print(f"      ✅ 模型輸出正常")
            
            working_size = size
            break
            
        except Exception as e:
            print(f"      ❌ 失敗: {type(e).__name__}")
    
    if working_size:
        print(f"\n   💡 推薦推理尺寸: {working_size}")
    
except Exception as e:
    print(f"   ❌ 模型載入失敗: {e}")
    import traceback
    traceback.print_exc()

# ==================== 4. 測試真實圖像推理 ====================
print(f"\n4️⃣  測試真實圖像...")

TEST_IMG = "/mnt/d/medsam3/sam3/s0/s0_017_ 12489837_ 尤文邦_1_1.png"

if os.path.exists(TEST_IMG) and working_size:
    try:
        from torchvision import transforms as T
        
        img_pil = Image.open(TEST_IMG).convert("L")
        img_resized = img_pil.resize(working_size, Image.BILINEAR)
        img_tensor = T.ToTensor()(img_resized).repeat(3, 1, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = model.backbone(img_tensor, ["liver"])
            feat = features['vision_features'] if isinstance(features, dict) else features
            logits = model.segmentation_head.semantic_seg_head(feat)
            probs = F.interpolate(logits, size=working_size, mode='bilinear', align_corners=False)
            prob_map = torch.sigmoid(probs)[0, 0].cpu().numpy()
        
        print(f"   ✅ 推理成功")
        print(f"      概率範圍: [{prob_map.min():.4f}, {prob_map.max():.4f}]")
        print(f"      概率平均: {prob_map.mean():.4f}")
        print(f"      >0.5 像素: {(prob_map > 0.5).sum() / prob_map.size * 100:.1f}%")
        print(f"      >0.7 像素: {(prob_map > 0.7).sum() / prob_map.size * 100:.1f}%")
        
        # 生成測試 mask
        os.makedirs("diagnostic_output", exist_ok=True)
        
        for thresh in [0.3, 0.5, 0.7]:
            mask = (prob_map > thresh).astype(np.uint8) * 255
            cv2.imwrite(f"diagnostic_output/test_thresh_{thresh}.png", mask)
        
        # 概率熱圖
        heatmap = (prob_map * 255).astype(np.uint8)
        cv2.imwrite("diagnostic_output/probability_heatmap.png", heatmap)
        
        print(f"\n   💾 已保存測試結果到 diagnostic_output/")
        print(f"      請檢查不同閾值的效果:")
        print(f"         - test_thresh_0.3.png")
        print(f"         - test_thresh_0.5.png")
        print(f"         - test_thresh_0.7.png")
        print(f"         - probability_heatmap.png (概率熱圖)")
        
    except Exception as e:
        print(f"   ❌ 推理失敗: {e}")
        import traceback
        traceback.print_exc()

# ==================== 總結 ====================
print(f"\n{'='*70}")
print(f"📊 診斷總結")
print(f"{'='*70}")
print(f"\n請檢查 diagnostic_output/ 目錄中的圖片:")
print(f"1. 如果所有閾值的 mask 都很差 → 模型沒有學習到有效特徵，需要重新訓練")
print(f"2. 如果某個閾值的 mask 還可以 → 調整推理閾值")
print(f"3. 如果概率熱圖顯示前景/背景顛倒 → 嘗試反轉概率")
print(f"4. 檢查訓練數據的 mask 是否正確標註")
print(f"\n{'='*70}")