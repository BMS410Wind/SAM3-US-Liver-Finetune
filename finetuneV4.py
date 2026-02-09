import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from torch.amp import autocast, GradScaler
from sam3.model_builder import build_sam3_image_model
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR

# --- 1. 路徑設定 ---
BASE_PATH = "/mnt/d/medsam3/sam3"
IMG_DIR = os.path.join(BASE_PATH, "images")
MASK_DIR = os.path.join(BASE_PATH, "masks")
SAVE_PATH = "sam3_liver_full_finetuned_v5_best.pth"

# --- 2. 改進的 Loss (增加 Focal Loss 處理類別不平衡) ---
class CombinedSegmentationLoss(nn.Module):
    """
    組合 Dice Loss + Focal Loss + BCE
    - Focal Loss: 處理前景/背景不平衡
    - Dice Loss: 直接優化分割指標
    - BCE: 提供穩定梯度
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def focal_loss(self, inputs, targets):
        """Focal Loss 減少簡單樣本權重"""
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()
    
    def dice_loss(self, inputs, targets, smooth=1e-5):
        """Dice Loss"""
        probs = torch.sigmoid(inputs)
        intersection = (probs * targets).sum()
        dice = (2. * intersection + smooth) / (probs.sum() + targets.sum() + smooth)
        return 1 - dice
    
    def forward(self, inputs, targets):
        if inputs.shape[-2:] != targets.shape[-2:]:
            inputs = F.interpolate(inputs, size=targets.shape[-2:], mode='bilinear', align_corners=False)
        
        # 組合損失：30% Focal + 50% Dice + 20% BCE
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        bce = F.binary_cross_entropy_with_logits(inputs, targets)
        
        return 0.3 * focal + 0.5 * dice + 0.2 * bce

# --- 3. 增強的數據集 (加入數據增強) ---
class LiverDataset(Dataset):
    def __init__(self, img_dir, mask_dir, is_train=True, size=(1008, 1008)):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.size = size
        self.is_train = is_train
        
        self.img_names = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg'))])
        print(f"📊 找到 {len(self.img_names)} 張影像")
        
        # 數據增強
        if is_train:
            self.aug_transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.3),
                T.RandomRotation(degrees=15),
                T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            ])
        else:
            self.aug_transform = None

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        mask_name = f"mask_{img_name}"
        
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"找不到對應的 Mask: {mask_path}")

        # 讀取圖像
        image = Image.open(img_path).convert("L").resize(self.size, Image.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize(self.size, Image.NEAREST)
        
        # 數據增強（同時對 image 和 mask 進行相同的變換）
        if self.is_train and self.aug_transform:
            seed = np.random.randint(2147483647)
            
            # 對圖像和 mask 使用相同的隨機種子
            torch.manual_seed(seed)
            image = self.aug_transform(image)
            torch.manual_seed(seed)
            mask = self.aug_transform(mask)
        
        # 轉換為張量
        image_tensor = T.ToTensor()(image).repeat(3, 1, 1)  # L -> RGB
        mask_tensor = T.ToTensor()(mask)
        
        # 對比度增強（僅對圖像）
        if self.is_train:
            image_tensor = T.RandomAutocontrast(p=0.3)(image_tensor)
            image_tensor = T.ColorJitter(brightness=0.2, contrast=0.2)(image_tensor)
        
        return {
            "image": image_tensor,
            "mask": mask_tensor
        }

    
# --- 4. 改進的訓練邏輯 ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  使用設備: {device}")
    
    # 載入模型
    model = build_sam3_image_model().to(device)
    
    # 3. 【修正點】在這裡載入權重
    checkpoint_path = "finetune_weight/sam3_liver_full_finetuned_v5_best.pth"
    if os.path.exists(checkpoint_path):
        print(f"📦 正在載入權重: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✅ 權重載入成功！")
    else:
        print("⚠️ 找不到權重檔，將從頭開始訓練。")
        
    # 🔧 改進的解凍策略：逐步解凍
    model.requires_grad_(False)
    model.segmentation_head.requires_grad_(True)
    
    # 解凍最後3個 Backbone Levels（而非2個）
    if hasattr(model.backbone, 'model'):
        for param in model.backbone.model.levels[-1].parameters(): 
            param.requires_grad = True
        for param in model.backbone.model.levels[-2].parameters(): 
            param.requires_grad = True
        for param in model.backbone.model.levels[-3].parameters(): 
            param.requires_grad = True
    
    # 🔧 差異化學習率
    params = [
        {'params': model.segmentation_head.parameters(), 'lr': 1e-3},  # Head 高學習率
        {'params': [p for n, p in model.backbone.named_parameters() if p.requires_grad], 'lr': 1e-4}  # Backbone 低學習率
    ]
    
    optimizer = torch.optim.AdamW(params, weight_decay=0.01)
    
    # 🔧 OneCycleLR 替代 CosineAnnealing（更快收斂）
    EPOCHS = 500  # 減少 epoch 數，因為有數據增強
    BATCH_SIZE = 2  # RTX 3080 10GB 適配（1008x1008 圖像較大）
    
    dataloader = DataLoader(
        LiverDataset(IMG_DIR, MASK_DIR, is_train=True), 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=2,  # 3080 優化：減少 CPU 負載
        pin_memory=True,
        persistent_workers=True  # 保持 worker 活躍，減少重啟開銷
    )
    
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=[1e-3, 1e-4], 
        epochs=EPOCHS, 
        steps_per_epoch=len(dataloader),
        pct_start=0.3,  # 前30% epoch 學習率上升
        div_factor=25,
        final_div_factor=1000
    )
    
    criterion = CombinedSegmentationLoss()
    scaler = GradScaler('cuda')
    
    # 追蹤最佳模型
    best_loss = float('inf')

    print("🚀 開始訓練...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
            
            with autocast('cuda'):
                features = model.backbone(images, ["liver"])
                real_feat = features['vision_features'] if isinstance(features, dict) else features
                pred_logits = model.segmentation_head.semantic_seg_head(real_feat)
                loss = criterion(pred_logits, masks)
            
            scaler.scale(loss).backward()
            
            # 梯度裁剪防止爆炸
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()  # OneCycleLR 每個 batch 更新
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"💾 保存最佳模型 (Loss: {avg_loss:.5f})")
        
        if (epoch + 1) % 5 == 0:
            current_lr = scheduler.get_last_lr()
            print(f"🔥 Epoch {epoch+1:03d} | Loss: {avg_loss:.5f} | LR: {current_lr[0]:.2e}/{current_lr[1]:.2e}")

    print(f"✅ 訓練完成！最佳模型儲存至: {SAVE_PATH} (Loss: {best_loss:.5f})")

if __name__ == "__main__":
    train()