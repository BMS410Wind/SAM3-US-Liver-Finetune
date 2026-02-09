import numpy as np
import gradio as gr
import os
import glob
import cv2
from PIL import Image

class AnnotatorState:
    def __init__(self):
        self.image_files = []
        self.current_points = []
        self.current_img_w = 0
        self.current_img_h = 0
        self.save_path = "" # 使用者自定義儲存路徑

state = AnnotatorState()

def load_folder(img_path, save_path):
    """載入圖片並設定儲存位置"""
    if not os.path.isdir(img_path):
        return gr.update(maximum=0), "❌ 圖片來源路徑無效", None, None
    
    # 建立儲存目錄（如果不存在）
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    state.save_path = save_path
    
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    state.image_files = []
    for e in exts:
        state.image_files.extend(glob.glob(os.path.join(img_path, e)))
    state.image_files.sort()
    
    if not state.image_files:
        return gr.update(maximum=0), "⚠️ 此資料夾內無圖片", None, None
    
    return gr.update(maximum=len(state.image_files)-1, value=0, interactive=True), f"✅ 已載入 {len(state.image_files)} 張圖片", *update_display(0)

def update_display(index):
    if not state.image_files: return None, None
    
    img_path = state.image_files[index]
    raw_img = Image.open(img_path).convert("RGB")
    state.current_img_w, state.current_img_h = raw_img.size
    state.current_points = []
    
    # 從使用者指定的儲存路徑讀取 Mask
    mask_name = f"mask_{os.path.basename(img_path)}"
    mask_full_path = os.path.join(state.save_path, mask_name)
    
    if os.path.exists(mask_full_path):
        mask_view = Image.open(mask_full_path).convert("L")
    else:
        mask_view = Image.fromarray(np.zeros((state.current_img_h, state.current_img_w), dtype=np.uint8))
        
    return raw_img, mask_view

def handle_click(index, img, evt: gr.SelectData):
    ix, iy = evt.index[0], evt.index[1]
    state.current_points.append([ix, iy])
    
    preview_img = np.array(img).copy()
    for i in range(len(state.current_points)):
        cv2.circle(preview_img, tuple(state.current_points[i]), 5, (255, 0, 0), -1)
        if i > 0:
            cv2.line(preview_img, tuple(state.current_points[i-1]), tuple(state.current_points[i]), (0, 255, 0), 2)
    return preview_img

def finalize_polygon(index):
    if len(state.current_points) < 3:
        return None, "❌ 至少需要 3 個頂點"
    
    mask = np.zeros((state.current_img_h, state.current_img_w), dtype=np.uint8)
    pts = np.array(state.current_points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    
    # 儲存到使用者指定的資料夾
    img_path = state.image_files[index]
    save_full_path = os.path.join(state.save_path, f"mask_{os.path.basename(img_path)}")
    Image.fromarray(mask).save(save_full_path)
    
    return mask, f"✅ 成功儲存至: {save_full_path}"

with gr.Blocks(title="Custom Path Annotator") as demo:
    gr.Markdown("# 🖋️ 專業多邊形標註工具 (自定義路徑版)")
    
    with gr.Row():
        with gr.Column():
            img_dir = gr.Textbox(label="1. 圖片來源路徑 (Source)", value="/mnt/d/medsam3/sam3/images")
        with gr.Column():
            save_dir = gr.Textbox(label="2. 標註儲存路徑 (Save To)", value="/mnt/d/medsam3/sam3/labels")
        btn_load = gr.Button("📂 載入並開始標註", variant="secondary")

    slider = gr.Slider(label="切換圖片", minimum=0, maximum=100, step=1, interactive=False)
    status_msg = gr.Markdown("請輸入路徑後點擊載入")

    with gr.Row():
        with gr.Column():
            view_raw = gr.Image(label="左：點擊原圖標註頂點", interactive=True)
            with gr.Row():
                btn_done = gr.Button("🎯 生成區域並儲存", variant="primary")
                btn_clear = gr.Button("🧹 清除頂點")
        with gr.Column():
            view_mask = gr.Image(label="右：生成的二值化 Mask")

    # 事件邏輯
    btn_load.click(load_folder, [img_dir, save_dir], [slider, status_msg, view_raw, view_mask])
    slider.change(update_display, slider, [view_raw, view_mask])
    view_raw.select(handle_click, [slider, view_raw], view_raw)
    btn_done.click(finalize_polygon, slider, [view_mask, status_msg])
    btn_clear.click(fn=lambda i: update_display(i)[0], inputs=slider, outputs=view_raw)

if __name__ == "__main__":
    demo.launch()