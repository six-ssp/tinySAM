import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import clip
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.modeling.prompt_encoder import PromptEncoder
from segment_anything.modeling.mask_decoder import MaskDecoder


# ---------- 轻量骨干 ----------
class TinyBackbone(torch.nn.Module):
    def __init__(self, img_size=1024):
        super().__init__()
        self.img_size = img_size
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, 2, 1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 32, 3, 2, 1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 64, 3, 2, 1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 128, 3, 2, 1),
        )

    def forward(self, x):
        x = self.layers(x)
        x = torch.cat([x, x], dim=1)   # 128→256
        return x


# ---------- 加载模型 ----------
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = "tiny_sam.pth"
    state = torch.load(ckpt, map_location=device)

    # 占位骨架
    sam = sam_model_registry["vit_b"](checkpoint=None).to(device)
    sam.image_encoder = TinyBackbone(img_size=1024)

    # prompt_encoder
    pe_dict = {k[len("prompt_encoder."):]: v
               for k, v in state.items()
               if k.startswith("prompt_encoder.")}
    sam.prompt_encoder.load_state_dict(pe_dict)

    # mask_decoder
    md_dict = {k[len("mask_decoder."):]: v
               for k, v in state.items()
               if k.startswith("mask_decoder.")}
    sam.mask_decoder.load_state_dict(md_dict)

    predictor = SamPredictor(sam)

    # CLIP
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    return predictor, clip_model, preprocess


predictor, clip_model, preprocess = load_models()

# ---------- Streamlit 前端 ----------
st.title("TinySAM 轻量图像分割 Web 应用")
uploaded = st.file_uploader("上传图片", type=["jpg", "png", "jpeg"])

if uploaded is not None:
    img_pil = Image.open(uploaded)
    img_np = np.array(img_pil)
    st.image(img_np, caption="原始图片", use_column_width=True)

    prompt = st.text_input("文本提示", value="black swan")
    if st.button("开始分割"):
        with st.spinner("分割中..."):
            # 统一 1024
            img_rgb = cv2.resize(img_np, (1024, 1024), interpolation=cv2.INTER_LINEAR)
            h, w = img_rgb.shape[:2]

            # CLIP 中心点（简化）
            point_coords = np.array([[w // 2, h // 2]])
            point_labels = np.array([1])

            # 分割
            predictor.set_image(img_rgb)
            masks, _, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,
            )

            # 可视化
            mask = masks[0]
            overlay = img_rgb.copy()
            overlay[mask] = [255, 0, 0]
            result = cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0)

            st.image(result, caption="分割结果", use_column_width=True)