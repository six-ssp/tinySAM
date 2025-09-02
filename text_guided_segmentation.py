import torch
import clip
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image                      # 关键：供 CLIP 使用
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.modeling.prompt_encoder import PromptEncoder
from segment_anything.modeling.mask_decoder import MaskDecoder


# ---------- 自定义轻量骨干网络 ----------
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
        x = torch.cat([x, x], dim=1)        # 128 → 256
        return x


# ---------- 1. 构建 TinySAM ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt_path = "tiny_sam.pth"
state_dict = torch.load(ckpt_path, map_location=device)

sam = sam_model_registry["vit_b"](checkpoint=None).to(device)
sam.image_encoder = TinyBackbone(img_size=1024)

# 加载 prompt_encoder 权重（去掉前缀）
pe_dict = {k[len("prompt_encoder."):]: v
           for k, v in state_dict.items()
           if k.startswith("prompt_encoder.")}
sam.prompt_encoder.load_state_dict(pe_dict)

# 加载 mask_decoder 权重（去掉前缀）
md_dict = {k[len("mask_decoder."):]: v
           for k, v in state_dict.items()
           if k.startswith("mask_decoder.")}
sam.mask_decoder.load_state_dict(md_dict)

predictor = SamPredictor(sam)

# ---------- 2. 加载 CLIP ----------
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# ---------- 3. 读图 ----------
image_path = r"images/complex_scene.jpg"
image_bgr = cv2.imread(image_path)
assert image_bgr is not None, f"找不到图片：{image_path}"
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# 统一 1024×1024
image_resized = cv2.resize(image_rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
image_pil = Image.fromarray(image_resized)  # ← 转成 PIL，供 CLIP 使用

# ---------- 4. 文本提示生成伪点 ----------
text_prompt = "black swan"
text = clip.tokenize([text_prompt]).to(device)

image_clip = preprocess(image_pil).unsqueeze(0).to(device)
with torch.no_grad():
    text_feat = clip_model.encode_text(text)
    img_feat  = clip_model.encode_image(image_clip)
    sim = torch.nn.functional.cosine_similarity(text_feat, img_feat).item()

# 简化：中心点
point_coords = np.array([[512, 512]])
point_labels = np.array([1])

# ---------- 5. TinySAM 分割 ----------
predictor.set_image(image_resized)
masks, _, _ = predictor.predict(
    point_coords=point_coords,
    point_labels=point_labels,
    multimask_output=False,
)

# ---------- 6. 可视化 ----------
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(image_resized)
plt.scatter(point_coords[:, 0], point_coords[:, 1], c='red', s=100)
plt.title(f"Text Prompt: '{text_prompt}'\nCLIP score={sim:.2f}")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(image_resized)
plt.imshow(masks[0], alpha=0.5, cmap='jet')
plt.title("TinySAM Segmentation")
plt.axis("off")

plt.tight_layout()
plt.show()