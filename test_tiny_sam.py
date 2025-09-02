import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

# ---------- 1. 用 ViT-B 骨架 + 蒸馏权重 ----------
device = "cpu"
ckpt_path = "sam_vit_b_distilled.pth"          # 你的蒸馏权重
sam = sam_model_registry["vit_b"](checkpoint=None).to(device)

# 加载蒸馏后的 prompt_encoder & mask_decoder
state = torch.load(ckpt_path, map_location=device)

sam.prompt_encoder.load_state_dict(
    {k[len("prompt_encoder."):]: v
     for k, v in state.items()
     if k.startswith("prompt_encoder.")}
)

sam.mask_decoder.load_state_dict(
    {k[len("mask_decoder."):]: v
     for k, v in state.items()
     if k.startswith("mask_decoder.")}
)

predictor = SamPredictor(sam)

# ---------- 2. 读图 ----------
image_path = r"D:\TinySAM\segment-anything\images\complex_scene.jpg"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ---------- 3. 预测 ----------
predictor.set_image(image_rgb)

# 点提示
points = np.array([[200, 300], [400, 500]])
labels = np.array([1, 1])
masks_point, _, _ = predictor.predict(
    point_coords=points,
    point_labels=labels,
    multimask_output=False
)

# 框提示
box = np.array([[150, 250, 350, 450]])
masks_box, _, _ = predictor.predict(
    box=box,
    multimask_output=False
)

# ---------- 4. 可视化 ----------
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.scatter(points[:, 0], points[:, 1], color='red', s=100)
plt.title("Original + Points"); plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(image_rgb)
plt.imshow(masks_point[0], alpha=0.5, cmap='jet')
plt.title("Point Prompt"); plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(image_rgb)
plt.imshow(masks_box[0], alpha=0.5, cmap='jet')
plt.title("Box Prompt"); plt.axis("off")

plt.tight_layout()
plt.show()