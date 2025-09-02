import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import sam_model_registry
from segment_anything import SamPredictor

# ---------- 加载原始vit_b模型 ----------
orig_model_type = "vit_b"
orig_checkpoint = "sam_vit_b_01ec64.pth"  # 原始小模型权重
orig_device = "cpu"
orig_sam = sam_model_registry[orig_model_type](checkpoint=orig_checkpoint)
orig_sam.to(device=orig_device)
orig_predictor = SamPredictor(orig_sam)

# ---------- 加载蒸馏后的vit_b模型 ----------
distilled_checkpoint = "sam_vit_b_distilled.pth"  # 刚保存的蒸馏模型
distilled_sam = sam_model_registry[orig_model_type](checkpoint=None)  # 先初始化空模型
distilled_sam.load_state_dict(torch.load(distilled_checkpoint, map_location=orig_device))
distilled_sam.to(device=orig_device)
distilled_sam.eval()
distilled_predictor = SamPredictor(distilled_sam)

# ---------- 加载测试图片 ----------
image_path = "D:/TinySAM/segment-anything/notebooks/images/cat.jpg"  # 之前用的小图，也可以换其他图
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ---------- 定义提示点（选图中一个明显物体的中心） ----------
h, w = image.shape[:2]
input_point = np.array([[w//2, h//2]])  # 图片中心点
input_label = np.array([1])  # 1表示“前景点”

# ---------- 原始模型预测 ----------
orig_predictor.set_image(image)
orig_masks, _, _ = orig_predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False,
)

# ---------- 蒸馏模型预测 ----------
distilled_predictor.set_image(image)
distilled_masks, _, _ = distilled_predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False,
)

# ---------- 可视化对比 ----------
plt.figure(figsize=(12, 5))

# 子图1：原始图片 + 提示点
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.scatter(input_point[:, 0], input_point[:, 1], color='red', s=100)
plt.title("原始图片 + 提示点")
plt.axis("off")

# 子图2：原始vit_b模型分割结果
plt.subplot(1, 3, 2)
plt.imshow(image)
plt.imshow(orig_masks[0], alpha=0.5, cmap='jet')
plt.title("原始 vit_b 模型")
plt.axis("off")

# 子图3：蒸馏后模型分割结果
plt.subplot(1, 3, 3)
plt.imshow(image)
plt.imshow(distilled_masks[0], alpha=0.5, cmap='jet')
plt.title("蒸馏后模型")
plt.axis("off")

plt.tight_layout()
plt.show()