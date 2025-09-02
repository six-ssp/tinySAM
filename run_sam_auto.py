import torch
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator

# 1. 加载 SAM 模型
model_type = "vit_h"
sam_checkpoint = "sam_vit_h_4b8939.pth"  # 权重文件要在当前目录
device = "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# 2. 创建自动掩码生成器
mask_generator = SamAutomaticMaskGenerator(sam)

# 3. 加载测试图片
image_path = "D:/TinySAM/segment-anything/notebooks/images/cat.jpg"  # 刚才放的图片路径
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV 读入是 BGR，转成 RGB 用于画图

# 4. 生成所有可能的掩码
masks = mask_generator.generate(image)

# 5. 可视化结果（把掩码叠在原图上）
plt.figure(figsize=(10, 10))
plt.imshow(image)
for mask in masks:
    plt.imshow(mask["segmentation"], alpha=0.3)  # alpha 是透明度，能看到原图和掩码
plt.axis("off")
plt.show()