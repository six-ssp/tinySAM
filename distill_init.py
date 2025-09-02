import torch
import os
from segment_anything import sam_model_registry
from PIL import Image  # 用于读取图片
import numpy as np

# ---------- 教师模型（大模型，vit_h） ----------
teacher_model_type = "vit_h"
teacher_checkpoint = "sam_vit_h_4b8939.pth"  # 确保该权重文件在当前目录
teacher_device = "cpu"
teacher_sam = sam_model_registry[teacher_model_type](checkpoint=teacher_checkpoint)
teacher_sam.to(device=teacher_device)
teacher_sam.eval()  # 教师模型仅用于推理（输出"标准答案"）

# ---------- 学生模型（小模型，vit_b） ----------
student_model_type = "vit_b"
student_checkpoint = "sam_vit_b_01ec64.pth"  # 确保该权重文件在当前目录
student_device = "cpu"
student_sam = sam_model_registry[student_model_type](checkpoint=student_checkpoint)
student_sam.to(device=student_device)
student_sam.train()  # 学生模型需要训练（学习模仿教师）

# ---------- 读取你的数据集图片（适配任意命名方式） ----------
# 你的图片文件夹路径（根据实际路径修改）
data_dir = "D:/TinySAM/coco_small/images"

# 读取文件夹中所有.jpg图片（不管命名格式）
image_paths = []
for filename in os.listdir(data_dir):
    # 只筛选.jpg格式的文件（不区分大小写）
    if filename.lower().endswith(".jpg"):
        full_path = os.path.join(data_dir, filename)
        image_paths.append(full_path)

# 验证数据集读取结果
print("✅ 数据集读取状态：")
if len(image_paths) == 0:
    print(f"⚠️ 未找到图片！请检查路径：{data_dir}")
else:
    print(f"✅ 成功找到 {len(image_paths)} 张图片")
    # 打印前3张图片的路径（展示你的命名方式）
    print("示例图片路径：")
    for path in image_paths[:3]:
        print(f"- {path}")

    # 测试读取第一张图片，检查是否能正常打开
    try:
        test_image = Image.open(image_paths[0]).convert("RGB")
        test_image_np = np.array(test_image)
        print(f"✅ 测试图片正常，尺寸：{test_image_np.shape}（高度×宽度×通道数）")
    except Exception as e:
        print(f"⚠️ 图片读取失败：{str(e)}")

# 打印模型参数信息
print("\n📊 模型参数信息：")
print(f"教师模型（vit_h）参数总量：{sum(p.numel() for p in teacher_sam.parameters())/1e9:.2f}B")
print(f"学生模型（vit_b）参数总量：{sum(p.numel() for p in student_sam.parameters())/1e9:.2f}B")
