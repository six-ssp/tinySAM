import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry
from segment_anything import SamPredictor

# ---------- 数据集类 ----------
class COCOSmallDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(".jpg")]
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        return np.array(image), img_path

# ---------- 师生模型初始化 ----------
# 教师模型（固定参数，不训练）
teacher_model_type = "vit_h"
teacher_checkpoint = "sam_vit_h_4b8939.pth"
teacher_device = "cpu"
teacher_sam = sam_model_registry[teacher_model_type](checkpoint=teacher_checkpoint)
teacher_sam.to(device=teacher_device)
teacher_sam.eval()
teacher_predictor = SamPredictor(teacher_sam)

# 学生模型（需要训练，确保参数可求导）
student_model_type = "vit_b"
student_checkpoint = "sam_vit_b_01ec64.pth"
student_device = "cpu"
student_sam = sam_model_registry[student_model_type](checkpoint=student_checkpoint)
student_sam.to(device=student_device)
student_sam.train()  # 确保处于训练模式
student_predictor = SamPredictor(student_sam)

# ---------- 数据加载器 ----------
data_dir = "D:/TinySAM/coco_small/images"
dataset = COCOSmallDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# ---------- 优化器 ----------
optimizer = torch.optim.Adam(student_sam.parameters(), lr=1e-5)

# ---------- 蒸馏训练循环（修复梯度问题） ----------
num_epochs = 1
for epoch in range(num_epochs):
    total_loss = 0.0
    for i, (image_tensor, img_path) in enumerate(dataloader):
        # 1. 处理输入图片
        image = image_tensor.numpy()[0]  # 转为NumPy数组
        h, w = image.shape[:2]
        input_point = np.array([[w//2, h//2]])  # 中心点提示
        input_label = np.array([1])
        
        # 2. 教师模型生成伪标签（不跟踪梯度）
        with torch.no_grad():
            teacher_predictor.set_image(image)
            teacher_masks, _, _ = teacher_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
            )
        teacher_mask = torch.tensor(teacher_masks[0], device=student_device, dtype=torch.float32)
        
        # 3. 学生模型预测（关键：保留梯度信息）
        # 重置学生模型的梯度
        optimizer.zero_grad()
        
        # 学生模型处理图片（确保在计算图中）
        student_predictor.set_image(image)
        # 获取学生模型的预测（不使用no_grad，保留梯度）
        student_masks, _, _ = student_predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        # 将预测转为张量并确保在同一设备
        student_mask = torch.tensor(student_masks[0], device=student_device, dtype=torch.float32, requires_grad=True)
        
        # 4. 计算损失（确保两个张量形状一致）
        loss = torch.nn.functional.binary_cross_entropy(student_mask, teacher_mask)
        
        # 5. 反向传播和参数更新（关键步骤）
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新参数
        
        total_loss += loss.item()
        
        # 打印进度
        if i % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    print(f"\nEpoch {epoch+1} 平均损失: {avg_loss:.4f}\n")

# 保存蒸馏后的模型
torch.save(student_sam.state_dict(), "sam_vit_b_distilled.pth")
print("✅ 蒸馏训练完成！模型已保存为 sam_vit_b_distilled.pth")
