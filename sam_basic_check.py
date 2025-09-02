import torch
# 直接从sam.py文件导入（sam_model_registry的实际定义位置）
from segment_anything import sam_model_registry

# 配置模型信息
model_type = "vit_h"
sam_checkpoint = "sam_vit_h_4b8939.pth"  # 确保权重文件在当前目录
device = "cpu"

try:
    # 尝试加载模型
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    print("模型加载成功！SAM基础组件正常")
    
    # 打印模型基本信息
    print(f"\n模型类型: {model_type}")
    print(f"设备: {device}")
    print(f"参数总量: {sum(p.numel() for p in sam.parameters())/1e9:.2f}B")
except Exception as e:
    print(f"加载失败: {str(e)}")
    # 检查权重文件是否存在
    import os
    if not os.path.exists(sam_checkpoint):
        print(f"权重文件不存在: {sam_checkpoint}")
    else:
        print("权重文件存在，但加载过程出错")