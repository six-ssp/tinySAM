import torch
from segment_anything import sam_model_registry
from mobilevit import MobileViT

# ---------- 设备 ----------
device = "cpu"

# ---------- 1. 初始化一个空的 SAM（ViT-B 骨架） ----------
# 先让它加载原始权重，后面我们会覆盖 image_encoder 的权重
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth").to(device)

# ---------- 2. 构造轻量 image encoder ----------
mobile_vit = MobileViT(
    image_size=(256, 256),
    dims=[144, 192, 240],
    channels=[16, 32, 48, 64, 80, 96, 112, 128, 144, 160],
    num_classes=1000
)

# ---------- 3. 把 MobileViT 挂到 sam 上 ----------
# 注意：SAM 的 image_encoder 结构要与 MobileViT 完全一致
# 这里假设 MobileViT 输出通道数与 sam.image_encoder 的 patch_embed 对齐
# 如果通道不一致，需要再包一层适配层
sam.image_encoder = mobile_vit

# ---------- 4. 加载蒸馏后的 prompt_encoder & mask_decoder 参数 ----------
distilled_state = torch.load("sam_vit_b_distilled.pth", map_location=device)
# 只加载这两个子模块
sam.prompt_encoder.load_state_dict(
    {k[len("prompt_encoder."):]: v
     for k, v in distilled_state.items()
     if k.startswith("prompt_encoder.")}
)
sam.mask_decoder.load_state_dict(
    {k[len("mask_decoder."):]: v
     for k, v in distilled_state.items()
     if k.startswith("mask_decoder.")}
)

# ---------- 5. 保存 TinySAM ----------
torch.save(sam.state_dict(), "tiny_sam.pth")
print("✅ TinySAM 已保存为 tiny_sam.pth")