import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2
from torchvision import datasets
import numpy as np
import os
import json
import sys
import cv2
import matplotlib.pyplot as plt

# ==========================================
# 1. 路径与参数配置 (保持不变)
# ==========================================
project_root = "/root/dinov3"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

data_root = "/root/autodl-tmp/datasets/NWPU_data"
label_map_path = "/root/autodl-tmp/datasets/label_map.json"
output_dir = "/root/autodl-tmp/outputs"
dinov3_weights = "/root/autodl-tmp/weights/dinov3_vitl16_pretrain_sat493m.pth"
head_weights_path = f"{output_dir}/nwpu_attn_head_best.pth"
features_path = f"{output_dir}/test_attn_features.npy"
labels_path = f"{output_dir}/test_attn_labels.npy"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 45
N_passes = 20  # MCD 推理次数

# 判定低适配区的阈值 (保持不变)
THRESHOLD_FEATURE_DIST = 0.35
THRESHOLD_MCD_VAR = 0.005

# ==========================================
# 2. 准备特征参考库 (保持不变)
# ==========================================
print("正在加载特征参考库...")
train_features = np.load(features_path)
train_labels = np.load(labels_path)
class_centers = [np.mean(train_features[train_labels == c], axis=0) for c in range(num_classes)]
centroids_tensor = torch.tensor(np.array(class_centers), dtype=torch.float32).to(device)

# ==========================================
# 3. 还原模型结构与加载权重 (保持不变)
# ==========================================
class AttentionClassifier(nn.Module):
    def __init__(self, embed_dim, num_classes, num_heads=8, dropout=0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x_patches):
        b = x_patches.shape[0]
        q = self.query.expand(b, -1, -1)
        # attn_weights shape: [batch_size, 1, num_patches]
        attn_out, attn_weights = self.attn(q, x_patches, x_patches)
        feat = self.dropout(attn_out.squeeze(1))
        logits = self.fc(feat)
        return logits, feat, attn_weights

try:
    import hubconf
    model = hubconf.dinov3_vitl16(pretrained=False)
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(dinov3_weights, map_location="cpu").get('model', torch.load(dinov3_weights, map_location="cpu")).items()}, strict=False)
    classifier_head = AttentionClassifier(model.embed_dim, num_classes)
    classifier_head.load_state_dict(torch.load(head_weights_path, map_location="cpu"))
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    sys.exit(1)

model.to(device).eval()
classifier_head.to(device)

# ==========================================
# 4. 数据准备 (仅取前 8 张图进行贴图测试)
# ==========================================
print("正在构建数据管道并提取验证集前 8 张图...")
standard_transform = v2.Compose([
    v2.ToImage(), v2.Resize((256, 256), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.430, 0.411, 0.296), std=(0.213, 0.156, 0.143))
])
with open(label_map_path, "r", encoding="utf-8") as f:
    idx_to_class_map = json.load(f)
ordered_classes = [idx_to_class_map[str(i)] for i in range(len(idx_to_class_map))]

ds = datasets.ImageFolder(root=data_root, transform=standard_transform)
c2i = {cls_name: i for i, cls_name in enumerate(ordered_classes)}
new_samples = [(path, c2i[os.path.basename(os.path.dirname(path))]) for path, _ in ds.samples]
ds.classes, ds.class_to_idx, ds.samples, ds.targets = ordered_classes, c2i, new_samples, [s[1] for s in new_samples]

indices = torch.randperm(len(ds), generator=torch.Generator().manual_seed(42)).tolist()
split = int(0.8 * len(ds))
val_subset = Subset(ds, indices[split:split+8]) # 取验证集的前 8 张
test_loader = DataLoader(val_subset, batch_size=8, shuffle=False)

# 用于还原被 Normalize 过的图像
def unnormalize(tensor):
    mean = torch.tensor([0.430, 0.411, 0.296]).view(3, 1, 1).to(device)
    std = torch.tensor([0.213, 0.156, 0.143]).view(3, 1, 1).to(device)
    return torch.clamp(tensor * std + mean, 0, 1)

# ==========================================
# 5. 推理与热力图计算 (优化 MCD 逻辑)
# ==========================================
print("🚀 执行量化推理与注意力图提取...")
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

with torch.no_grad():
    # 提取 Patch Tokens 用于 Attention 聚合
    patch_tokens = model.forward_features(images)['x_norm_patchtokens']
    
    # 提取干净的特征与原始注意力图 (关闭 Dropout)
    classifier_head.eval()
    outputs, pure_feat, attn_weights = classifier_head(patch_tokens)
    
    # [核心：蒙特卡洛多次推理计算方差]
    # ⚠️ 极其关键：将 head 设为 train 模式，强行激活 Dropout
    classifier_head.train() 
    mcd_list = []
    for _ in range(N_passes):
        logits, _, _ = classifier_head(patch_tokens)
        probs = torch.softmax(logits, dim=1)
        mcd_list.append(probs)
        
    mcd_probs = torch.stack(mcd_list) # Shape: [20, batch_size, 45] (例如 20, 32, 45)
    
    # 计算指标
    # 计算 20 次预测的平均概率分布
    mean_probs = mcd_probs.mean(dim=0)
    
    # [指标 A] 预测类别 与 Softmax 置信度 (基于平均概率分布)
    softmax_confidences, predictions = torch.max(mean_probs, dim=1)

# ==========================================
# 6. 可视化贴图渲染 (核心修改：从点到边界的掩码生成)
# ==========================================
print("正在执行“从点到边界”的形态学贴图变换...")
fig, axes = plt.subplots(2, 4, figsize=(22, 11))
axes = axes.flatten()

for i in range(images.size(0)):
    # 1. 计算当前样本的量化指标
    pred_idx = predictions[i].item()
    gt_idx = labels[i].item()
    # 取出这 20 次推理中，模型对于其最终预测类别的概率方差
    mcd_var = torch.var(mcd_probs[:, i, pred_idx]).item()
    
    # 计算特征距离 [指标 C: 余弦距离]
    cos_sim = F.cosine_similarity(pure_feat[i].unsqueeze(0), centroids_tensor[pred_idx].unsqueeze(0))
    feature_dist = 1.0 - cos_sim.item()
    
    # 2. 判断区域适配性并设定标题颜色
    # 如果两个不确定性指标触碰红线，判定为低适配区
    is_low_adapt = (feature_dist > THRESHOLD_FEATURE_DIST) or (mcd_var > THRESHOLD_MCD_VAR)
    box_color = 'red' if is_low_adapt else 'green'
    status_text = "LOW ADAPT" if is_low_adapt else "HIGH ADAPT"
    
    # --- 3. 核心可视化逻辑修改：生成贴合边界的掩码 (只针对 HIGH ADAPT) ---
    # 提取原始分辨率注意力图并归一化到 uint8 用于图像处理
    num_side = int(np.sqrt(attn_weights.shape[2])) # 16
    raw_map = attn_weights[i, 0, :].view(1, 1, num_side, num_side)
    # 双线性插值放大到 256x256，对齐原图尺寸
    attn_map = F.interpolate(raw_map, size=(256, 256), mode='bilinear', align_corners=False)
    attn_map = attn_map.squeeze().cpu().numpy()
    # 归一化到 [0, 255]
    attn_map_uint8 = np.uint8(255 * (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8))

    # 生成基础热力图色彩并转换格式
    heatmap_raw = cv2.applyColorMap(attn_map_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_raw, cv2.COLOR_BGR2RGB)

    # 处理底图
    img_display = unnormalize(images[i]).permute(1, 2, 0).cpu().numpy()
    img_display = np.uint8(255 * img_display)
    
    overlay = img_display.copy()

    # ====== 新增条件可视化逻辑 ======
    if not is_low_adapt:
        # 🟢 【高适配区 (绿色)】：强制提取清晰连贯的物体掩码以“贴合边缘”
        
        # A. 应用大津法获取自动阈值掩码 (Otsu's Thresholding)
        # 这步会自动识别出模型最关注的物体（通常是建筑轮廓）的核心区域
        _, binary_mask = cv2.threshold(attn_map_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # B. 形态学处理来提取“边界区域”
        # 先使用形态学开运算去噪声平滑
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open)
        
        # 再膨胀或形态学闭运算填充空洞连接邻近块
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)) # 较大核连接
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close)

        # 形态学膨胀让掩码完整包裹建筑边界
        # binary_mask = cv2.dilate(binary_mask, kernel_close, iterations=1)

        # C. 掩码渲染：局部混合，只在掩码区域叠加颜色
        # 我们保留 JET 颜色的色彩分布，但只在二值化掩码区域内显示
        mask_bool = binary_mask > 0
        heatmap_and = cv2.bitwise_and(heatmap_rgb, heatmap_rgb, mask=binary_mask)
        
        # 掩码区域透明度 0.6
        overlay[mask_bool] = cv2.addWeighted(img_display[mask_bool], 0.4, 
                                                heatmap_and[mask_bool], 0.6, 0).squeeze()
        
        # 将 matplotlib 绘图时强调其为硬掩码
        ax_prefix = "[Otsu Mask] "

    else:
        # 🔴 【低适配区 (红色)】：强调分散、混乱。使用模糊图
        
        # Gamma校正拉伸暗部，大核高斯模糊平滑烟雾感
        attn_map_smooth = np.power(attn_map, 0.4) 
        attn_map_smooth = cv2.GaussianBlur(attn_map_smooth, (31, 31), 0)
        attn_map_smooth = np.uint8(255 * (attn_map_smooth - attn_map_smooth.min()) / (attn_map_smooth.max() - attn_map_smooth.min() + 1e-8))
        
        heatmap_smooth_rgb = cv2.applyColorMap(attn_map_smooth, cv2.COLORMAP_JET)
        heatmap_smooth_rgb = cv2.cvtColor(heatmap_smooth_rgb, cv2.COLOR_BGR2RGB)
        
        # 全局 Alpha Blending (透明度 0.5)
        overlay = cv2.addWeighted(img_display, 0.5, heatmap_smooth_rgb, 0.5, 0)
        
        # 将 matplotlib 绘图时强调其为模糊图
        ax_prefix = "[Gamma Gauss Blur] "

    # 绘制图形
    ax = axes[i]
    ax.imshow(overlay)
    
    # 绘制表示高/低适配区的边框
    for spine in ax.spines.values():
        spine.set_edgecolor(box_color)
        spine.set_linewidth(6)
        
    ax.set_xticks([]); ax.set_yticks([])
    
    # 添加指标文本
    ax.set_title(f"GT:{ordered_classes[gt_idx]} PR:{ordered_classes[pred_idx]}\n"
                 f"{ax_prefix}[{status_text}]\n"
                 f"Dist:{feature_dist:.3f} Var:{mcd_var:.5f}", 
                 color=box_color, fontsize=10, fontweight='bold')

plt.tight_layout()
save_path = f"{output_dir}/adaptation_overlay_otsu_mask.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✅ 最終可视化贴图已生成：{save_path}")