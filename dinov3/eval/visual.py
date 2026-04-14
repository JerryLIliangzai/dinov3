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
# 1. 路径与参数配置
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

# 判定低适配区的阈值 (根据你之前跑出的 CSV 数据大致设定)
THRESHOLD_FEATURE_DIST = 0.35
THRESHOLD_MCD_VAR = 0.005

# ==========================================
# 2. 准备特征参考库 (用于距离计算)
# ==========================================
print("正在加载特征参考库...")
train_features = np.load(features_path)
train_labels = np.load(labels_path)
class_centers = [np.mean(train_features[train_labels == c], axis=0) for c in range(num_classes)]
centroids_tensor = torch.tensor(np.array(class_centers), dtype=torch.float32).to(device)

# ==========================================
# 3. 还原模型结构与加载权重
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
# 4. 数据准备 (仅取前 8 张图用于可视化)
# ==========================================
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

# 用于还原被 Normalize 过的图像，以便人类观看
def unnormalize(tensor):
    mean = torch.tensor([0.430, 0.411, 0.296]).view(3, 1, 1).to(device)
    std = torch.tensor([0.213, 0.156, 0.143]).view(3, 1, 1).to(device)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1)

# ==========================================
# 5. 推理与热力图生成
# ==========================================
print("🚀 开始生成可视化结果...")
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

with torch.no_grad():
    patch_tokens = model.forward_features(images)['x_norm_patchtokens']
    
    # 提取纯特征与 Attention Map
    classifier_head.eval()
    _, pure_feat, attn_weights = classifier_head(patch_tokens)
    
    # MC Dropout 提取方差与 Softmax
    classifier_head.train()
    mcd_probs = []
    for _ in range(N_passes):
        logits, _, _ = classifier_head(patch_tokens)
        mcd_probs.append(torch.softmax(logits, dim=1))
    mcd_probs = torch.stack(mcd_probs)
    mean_probs = mcd_probs.mean(dim=0)
    softmax_conf, preds = torch.max(mean_probs, dim=1)

# ==========================================
# 6. 开始画图叠加
# ==========================================
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i in range(images.size(0)):
    # 1. 计算指标
    pred_class = preds[i].item()
    gt_class = labels[i].item()
    mcd_var = torch.var(mcd_probs[:, i, pred_class]).item()
    
    cos_sim = F.cosine_similarity(pure_feat[i].unsqueeze(0), centroids_tensor[pred_class].unsqueeze(0))
    feature_dist = 1.0 - cos_sim.item()
    
    # 2. 判断是否为低适配区
    is_low_adapt = (feature_dist > THRESHOLD_FEATURE_DIST) or (mcd_var > THRESHOLD_MCD_VAR)
    box_color = 'red' if is_low_adapt else 'green'
    status_text = "Low Adapt" if is_low_adapt else "High Adapt"
    
    # 3. 处理 Attention Map
    # DINOv3 ViT-L/16 对应 256x256 图片的 patch 数量是 16x16=256
    num_patches_side = int(np.sqrt(attn_weights.shape[2])) # 16
    attn_map = attn_weights[i, 0, :].view(1, 1, num_patches_side, num_patches_side) # [1, 1, 16, 16]
    # 上采样到 256x256
    attn_map = F.interpolate(attn_map, size=(256, 256), mode='bilinear', align_corners=False)
    attn_map = attn_map.squeeze().cpu().numpy()
    
    # 归一化热力图
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
    heatmap = cv2.applyColorMap(np.uint8(255 * attn_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # 4. 处理原图并融合
    img_display = unnormalize(images[i]).permute(1, 2, 0).cpu().numpy()
    img_display = np.uint8(255 * img_display)
    
    # 融合原图与热力图 (Alpha Blending)
    alpha = 0.5
    overlay = cv2.addWeighted(img_display, alpha, heatmap, 1 - alpha, 0)
    
    # 5. 在 matplotlib 中绘制
    ax = axes[i]
    ax.imshow(overlay)
    
    # 绘制表示高/低适配区的边框
    for spine in ax.spines.values():
        spine.set_edgecolor(box_color)
        spine.set_linewidth(5)
        
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 添加指标文本
    title_color = 'red' if is_low_adapt else 'black'
    ax.set_title(f"True: {ordered_classes[gt_class]}\n"
                 f"Pred: {ordered_classes[pred_class]}\n"
                 f"Status: {status_text}\n"
                 f"S-Conf: {softmax_conf[i].item():.2f} | Dist: {feature_dist:.3f}", 
                 color=title_color, fontsize=10)

plt.tight_layout()
plt.savefig(f"{output_dir}/adaptation_overlay_results.png", dpi=300)
print(f"🎉 可视化贴图已生成并保存至: {output_dir}/adaptation_overlay_results.png")