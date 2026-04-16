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
thresholds_json_path = f"{output_dir}/simple_thresholds.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 45
N_passes = 20

# ==========================================
# 2. 阈值设置
# ==========================================
USE_AUTO_THRESHOLDS = True

MANUAL_THRESHOLD_SOFTMAX_CONF = 0.55
MANUAL_THRESHOLD_MCD_VAR = 0.005
MANUAL_THRESHOLD_FEATURE_DIST = 0.35

if USE_AUTO_THRESHOLDS and os.path.exists(thresholds_json_path):
    with open(thresholds_json_path, "r", encoding="utf-8") as f:
        thresholds = json.load(f)

    THRESHOLD_SOFTMAX_CONF = thresholds["threshold_softmax_conf"]
    THRESHOLD_MCD_VAR = thresholds["threshold_mcd_var"]
    THRESHOLD_FEATURE_DIST = thresholds["threshold_feature_dist"]

    print("✅ 已读取自动阈值：")
    print(json.dumps(thresholds, indent=2, ensure_ascii=False))
else:
    THRESHOLD_SOFTMAX_CONF = MANUAL_THRESHOLD_SOFTMAX_CONF
    THRESHOLD_MCD_VAR = MANUAL_THRESHOLD_MCD_VAR
    THRESHOLD_FEATURE_DIST = MANUAL_THRESHOLD_FEATURE_DIST

    print("⚠️ 未读取自动阈值，使用手工阈值：")
    print({
        "threshold_softmax_conf": THRESHOLD_SOFTMAX_CONF,
        "threshold_mcd_var": THRESHOLD_MCD_VAR,
        "threshold_feature_dist": THRESHOLD_FEATURE_DIST
    })

# ==========================================
# 3. 读取特征中心
# ==========================================
print("正在加载特征参考库...")
train_features = np.load(features_path)
train_labels = np.load(labels_path)

class_centers = []
for c in range(num_classes):
    c_feats = train_features[train_labels == c]
    if len(c_feats) == 0:
        c_center = np.zeros((train_features.shape[1],), dtype=np.float32)
    else:
        c_center = np.mean(c_feats, axis=0)
    class_centers.append(c_center)

centroids_tensor = torch.tensor(np.array(class_centers), dtype=torch.float32).to(device)

# ==========================================
# 4. 模型结构：DensePatchClassifier
# ==========================================
class DensePatchClassifier(nn.Module):
    def __init__(self, embed_dim, num_classes, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.patch_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x_patches):
        """
        x_patches: [B, N, C]
        return:
            patch_logits: [B, N, K]
            img_logits:   [B, K]
            patch_feat:   [B, N, C]
        """
        patch_logits = self.patch_head(x_patches)
        img_logits = patch_logits.mean(dim=1)
        return patch_logits, img_logits, x_patches

# ==========================================
# 5. 加载模型
# ==========================================
try:
    try:
        from dinov3.hub import dinov3_vitl16
    except ImportError:
        import hubconf
        dinov3_vitl16 = hubconf.dinov3_vitl16

    model = dinov3_vitl16(pretrained=False)

    checkpoint = torch.load(dinov3_weights, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    classifier_head = DensePatchClassifier(model.embed_dim, num_classes)
    classifier_head.load_state_dict(torch.load(head_weights_path, map_location="cpu"))

except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    sys.exit(1)

model.to(device).eval()
classifier_head.to(device)

# ==========================================
# 6. 数据准备
# ==========================================
standard_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((256, 256), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.430, 0.411, 0.296), std=(0.213, 0.156, 0.143))
])

with open(label_map_path, "r", encoding="utf-8") as f:
    idx_to_class_map = json.load(f)
ordered_classes = [idx_to_class_map[str(i)] for i in range(len(idx_to_class_map))]

ds = datasets.ImageFolder(root=data_root, transform=standard_transform)
c2i = {cls_name: i for i, cls_name in enumerate(ordered_classes)}
new_samples = [(path, c2i[os.path.basename(os.path.dirname(path))]) for path, _ in ds.samples]
ds.classes = ordered_classes
ds.class_to_idx = c2i
ds.samples = new_samples
ds.targets = [s[1] for s in new_samples]

indices = torch.randperm(len(ds), generator=torch.Generator().manual_seed(42)).tolist()
split = int(0.8 * len(ds))
val_subset = Subset(ds, indices[split:split + 8])
test_loader = DataLoader(val_subset, batch_size=8, shuffle=False)

def unnormalize(tensor):
    mean = torch.tensor([0.430, 0.411, 0.296]).view(3, 1, 1).to(device)
    std = torch.tensor([0.213, 0.156, 0.143]).view(3, 1, 1).to(device)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1)

# ==========================================
# 7. 推理与“类响应热力图”生成
#    用每个 patch 对最终预测类的概率，替代旧 attention map
# ==========================================
print("🚀 开始生成可视化结果...")
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

with torch.no_grad():
    patch_tokens = model.forward_features(images)["x_norm_patchtokens"]

    classifier_head.eval()
    patch_logits_clean, img_logits_clean, patch_feat = classifier_head(patch_tokens)
    pure_feat = patch_feat.mean(dim=1)                 # [B, C]

    patch_probs_clean = torch.softmax(patch_logits_clean, dim=-1)   # [B, N, K]

    classifier_head.train()
    mcd_probs = []
    for _ in range(N_passes):
        _, img_logits_mc, _ = classifier_head(patch_tokens)
        mcd_probs.append(torch.softmax(img_logits_mc, dim=1))

    mcd_probs = torch.stack(mcd_probs, dim=0)    # [T, B, K]
    mean_probs = mcd_probs.mean(dim=0)           # [B, K]
    softmax_conf, preds = torch.max(mean_probs, dim=1)

# ==========================================
# 8. 画图
# ==========================================
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i in range(images.size(0)):
    pred_class = preds[i].item()
    gt_class = labels[i].item()

    # 图像级指标
    mcd_var = torch.var(mcd_probs[:, i, pred_class]).item()

    cos_sim = F.cosine_similarity(
        pure_feat[i].unsqueeze(0),
        centroids_tensor[pred_class].unsqueeze(0)
    )
    feature_dist = 1.0 - cos_sim.item()

    img_conf = softmax_conf[i].item()

    # 简单阈值判定
    if (img_conf < THRESHOLD_SOFTMAX_CONF) or (
        (mcd_var > THRESHOLD_MCD_VAR) and (feature_dist > THRESHOLD_FEATURE_DIST)
    ):
        is_low_adapt = True
        box_color = "red"
        status_text = "Low Adapt"
        title_color = "red"
    else:
        is_low_adapt = False
        box_color = "green"
        status_text = "High Adapt"
        title_color = "black"

    # 用 patch 对最终预测类的概率作为可视化热力图
    patch_map = patch_probs_clean[i, :, pred_class]   # [N]
    side = int(np.sqrt(patch_map.numel()))
    patch_map = patch_map.view(1, 1, side, side)

    patch_map = F.interpolate(patch_map, size=(256, 256), mode="bilinear", align_corners=False)
    patch_map = patch_map.squeeze().cpu().numpy()

    patch_min = patch_map.min()
    patch_max = patch_map.max()
    if patch_max - patch_min < 1e-8:
        patch_map = np.zeros_like(patch_map)
    else:
        patch_map = (patch_map - patch_min) / (patch_max - patch_min)

    patch_map = cv2.GaussianBlur(patch_map, (9, 9), 0)

    heatmap = cv2.applyColorMap(np.uint8(255 * patch_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    img_display = unnormalize(images[i]).permute(1, 2, 0).cpu().numpy()
    img_display = np.uint8(255 * img_display)

    alpha = 0.55
    overlay = cv2.addWeighted(img_display, alpha, heatmap, 1 - alpha, 0)

    ax = axes[i]
    ax.imshow(overlay)

    for spine in ax.spines.values():
        spine.set_edgecolor(box_color)
        spine.set_linewidth(5)

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title(
        f"True: {ordered_classes[gt_class]}\n"
        f"Pred: {ordered_classes[pred_class]}\n"
        f"Status: {status_text}\n"
        f"ImgConf: {img_conf:.2f} | MCD: {mcd_var:.4f} | Dist: {feature_dist:.3f}",
        color=title_color,
        fontsize=10
    )

plt.tight_layout()
save_path = f"{output_dir}/adaptation_overlay_results.png"
plt.savefig(save_path, dpi=300)
print(f"🎉 可视化贴图已生成并保存至: {save_path}")