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
import csv

# ==========================================
# 1. 核心路径与参数配置
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

results_csv_path = f"{output_dir}/uncertainty_results.csv"
thresholds_json_path = f"{output_dir}/simple_thresholds.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 45
batch_size = 32
N_passes = 20

# ==========================================
# 2. 读取特征中心
# ==========================================
print("正在加载特征参考库并计算类别中心...")
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
print(f"✅ 成功计算 {num_classes} 个类别的特征中心点！")

# ==========================================
# 3. 还原模型结构与加载权重
#    使用 DensePatchClassifier，匹配你现在的权重
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
        patch_logits = self.patch_head(x_patches)   # [B, N, K]
        img_logits = patch_logits.mean(dim=1)       # [B, K]
        return patch_logits, img_logits, x_patches

print("正在载入 DINOv3 与 Dense Patch Head...")
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

    print("✅ 模型权重全部加载完毕！")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    sys.exit(1)

model.to(device)
classifier_head.to(device)
model.eval()

# ==========================================
# 4. 构建验证集
# ==========================================
print("正在构建验证集数据管道...")
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

val_subset = Subset(ds, indices[split:])
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=8)
print(f"✅ 验证集准备完毕！共计 {len(val_subset)} 张测试图像。")

# ==========================================
# 5. 核心推理：提取图像级三大指标
# ==========================================
print(f"🚀 开始执行量化推理 (MC Dropout 次数: {N_passes})...")

all_results = []
total_correct = 0

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(val_loader):
        images, labels = images.to(device), labels.to(device)

        # [A] Backbone 提取 patch tokens
        patch_tokens = model.forward_features(images)["x_norm_patchtokens"]  # [B, N, C]

        # [B] 纯净特征：用于 feature distance
        classifier_head.eval()
        _, img_logits_clean, patch_feat = classifier_head(patch_tokens)
        pure_feat = patch_feat.mean(dim=1)   # [B, C]

        # [C] MC Dropout：用于不确定性
        classifier_head.train()
        mcd_probs = []
        for _ in range(N_passes):
            _, img_logits_mc, _ = classifier_head(patch_tokens)
            probs = torch.softmax(img_logits_mc, dim=1)
            mcd_probs.append(probs)

        mcd_probs = torch.stack(mcd_probs, dim=0)   # [T, B, K]
        mean_probs = mcd_probs.mean(dim=0)          # [B, K]

        softmax_confidences, predictions = torch.max(mean_probs, dim=1)
        total_correct += (predictions == labels).sum().item()

        for i in range(images.size(0)):
            gt_label = labels[i].item()
            pred_label = predictions[i].item()
            is_correct = int(gt_label == pred_label)

            # MC Dropout variance
            mcd_var = torch.var(mcd_probs[:, i, pred_label]).item()

            # Feature distance: 余弦距离
            cos_sim = F.cosine_similarity(
                pure_feat[i].unsqueeze(0),
                centroids_tensor[pred_label].unsqueeze(0)
            )
            feature_distance = 1.0 - cos_sim.item()

            all_results.append({
                "Sample_ID": batch_idx * batch_size + i,
                "True_Class": ordered_classes[gt_label],
                "Pred_Class": ordered_classes[pred_label],
                "Is_Correct": is_correct,
                "Softmax_Conf": round(softmax_confidences[i].item(), 6),
                "MCD_Variance": round(mcd_var, 8),
                "Feature_Dist": round(feature_distance, 6)
            })

        if batch_idx % 10 == 0:
            print(f"正在处理 Batch [{batch_idx}/{len(val_loader)}]...")

# ==========================================
# 6. 保存 CSV
# ==========================================
final_acc = 100.0 * total_correct / len(val_subset)
print(f"\n🎉 所有验证集推理完成！最终分类准确率: {final_acc:.2f}%")

fieldnames = [
    "Sample_ID", "True_Class", "Pred_Class", "Is_Correct",
    "Softmax_Conf", "MCD_Variance", "Feature_Dist"
]
with open(results_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_results)

print(f"📊 详细量化指标已保存至: {results_csv_path}")

# ==========================================
# 7. 自动估计简单阈值
# ==========================================
correct_rows = [r for r in all_results if r["Is_Correct"] == 1]
if len(correct_rows) < 10:
    print("⚠️ 正确样本太少，自动阈值将退化为全部样本统计。")
    correct_rows = all_results

conf_arr = np.array([r["Softmax_Conf"] for r in correct_rows], dtype=np.float32)
mcd_arr = np.array([r["MCD_Variance"] for r in correct_rows], dtype=np.float32)
dist_arr = np.array([r["Feature_Dist"] for r in correct_rows], dtype=np.float32)

thresholds = {
    "threshold_softmax_conf": float(np.quantile(conf_arr, 0.10)),
    "threshold_mcd_var": float(np.quantile(mcd_arr, 0.90)),
    "threshold_feature_dist": float(np.quantile(dist_arr, 0.90)),
    "notes": "Simple thresholds estimated from correctly predicted validation samples."
}

with open(thresholds_json_path, "w", encoding="utf-8") as f:
    json.dump(thresholds, f, indent=2, ensure_ascii=False)

print(f"✅ 简单阈值已保存至: {thresholds_json_path}")
print(json.dumps(thresholds, indent=2, ensure_ascii=False))