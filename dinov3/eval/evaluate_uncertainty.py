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

# 训练阶段保存的三个文件
head_weights_path = f"{output_dir}/nwpu_attn_head_best.pth"
features_path = f"{output_dir}/test_attn_features.npy"
labels_path = f"{output_dir}/test_attn_labels.npy"
# 评估结果保存路径
results_csv_path = f"{output_dir}/uncertainty_results.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 45
batch_size = 32  # 调小 batch_size，因为要跑 20 次前向传播
N_passes = 20    # MC Dropout 推理次数

# ==========================================
# 2. [任务书指标 3] 特征空间密度估计准备：计算特征中心
# ==========================================
print("正在加载特征参考库并计算类别中心...")
train_features = np.load(features_path)
train_labels = np.load(labels_path)

class_centers = []
for c in range(num_classes):
    c_feats = train_features[train_labels == c]
    c_center = np.mean(c_feats, axis=0)
    class_centers.append(c_center)
    
# 转换为 Tensor 并放到 GPU 上加速后续计算
centroids_tensor = torch.tensor(np.array(class_centers), dtype=torch.float32).to(device)
print(f"✅ 成功计算 45 个类别的特征中心点！")

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
        attn_out, attn_weights = self.attn(q, x_patches, x_patches)
        feat = attn_out.squeeze(1)
        feat = self.dropout(feat)
        logits = self.fc(feat)
        return logits, feat, attn_weights

print("正在载入 DINOv3 与 Attention Head...")
try:
    try:
        from dinov3.hub import dinov3_vitl16
    except ImportError:
        import hubconf
        dinov3_vitl16 = hubconf.dinov3_vitl16
    
    model = dinov3_vitl16(pretrained=False)
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(dinov3_weights, map_location="cpu").get('model', torch.load(dinov3_weights, map_location="cpu")).items()}, strict=False)
    
    classifier_head = AttentionClassifier(model.embed_dim, num_classes)
    classifier_head.load_state_dict(torch.load(head_weights_path, map_location="cpu"))
    print("✅ 模型权重全部加载完毕！")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    sys.exit(1)

model.to(device)
classifier_head.to(device)
model.eval() # Backbone 永远冻结

# ==========================================
# 4. 严谨的数据准备：完美还原训练时的验证集
# ==========================================
print("正在构建验证集数据管道(严格物理隔离)...")
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

# ⚠️ 必须与训练时使用相同的随机种子 (42)，才能切出那 20% 完全没见过的数据
indices = torch.randperm(len(ds), generator=torch.Generator().manual_seed(42)).tolist()
split = int(0.8 * len(ds))

# 提取后 20% 的验证集
val_subset = Subset(ds, indices[split:])
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=8)
print(f"✅ 验证集准备完毕！共计 {len(val_subset)} 张测试图像。")

# ==========================================
# 5. 核心推理：提取三大指标
# ==========================================
print(f"🚀 开始执行量化推理 (MC Dropout 次数: {N_passes})...")

all_results = []
total_correct = 0

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(val_loader):
        images, labels = images.to(device), labels.to(device)
        
        # [步骤 A] DINOv3 提取基础特征 (仅需 1 次)
        patch_tokens = model.forward_features(images)['x_norm_patchtokens']
        
        # [步骤 B] 提取纯净特征用于密度估计 (关闭 Dropout)
        classifier_head.eval()
        _, pure_feat, _ = classifier_head(patch_tokens)
        
        # [步骤 C] MC Dropout 多次推理 (强行开启 Dropout)
        classifier_head.train()
        mcd_probs = []
        for _ in range(N_passes):
            logits, _, _ = classifier_head(patch_tokens)
            probs = torch.softmax(logits, dim=1)
            mcd_probs.append(probs)
            
        mcd_probs = torch.stack(mcd_probs) # Shape: [20, batch_size, 45]
        mean_probs = mcd_probs.mean(dim=0) # Shape: [batch_size, 45]
        
        # ---------------- 提取任务书要求的三个指标 ----------------
        
        # 【指标 1：软最大值置信度 Softmax Confidence】
        softmax_confidences, predictions = torch.max(mean_probs, dim=1)
        
        # 统计批次准确率
        total_correct += (predictions == labels).sum().item()
        
        for i in range(images.size(0)):
            gt_label = labels[i].item()
            pred_label = predictions[i].item()
            is_correct = (gt_label == pred_label)
            
            # 【指标 2：MC Dropout 方差】
            # 取出这 20 次推理中，模型对于其最终预测类别的概率波动（方差）
            # 方差大 -> 极其不确定 -> 低适配区
            mcd_var = torch.var(mcd_probs[:, i, pred_label]).item()
            
            # 【指标 3：特征空间密度估计 (马氏距离的替代：余弦距离)】
            # 计算纯净特征 pure_feat[i] 到其被预测类别的中心点 centroids_tensor[pred_label] 的余弦距离
            # 距离大 -> 领域偏移/异物 -> 低适配区
            cos_sim = F.cosine_similarity(pure_feat[i].unsqueeze(0), centroids_tensor[pred_label].unsqueeze(0))
            feature_distance = 1.0 - cos_sim.item() # 距离 = 1 - 相似度
            
            # 将该样本的信息保存
            all_results.append({
                "Sample_ID": batch_idx * batch_size + i,
                "True_Class": ordered_classes[gt_label],
                "Pred_Class": ordered_classes[pred_label],
                "Is_Correct": int(is_correct),
                "Softmax_Conf": round(softmax_confidences[i].item(), 4),
                "MCD_Variance": round(mcd_var, 6),
                "Feature_Dist": round(feature_distance, 4)
            })

        # 打印进度
        if batch_idx % 10 == 0:
            print(f"正在处理 Batch [{batch_idx}/{len(val_loader)}]...")

# ==========================================
# 6. 保存与汇总
# ==========================================
final_acc = 100.0 * total_correct / len(val_subset)
print(f"\n🎉 所有验证集推理完成！最终分类准确率: {final_acc:.2f}%")

# 保存结果到 CSV，供后续画图和计算阈值使用
with open(results_csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["Sample_ID", "True_Class", "Pred_Class", "Is_Correct", "Softmax_Conf", "MCD_Variance", "Feature_Dist"])
    writer.writeheader()
    writer.writerows(all_results)

print(f"📊 详细量化指标已保存至: {results_csv_path}")
print("下一步：你可以下载此 CSV 文件，分析错误分类样本的 MCD_Variance 和 Feature_Dist 是否显著高于正确样本，从而确定'高低适配区'的区分阈值。")