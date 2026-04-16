import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2
from torchvision import datasets
import numpy as np
import os
import json
import sys

# --- 1. 核心路径修复 ---
project_root = "/root/dinov3"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 2. 参数配置 (保留原地址) ---
data_root = "/root/autodl-tmp/datasets/NWPU_data"
label_map_path = "/root/autodl-tmp/datasets/label_map.json"
output_dir = "/root/autodl-tmp/outputs"
weights_path = "/root/autodl-tmp/weights/dinov3_vitl16_pretrain_sat493m.pth"

os.makedirs(output_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 45
lr = 1e-3
epochs = 10
batch_size = 64

# --- 3. DINOv3 官方 Transform 规范 ---
standard_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((256, 256), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.430, 0.411, 0.296), std=(0.213, 0.156, 0.143))
])

# --- 4. 载入固定映射与数据集对齐 ---
with open(label_map_path, "r", encoding="utf-8") as f:
    idx_to_class_map = json.load(f)
ordered_classes = [idx_to_class_map[str(i)] for i in range(len(idx_to_class_map))]

def get_fixed_dataset(root, ordered_classes, transform):
    ds = datasets.ImageFolder(root=root, transform=transform)
    c2i = {cls_name: i for i, cls_name in enumerate(ordered_classes)}
    new_samples = []
    for path, _ in ds.samples:
        folder_name = os.path.basename(os.path.dirname(path))
        new_samples.append((path, c2i[folder_name]))
    ds.classes = ordered_classes
    ds.class_to_idx = c2i
    ds.samples = new_samples
    ds.targets = [s[1] for s in new_samples]
    return ds

train_base = get_fixed_dataset(data_root, ordered_classes, standard_transform)
val_base = get_fixed_dataset(data_root, ordered_classes, standard_transform)

indices = torch.randperm(len(train_base), generator=torch.Generator().manual_seed(42)).tolist()
split = int(0.8 * len(train_base))

train_loader = DataLoader(
    Subset(train_base, indices[:split]),
    batch_size=batch_size,
    shuffle=True,
    num_workers=8
)
val_loader = DataLoader(
    Subset(val_base, indices[split:]),
    batch_size=batch_size,
    shuffle=False,
    num_workers=8
)

# --- 5. 定义新的 Dense Patch Classifier Head ---
# 保留“利用 patch token 做分类”的整体思路，但不再使用单 query attention 聚合，
# 改为对每个 patch 单独分类，再平均池化回图像级 logits。
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

# --- 6. 模型加载与初始化 ---
print("正在载入 DINOv3 Backbone...")
try:
    try:
        from dinov3.hub import dinov3_vitl16
        print("✅ 成功通过 dinov3.hub 导入")
    except ImportError:
        import hubconf
        dinov3_vitl16 = hubconf.dinov3_vitl16
        print("✅ 成功通过 hubconf 导入")

    model = dinov3_vitl16(pretrained=False)

    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location="cpu")
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        print("✅ 权重加载成功！")
    else:
        print(f"❌ 错误：在 {weights_path} 未找到权重文件！")
        sys.exit(1)
except Exception as e:
    print(f"❌ 最终导入失败: {e}")
    sys.exit(1)

# 冻结 Backbone 参数
for param in model.parameters():
    param.requires_grad = False

# 初始化新的 Dense Patch Classifier
classifier_head = DensePatchClassifier(model.embed_dim, num_classes)
model.to(device)
classifier_head.to(device)

# --- 7. 训练与验证 ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier_head.parameters(), lr=lr)

best_val_acc = 0.0
best_features = None
best_labels = None

for epoch in range(epochs):
    model.eval()   # Backbone 始终保持 eval 模式
    classifier_head.train()

    t_corr, t_tot = 0, 0
    t_loss_sum = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            # 获取全部 Patch Tokens 用于 Dense Patch 分类
            patch_tokens = model.forward_features(images)["x_norm_patchtokens"]

        optimizer.zero_grad()
        patch_logits, img_logits, patch_feat = classifier_head(patch_tokens)
        loss = criterion(img_logits, labels)
        loss.backward()
        optimizer.step()

        preds = img_logits.argmax(dim=1)
        t_corr += (preds == labels).sum().item()
        t_tot += labels.size(0)
        t_loss_sum += loss.item() * labels.size(0)

    classifier_head.eval()
    v_corr, v_tot = 0, 0
    current_features, current_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            # 提取 patch tokens
            patch_tokens = model.forward_features(images)["x_norm_patchtokens"]
            patch_logits, img_logits, patch_feat = classifier_head(patch_tokens)

            preds = img_logits.argmax(dim=1)
            v_corr += (preds == labels).sum().item()
            v_tot += labels.size(0)

            # 保存图像级特征中心估计所需特征：
            # 这里改成 patch feature 的平均，和 dense head 更一致
            img_feat = patch_feat.mean(dim=1)  # [B, C]
            current_features.append(img_feat.cpu().numpy())
            current_labels.append(labels.cpu().numpy())

    train_acc = 100.0 * t_corr / t_tot
    val_acc = 100.0 * v_corr / v_tot
    train_loss = t_loss_sum / t_tot

    print(
        f"Epoch [{epoch+1}/{epochs}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.2f}% | "
        f"Val Acc: {val_acc:.2f}%"
    )

    # 保存最佳结果
    if val_acc >= best_val_acc:
        best_val_acc = val_acc
        best_features = np.concatenate(current_features, axis=0)
        best_labels = np.concatenate(current_labels, axis=0)

        np.save(f"{output_dir}/test_attn_features.npy", best_features)
        np.save(f"{output_dir}/test_attn_labels.npy", best_labels)
        torch.save(classifier_head.state_dict(), f"{output_dir}/nwpu_attn_head_best.pth")

        print(f"✅ 已更新最佳模型，Val Acc = {best_val_acc:.2f}%")

# --- 8. 持久化说明 ---
print(f"🚀 Dense Patch Head 训练任务全部完成！结果已保存至 {output_dir}")
print(f"   - Head 权重: {output_dir}/nwpu_attn_head_best.pth")
print(f"   - 特征文件: {output_dir}/test_attn_features.npy")
print(f"   - 标签文件: {output_dir}/test_attn_labels.npy")