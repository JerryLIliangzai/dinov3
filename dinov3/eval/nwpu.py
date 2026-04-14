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
    ds.classes, ds.class_to_idx, ds.samples, ds.targets = ordered_classes, c2i, new_samples, [s[1] for s in new_samples]
    return ds

train_base = get_fixed_dataset(data_root, ordered_classes, standard_transform)
val_base = get_fixed_dataset(data_root, ordered_classes, standard_transform)

indices = torch.randperm(len(train_base), generator=torch.Generator().manual_seed(42)).tolist()
split = int(0.8 * len(train_base))

train_loader = DataLoader(Subset(train_base, indices[:split]), batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(Subset(val_base, indices[split:]), batch_size=batch_size, shuffle=False, num_workers=8)

# --- 5. 定义 Attention Classifier Head ---
class AttentionClassifier(nn.Module):
    def __init__(self, embed_dim, num_classes, num_heads=8, dropout=0.1):
        super().__init__()
        # 任务书要求：可学习的 Query 用于从 Patch 中提取分类特征
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        # 为后续蒙特卡洛 Dropout 任务做准备
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x_patches):
        # x_patches: [Batch, Num_Patches, Embed_Dim]
        b = x_patches.shape[0]
        q = self.query.expand(b, -1, -1)
        
        # Cross-attention: Query 关注所有的 Patch Tokens
        attn_out, attn_weights = self.attn(q, x_patches, x_patches)
        
        # 提取聚合后的特征向量
        feat = attn_out.squeeze(1)
        feat = self.dropout(feat)
        logits = self.fc(feat)
        return logits, feat, attn_weights

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
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
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

# 初始化新的 Attention Classifier
classifier_head = AttentionClassifier(model.embed_dim, num_classes)
model.to(device)
classifier_head.to(device)

# --- 7. 训练与验证 ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier_head.parameters(), lr=lr)

for epoch in range(epochs):
    model.eval() # Backbone 始终保持 eval 模式
    classifier_head.train()
    t_corr, t_tot = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        with torch.no_grad():
            # 获取全部 Patch Tokens 用于 Attention 聚合
            patch_tokens = model.forward_features(images)['x_norm_patchtokens']
            
        optimizer.zero_grad()
        outputs, _, _ = classifier_head(patch_tokens)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        t_corr += (outputs.max(1)[1] == labels).sum().item()
        t_tot += labels.size(0)

    classifier_head.eval()
    v_corr, v_tot = 0, 0
    all_features, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            # 提取 patch tokens
            patch_tokens = model.forward_features(images)['x_norm_patchtokens']
            outputs, feat, _ = classifier_head(patch_tokens)
            
            v_corr += (outputs.max(1)[1] == labels).sum().item()
            v_tot += labels.size(0)
            
            if epoch == epochs - 1:
                # 任务书要求：保存特征空间密度估计所需的特征向量 
                all_features.append(feat.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

    print(f"Epoch [{epoch+1}/{epochs}] | Train Acc: {100*t_corr/t_tot:.2f}% | Val Acc: {100*v_corr/v_tot:.2f}%")

# --- 8. 持久化 (更新文件名) ---
np.save(f"{output_dir}/test_attn_features.npy", np.concatenate(all_features))
np.save(f"{output_dir}/test_attn_labels.npy", np.concatenate(all_labels))
# 保存新的 Attention Head 权重
torch.save(classifier_head.state_dict(), f"{output_dir}/nwpu_attn_head_best.pth")
print(f"🚀 Attention Head 训练任务全部完成！结果已保存至 {output_dir}")