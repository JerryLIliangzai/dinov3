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

# --- 1. 核心路径修复：确保 Python 能够找到本地的 dinov3 模块 ---
# 使用 insert(0, ...) 确保本地代码优先级最高，解决 "本地 DINOv3 不可用"
project_root = "/root/dinov3"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 2. 参数配置 (统一使用绝对路径，最稳健) ---
data_root = "/root/autodl-tmp/datasets/NWPU_data" 
label_map_path = "/root/autodl-tmp/datasets/label_map.json"
output_dir = "/root/autodl-tmp/outputs"
# 直接指向 autodl-tmp 里的原始权重，不需要软链接
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
    # 强制修正 samples 中的标签索引，确保与 JSON 映射一致
    new_samples = []
    for path, _ in ds.samples:
        folder_name = os.path.basename(os.path.dirname(path))
        new_samples.append((path, c2i[folder_name]))
    ds.classes, ds.class_to_idx, ds.samples, ds.targets = ordered_classes, c2i, new_samples, [s[1] for s in new_samples]
    return ds

# 物理隔离划分
train_base = get_fixed_dataset(data_root, ordered_classes, standard_transform)
val_base = get_fixed_dataset(data_root, ordered_classes, standard_transform)

indices = torch.randperm(len(train_base), generator=torch.Generator().manual_seed(42)).tolist()
split = int(0.8 * len(train_base))

train_loader = DataLoader(Subset(train_base, indices[:split]), batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(Subset(val_base, indices[split:]), batch_size=batch_size, shuffle=False, num_workers=8)

print(f"✅ 数据准备就绪！训练集: {split}, 验证集: {len(train_base)-split}")

# # --- 5. 模型加载 (适配多种源码结构) ---
print("正在尝试从本地加载 DINOv3 Backbone...")
try:
    # 方案 A: 尝试从 dinov3.hub 导入 (如果它是模块)
    try:
        from dinov3.hub import dinov3_vitl16
        print("✅ 成功通过 dinov3.hub 导入")
    except ImportError:
        # 方案 B: 许多 DINO 变体将入口放在根目录的 hubconf.py
        # 既然我们在 /root/dinov3 运行，直接导入 hubconf
        import hubconf
        dinov3_vitl16 = hubconf.dinov3_vitl16
        print("✅ 成功通过 hubconf 导入")
    
    model = dinov3_vitl16(pretrained=False)
    
    if os.path.exists(weights_path):
        print(f"正在读取本地权重文件: {weights_path}")
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
    print("请检查 hubconf.py 中是否存在 dinov3_vitl16 函数。")
    sys.exit(1)

# 冻结参数并初始化分类头
for param in model.parameters():
    param.requires_grad = False

classifier_head = nn.Linear(model.embed_dim, num_classes)
nn.init.normal_(classifier_head.weight, std=0.01)
nn.init.constant_(classifier_head.bias, 0)

model.to(device)
classifier_head.to(device)

# --- 6. 训练与验证 ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier_head.parameters(), lr=lr)

for epoch in range(epochs):
    model.eval()
    classifier_head.train()
    t_corr, t_tot = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            features = model.forward_features(images)['x_norm_clstoken']
        optimizer.zero_grad()
        outputs = classifier_head(features)
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
            feat = model.forward_features(images)['x_norm_clstoken']
            outputs = classifier_head(feat)
            v_corr += (outputs.max(1)[1] == labels).sum().item()
            v_tot += labels.size(0)
            if epoch == epochs - 1:
                all_features.append(feat.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

    print(f"Epoch [{epoch+1}/{epochs}] | Train Acc: {100*t_corr/t_tot:.2f}% | Val Acc: {100*v_corr/v_tot:.2f}%")

# --- 7. 持久化 ---
np.save(f"{output_dir}/test_features.npy", np.concatenate(all_features))
np.save(f"{output_dir}/test_labels.npy", np.concatenate(all_labels))
torch.save(classifier_head.state_dict(), f"{output_dir}/nwpu_head_best.pth")
print(f"🚀 任务全部完成！结果已保存至 {output_dir}")