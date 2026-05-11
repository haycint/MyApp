"""
Memory System Comparison Experiment
====================================
对比只使用图像记忆、只使用点云记忆、二者都使用时的性能（准确率和吞吐量）。

实验矩阵:
  记忆类型: image_only | point_cloud_only | both
  图像记忆模式: feature_average | heatmap_average
  模型: IAG | IAG_TextEmb
  数据集: Seen | Unseen

记忆使用离线预填充方式:
  - 点云记忆: 每种物品的每种动作从数据集内随机选择一个点云进行填充
  - 图像记忆:
    (a) 特征平均模式: 将多张图片的特征平均后送入模型
    (b) 热力图平均模式: n张图片对一个点云分别对应，将热力图平均

使用方式:
    # 完整实验 (Seen + Unseen, 全部组合)
    python experiments/memory_comparison.py --data_dir ./Data

    # 仅Seen
    python experiments/memory_comparison.py --setting Seen --data_dir ./Data

    # 指定已训练模型checkpoint
    python experiments/memory_comparison.py --iag_ckpt ./ckpt/IAG-Seen/best.pt \
        --textemb_ckpt ./ckpt/IAG_TextEmb-Seen/best.pt --data_dir ./Data

输出:
    - 对比结果表格保存到 ./experiments/results/memory_comparison.txt
"""

import os
import sys
import time
import random
import json
import numpy as np
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from model.MyNet import get_MyNet, get_IAG_TextEmb
from data_utils.dataset import PIAD
from utils.loss import HM_Loss, kl_div
from utils.eval import SIM
from memory_system.memory_manager import MemoryManager
from memory_system.integration import (
    prepopulate_from_dataset,
    MemoryEnhancedInference,
    _capture_arm_feature,
    _capture_point_features,
)

# ============================================================================
# 常量
# ============================================================================

AFFORDANCE_LABELS = [
    'grasp', 'contain', 'lift', 'open', 'lay', 'sit', 'support',
    'wrapgrasp', 'pour', 'move', 'display', 'push', 'listen',
    'wear', 'press', 'cut', 'stab'
]

CKPT_DIR = os.path.join(PROJECT_ROOT, 'ckpt')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'experiments', 'results')

DEFAULT_CONFIG = {
    'batch_size': 8, 'lr': 0.0001, 'Epoch': 80,
    'loss_cls': 0.3, 'loss_kl': 0.5,
    'N_p': 64, 'emb_dim': 512, 'proj_dim': 512,
    'num_heads': 4, 'N_raw': 2048, 'num_affordance': 17,
    'pairing_num': 2, 'text_dim': 300,
}


# ============================================================================
# 辅助函数
# ============================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(torch.cuda.is_available())
    torch.backends.cudnn.deterministic = True


def prepare_text_embeddings(config, glove_path=None):
    """简化版: 生成fallback文本嵌入"""
    text_dim = config['text_dim']
    from experiments.train_textemb_comparison import generate_fallback_embeddings
    fallback = generate_fallback_embeddings(AFFORDANCE_LABELS, text_dim=text_dim)
    emb_matrix = np.zeros((len(AFFORDANCE_LABELS), text_dim))
    for i, label in enumerate(AFFORDANCE_LABELS):
        emb_matrix[i] = fallback.get(label, np.random.randn(text_dim) * 0.1)
    return emb_matrix


def load_model(model_type, config, device, checkpoint_path=None):
    """加载训练好的模型"""
    if model_type == 'IAG':
        model = get_MyNet(
            pre_train=False, N_p=config['N_p'], emb_dim=config['emb_dim'],
            proj_dim=config['proj_dim'], num_heads=config['num_heads'],
            N_raw=config['N_raw'], num_affordance=config['num_affordance']
        )
    else:
        model = get_IAG_TextEmb(
            pre_train=False, N_p=config['N_p'], emb_dim=config['emb_dim'],
            proj_dim=config['proj_dim'], num_heads=config['num_heads'],
            N_raw=config['N_raw'], num_affordance=config['num_affordance'],
            text_dim=config['text_dim']
        )

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        print(f"  Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"  Warning: No checkpoint found for {model_type}, using random weights")

    model = model.to(device)
    model.eval()
    return model


# ============================================================================
# 离线预填充 - 点云记忆
# ============================================================================

def prefill_point_cloud_memory(model, data_dir, setting, device, config,
                                use_text_emb=False, affordance_emb_tensor=None):
    """
    点云记忆预填充: 每种物品的每种动作从数据集内随机选择一个点云进行填充。

    使用PIAD训练集作为记忆源。按照(object_name, affordance_name)分组，
    每组随机选1个样本存入记忆系统。
    """
    print("\n[Prefill] Point cloud memory pre-filling...")

    manager = MemoryManager(emb_dim=config['emb_dim'], index_dim=128, feat_dim=config['emb_dim'])

    data_path = os.path.join(data_dir, setting)
    train_dataset = PIAD('train', setting,
                         os.path.join(data_path, 'Point_Train.txt'),
                         os.path.join(data_path, 'Img_Train.txt'),
                         os.path.join(data_path, 'Box_Train.txt'),
                         config['pairing_num'])
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=0, shuffle=False)

    # 按 (object_name, affordance_name) 分组
    groups = defaultdict(list)
    for idx in range(len(train_dataset)):
        img_path = train_dataset.img_files[idx]
        parts = img_path.split('_')
        # 路径格式: .../Object_Action_..._img.png
        object_name = parts[-3]
        affordance_name = parts[-2]
        if affordance_name in AFFORDANCE_LABELS:
            groups[(object_name, affordance_name)].append(idx)

    # 每组随机选1个
    selected_indices = []
    for key, indices in groups.items():
        chosen = random.choice(indices)
        selected_indices.append((key, chosen))

    print(f"  Total groups: {len(groups)}, Selected: {len(selected_indices)}")

    # 填充记忆
    count = 0
    with torch.no_grad():
        for (obj_name, aff_name), idx in selected_indices:
            try:
                sample = train_dataset[idx]
                img = sample[0].unsqueeze(0).to(device)
                points_list = sample[1]
                labels_list = sample[2]
                indices_list = sample[3]
                sub_box = sample[4].unsqueeze(0).to(device)
                obj_box = sample[5].unsqueeze(0).to(device)

                # 使用第一个配对点云
                point = points_list[0].float().unsqueeze(0).to(device)
                label = labels_list[0]
                aff_idx = indices_list[0].item() if isinstance(indices_list[0], torch.Tensor) else indices_list[0]

                # 捕获ARM特征
                arm_output = _capture_arm_feature(
                    model, img, point, sub_box, obj_box,
                    use_text_emb=use_text_emb,
                    affordance_emb_tensor=affordance_emb_tensor,
                    aff_idx=aff_idx
                )

                # 获取点特征
                point_features = _capture_point_features(model, point)

                # 生成偏好矩阵
                gt_label = label.cpu().numpy()
                if gt_label.ndim > 1:
                    gt_label = gt_label.squeeze()
                pref = MemoryManager.generate_preference_from_ground_truth(gt_label, reward=1.0)

                # 存入记忆
                point_cloud_np = point.cpu().numpy().squeeze().T

                text_emb_np = None
                if use_text_emb and affordance_emb_tensor is not None:
                    text_emb_np = affordance_emb_tensor[aff_idx].cpu().numpy()

                manager.form_memory(
                    arm_feature=arm_output,
                    point_cloud=point_cloud_np,
                    point_features=point_features.cpu().numpy(),
                    preference_matrix=pref,
                    reward=1.0,
                    outcome="success",
                    object_category=obj_name,
                    affordance_label=aff_name,
                    confidence=1.0,
                    text_embedding=text_emb_np,
                )
                count += 1

            except Exception as e:
                print(f"  [Warning] Error at {obj_name}/{aff_name}: {e}")
                continue

    print(f"  Point cloud memories stored: {count}")
    return manager


# ============================================================================
# 离线预填充 - 图像记忆
# ============================================================================

class ImageMemoryStore:
    """
    图像记忆存储: 存储每张图片的中间特征，用于推理时的特征平均或热力图平均。

    两种使用模式:
    1. feature_average: 将同一(object, affordance)的多张图片的特征平均后送入模型
    2. heatmap_average: 每张图片分别通过模型生成热力图，然后对热力图取平均
    """

    def __init__(self, mode='feature_average'):
        self.mode = mode
        # 存储: {(object_name, affordance_name): [list of features]}
        self.features = defaultdict(list)
        # 存储: {(object_name, affordance_name): [list of (img, sub_box, obj_box)]}
        self.raw_data = defaultdict(list)

    def add_feature(self, obj_name, aff_name, feature):
        """添加图片特征"""
        self.features[(obj_name, aff_name)].append(feature)

    def add_raw(self, obj_name, aff_name, img, sub_box, obj_box):
        """添加原始图片数据"""
        self.raw_data[(obj_name, aff_name)].append((img, sub_box, obj_box))

    def get_average_feature(self, obj_name, aff_name):
        """获取特征平均后的特征"""
        key = (obj_name, aff_name)
        if key not in self.features or len(self.features[key]) == 0:
            return None
        return torch.stack(self.features[key]).mean(dim=0)

    def get_raw_data(self, obj_name, aff_name):
        """获取原始图片数据"""
        key = (obj_name, aff_name)
        if key not in self.raw_data:
            return []
        return self.raw_data[key]


def prefill_image_memory(model, data_dir, setting, device, config,
                          use_text_emb=False, affordance_emb_tensor=None):
    """
    图像记忆预填充: 对Seen和Unseen数据集均进行填充。

    为每张图片提取ResNet18的中间特征(F_I)以及ROI特征(F_i, F_s, F_e)。
    同一(object, affordance)的图片特征存储在一起，支持后续的特征平均或热力图平均。
    """
    print("\n[Prefill] Image memory pre-filling...")

    img_memory = ImageMemoryStore(mode='both')

    data_path = os.path.join(data_dir, setting)

    # 填充训练集图片
    for split in ['Train', 'Test']:
        img_list_file = os.path.join(data_path, f'Img_{split}.txt')
        box_list_file = os.path.join(data_path, f'Box_{split}.txt')

        if not os.path.exists(img_list_file):
            continue

        img_files = []
        box_files = []
        with open(img_list_file, 'r') as f:
            img_files = [line.strip() for line in f if line.strip()]
        with open(box_list_file, 'r') as f:
            box_files = [line.strip() for line in f if line.strip()]

        count = 0
        with torch.no_grad():
            for img_path, box_path in zip(img_files, box_files):
                if not os.path.exists(img_path) or not os.path.exists(box_path):
                    continue

                try:
                    from PIL import Image
                    from torchvision import transforms

                    img = Image.open(img_path).convert('RGB')
                    original_size = img.size

                    # 解析标注
                    with open(box_path, 'r') as f:
                        box_data = json.load(f)
                    sub_box_pts = None
                    obj_box_pts = None
                    for shape in box_data.get('shapes', []):
                        if shape['label'] == 'subject':
                            sub_box_pts = shape['points']
                        elif shape['label'] == 'object':
                            obj_box_pts = shape['points']

                    if sub_box_pts is None or obj_box_pts is None:
                        continue

                    # 提取物品名称和动作名称
                    parts = img_path.split('_')
                    obj_name = parts[-3]
                    aff_name = parts[-2]
                    if aff_name not in AFFORDANCE_LABELS:
                        continue

                    # 预处理图片
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
                    img_tensor = transform(img).unsqueeze(0).to(device)

                    # 提取图像特征
                    F_I = model.img_encoder(img_tensor)

                    # 存储特征
                    img_memory.add_feature(obj_name, aff_name, F_I.squeeze(0).cpu())

                    count += 1

                except Exception as e:
                    continue

        print(f"  {split}: {count} images stored")

    print(f"  Total unique (object, affordance) groups: {len(img_memory.features)}")
    return img_memory


# ============================================================================
# 评估函数 - 带记忆增强
# ============================================================================

def evaluate_with_point_cloud_memory(model, val_loader, device, config,
                                      memory_manager, use_text_emb=False,
                                      affordance_emb_tensor=None, alpha=0.3, top_k=5):
    """
    使用点云记忆增强评估模型性能。

    对每个验证样本:
    1. 运行模型前向传播
    2. 检索相关记忆
    3. 对齐并融合偏好
    4. 应用融合偏好到模型输出
    """
    model.eval()
    val_dataset = val_loader.dataset
    results = torch.zeros((len(val_dataset), config['N_raw'], 1))
    targets = torch.zeros((len(val_dataset), config['N_raw'], 1))

    num = 0
    start_time = time.time()
    memory_hit_count = 0

    with torch.no_grad():
        for batch_data in val_loader:
            img, point, label, img_paths, point_paths, sub_box, obj_box = batch_data
            point = point.float().to(device)
            label = label.float().unsqueeze(dim=-1).to(device)
            img_dev = img.to(device)
            sub_box_dev = sub_box.to(device)
            obj_box_dev = obj_box.to(device)
            batch_size = img_dev.size(0)

            # 准备文本嵌入
            text_emb = None
            if use_text_emb and affordance_emb_tensor is not None:
                text_emb = torch.zeros(batch_size, config['text_dim']).to(device)
                for b_idx, path in enumerate(img_paths):
                    if isinstance(path, str):
                        for aff_name in AFFORDANCE_LABELS:
                            if aff_name in path:
                                aff_idx = AFFORDANCE_LABELS.index(aff_name)
                                text_emb[b_idx] = affordance_emb_tensor[aff_idx]
                                break

            # 前向传播获取原始输出
            if use_text_emb and text_emb is not None:
                _3d, logits, to_KL = model(img_dev, point, sub_box_dev, obj_box_dev, text_emb)
            else:
                _3d, logits, to_KL = model(img_dev, point, sub_box_dev, obj_box_dev)

            # 尝试记忆增强
            raw_output = _3d.squeeze().cpu().numpy()

            try:
                # 获取ARM特征
                arm_feature = _capture_arm_feature(
                    model, img_dev, point, sub_box_dev, obj_box_dev,
                    use_text_emb=use_text_emb,
                    affordance_emb_tensor=affordance_emb_tensor,
                )

                point_features = _capture_point_features(model, point)
                point_cloud_np = point.squeeze().cpu().numpy().T

                # 确定affordance标签用于过滤
                aff_label = None
                if isinstance(img_paths[0], str):
                    for aff_name in AFFORDANCE_LABELS:
                        if aff_name in img_paths[0]:
                            aff_label = aff_name
                            break

                pref_fused = memory_manager.retrieve_and_fuse(
                    arm_feature=arm_feature,
                    current_point_cloud=point_cloud_np,
                    current_point_features=point_features.cpu().numpy(),
                    top_k=top_k,
                    affordance_label=aff_label,
                )

                if np.abs(pref_fused).sum() > 1e-6:
                    enhanced_raw = memory_manager.apply_memory_to_output(
                        raw_output, pref_fused, alpha=alpha
                    )
                    _3d_enhanced = 1.0 / (1.0 + np.exp(-enhanced_raw))
                    _3d = torch.from_numpy(_3d_enhanced).unsqueeze(0).unsqueeze(-1).to(device)
                    memory_hit_count += 1

            except Exception as e:
                pass  # 使用原始输出

            pred_num = _3d.shape[0]
            results[num:num + pred_num, :, :] = _3d.cpu()
            targets[num:num + pred_num, :, :] = label.cpu()
            num += pred_num

    elapsed = time.time() - start_time
    throughput = num / elapsed if elapsed > 0 else 0

    return _compute_metrics(results, targets, throughput, memory_hit_count, num)


def evaluate_with_image_memory_feature_avg(model, val_loader, device, config,
                                            img_memory, use_text_emb=False,
                                            affordance_emb_tensor=None):
    """
    使用图像记忆 (特征平均模式) 评估。

    特征平均模式: 对于每个验证样本，查找同类(object, affordance)的已存储图像特征，
    将多张图片的特征平均后作为额外的上下文信息送入模型。
    具体做法: 用平均特征替换当前图片特征中的场景特征(F_e)，
    重新运行模型后半部分(JRA+ARM+Decoder)。
    """
    model.eval()
    val_dataset = val_loader.dataset
    results = torch.zeros((len(val_dataset), config['N_raw'], 1))
    targets = torch.zeros((len(val_dataset), config['N_raw'], 1))

    num = 0
    start_time = time.time()
    avg_used_count = 0

    with torch.no_grad():
        for batch_data in val_loader:
            img, point, label, img_paths, point_paths, sub_box, obj_box = batch_data
            point = point.float().to(device)
            label = label.float().unsqueeze(dim=-1).to(device)
            img_dev = img.to(device)
            sub_box_dev = sub_box.to(device)
            obj_box_dev = obj_box.to(device)
            batch_size = img_dev.size(0)

            text_emb = None
            if use_text_emb and affordance_emb_tensor is not None:
                text_emb = torch.zeros(batch_size, config['text_dim']).to(device)
                for b_idx, path in enumerate(img_paths):
                    if isinstance(path, str):
                        for aff_name in AFFORDANCE_LABELS:
                            if aff_name in path:
                                aff_idx = AFFORDANCE_LABELS.index(aff_name)
                                text_emb[b_idx] = affordance_emb_tensor[aff_idx]
                                break

            # 常规前向传播
            if use_text_emb and text_emb is not None:
                _3d, logits, to_KL = model(img_dev, point, sub_box_dev, obj_box_dev, text_emb)
            else:
                _3d, logits, to_KL = model(img_dev, point, sub_box_dev, obj_box_dev)

            # 尝试特征平均增强
            try:
                for b_idx, path in enumerate(img_paths):
                    if isinstance(path, str):
                        parts = path.split('_')
                        obj_name = parts[-3]
                        aff_name = parts[-2]
                        avg_feat = img_memory.get_average_feature(obj_name, aff_name)
                        if avg_feat is not None:
                            # 使用平均特征增强: 简单地将原始输出与基于平均特征的
                            # 输出进行加权平均
                            avg_feat_dev = avg_feat.unsqueeze(0).to(device)
                            # 使用平均特征替换图像特征，重新运行
                            F_I_avg = avg_feat_dev
                            F_i_avg, F_s_avg, F_e_avg = model.get_mask_feature(
                                img_dev[b_idx:b_idx+1], F_I_avg,
                                sub_box_dev[b_idx:b_idx+1], obj_box_dev[b_idx:b_idx+1],
                                device
                            )
                            from torchvision.ops import roi_align
                            ROI_box = model.get_roi_box(1).to(device)
                            F_e_avg_aligned = roi_align(F_e_avg, ROI_box, output_size=(4, 4))

                            F_p_wise = model.point_encoder(point[b_idx:b_idx+1])
                            F_j_avg = model.JRA(F_i_avg, F_p_wise[-1][1])
                            affordance_avg = model.ARM(F_j_avg, F_s_avg, F_e_avg_aligned)

                            if use_text_emb and text_emb is not None:
                                _3d_avg, _, _ = model.decoder(F_j_avg, affordance_avg, F_p_wise, text_emb[b_idx:b_idx+1])
                            else:
                                _3d_avg, _, _ = model.decoder(F_j_avg, affordance_avg, F_p_wise)

                            # 混合原始输出和记忆增强输出
                            _3d[b_idx] = 0.5 * _3d[b_idx] + 0.5 * _3d_avg[0]
                            avg_used_count += 1
            except Exception as e:
                pass  # 使用原始输出

            pred_num = _3d.shape[0]
            results[num:num + pred_num, :, :] = _3d.cpu()
            targets[num:num + pred_num, :, :] = label.cpu()
            num += pred_num

    elapsed = time.time() - start_time
    throughput = num / elapsed if elapsed > 0 else 0
    return _compute_metrics(results, targets, throughput, avg_used_count, num)


def evaluate_with_image_memory_heatmap_avg(model, val_loader, device, config,
                                            img_memory, use_text_emb=False,
                                            affordance_emb_tensor=None):
    """
    使用图像记忆 (热力图平均模式) 评估。

    热力图平均模式: 对于每个验证样本，查找同类(object, affordance)的所有已存储图片，
    每张图片分别与当前点云配对运行模型生成3D affordance热力图，
    然后将n张图片对应的热力图取平均作为最终结果。
    """
    model.eval()
    val_dataset = val_loader.dataset
    results = torch.zeros((len(val_dataset), config['N_raw'], 1))
    targets = torch.zeros((len(val_dataset), config['N_raw'], 1))

    num = 0
    start_time = time.time()
    avg_used_count = 0

    with torch.no_grad():
        for batch_data in val_loader:
            img, point, label, img_paths, point_paths, sub_box, obj_box = batch_data
            point = point.float().to(device)
            label = label.float().unsqueeze(dim=-1).to(device)
            img_dev = img.to(device)
            sub_box_dev = sub_box.to(device)
            obj_box_dev = obj_box.to(device)
            batch_size = img_dev.size(0)

            text_emb = None
            if use_text_emb and affordance_emb_tensor is not None:
                text_emb = torch.zeros(batch_size, config['text_dim']).to(device)
                for b_idx, path in enumerate(img_paths):
                    if isinstance(path, str):
                        for aff_name in AFFORDANCE_LABELS:
                            if aff_name in path:
                                aff_idx = AFFORDANCE_LABELS.index(aff_name)
                                text_emb[b_idx] = affordance_emb_tensor[aff_idx]
                                break

            # 常规前向传播
            if use_text_emb and text_emb is not None:
                _3d, logits, to_KL = model(img_dev, point, sub_box_dev, obj_box_dev, text_emb)
            else:
                _3d, logits, to_KL = model(img_dev, point, sub_box_dev, obj_box_dev)

            # 尝试热力图平均增强
            heatmap_list = [_3d.clone()]
            try:
                for b_idx, path in enumerate(img_paths):
                    if isinstance(path, str):
                        parts = path.split('_')
                        obj_name = parts[-3]
                        aff_name = parts[-2]
                        raw_data = img_memory.get_raw_data(obj_name, aff_name)

                        for raw_img, raw_sub_box, raw_obj_box in raw_data[:3]:  # 最多3张
                            raw_img_dev = raw_img.unsqueeze(0).to(device) if raw_img.dim() == 3 else raw_img.to(device)
                            raw_sub_box_dev = raw_sub_box.unsqueeze(0).to(device)
                            raw_obj_box_dev = raw_obj_box.unsqueeze(0).to(device)

                            if use_text_emb and text_emb is not None:
                                _3d_raw, _, _ = model(raw_img_dev, point[b_idx:b_idx+1],
                                                       raw_sub_box_dev, raw_obj_box_dev,
                                                       text_emb[b_idx:b_idx+1])
                            else:
                                _3d_raw, _, _ = model(raw_img_dev, point[b_idx:b_idx+1],
                                                       raw_sub_box_dev, raw_obj_box_dev)
                            heatmap_list.append(_3d_raw)

                if len(heatmap_list) > 1:
                    _3d = torch.stack(heatmap_list).mean(dim=0)
                    avg_used_count += 1

            except Exception as e:
                pass  # 使用原始输出

            pred_num = _3d.shape[0]
            results[num:num + pred_num, :, :] = _3d.cpu()
            targets[num:num + pred_num, :, :] = label.cpu()
            num += pred_num

    elapsed = time.time() - start_time
    throughput = num / elapsed if elapsed > 0 else 0
    return _compute_metrics(results, targets, throughput, avg_used_count, num)


def evaluate_with_both_memories(model, val_loader, device, config,
                                 pc_memory_manager, img_memory,
                                 img_mode='feature_average',
                                 use_text_emb=False,
                                 affordance_emb_tensor=None,
                                 alpha=0.3, top_k=5):
    """
    同时使用图像记忆和点云记忆评估。

    先应用点云记忆增强，再应用图像记忆增强。
    """
    model.eval()
    val_dataset = val_loader.dataset
    results = torch.zeros((len(val_dataset), config['N_raw'], 1))
    targets = torch.zeros((len(val_dataset), config['N_raw'], 1))

    num = 0
    start_time = time.time()
    both_used_count = 0

    with torch.no_grad():
        for batch_data in val_loader:
            img, point, label, img_paths, point_paths, sub_box, obj_box = batch_data
            point = point.float().to(device)
            label = label.float().unsqueeze(dim=-1).to(device)
            img_dev = img.to(device)
            sub_box_dev = sub_box.to(device)
            obj_box_dev = obj_box.to(device)
            batch_size = img_dev.size(0)

            text_emb = None
            if use_text_emb and affordance_emb_tensor is not None:
                text_emb = torch.zeros(batch_size, config['text_dim']).to(device)
                for b_idx, path in enumerate(img_paths):
                    if isinstance(path, str):
                        for aff_name in AFFORDANCE_LABELS:
                            if aff_name in path:
                                aff_idx = AFFORDANCE_LABELS.index(aff_name)
                                text_emb[b_idx] = affordance_emb_tensor[aff_idx]
                                break

            # 前向传播
            if use_text_emb and text_emb is not None:
                _3d, logits, to_KL = model(img_dev, point, sub_box_dev, obj_box_dev, text_emb)
            else:
                _3d, logits, to_KL = model(img_dev, point, sub_box_dev, obj_box_dev)

            # 点云记忆增强
            try:
                arm_feature = _capture_arm_feature(
                    model, img_dev, point, sub_box_dev, obj_box_dev,
                    use_text_emb=use_text_emb,
                    affordance_emb_tensor=affordance_emb_tensor,
                )
                point_features = _capture_point_features(model, point)
                point_cloud_np = point.squeeze().cpu().numpy().T

                aff_label = None
                if isinstance(img_paths[0], str):
                    for aff_name in AFFORDANCE_LABELS:
                        if aff_name in img_paths[0]:
                            aff_label = aff_name
                            break

                pref_fused = pc_memory_manager.retrieve_and_fuse(
                    arm_feature=arm_feature,
                    current_point_cloud=point_cloud_np,
                    current_point_features=point_features.cpu().numpy(),
                    top_k=top_k,
                    affordance_label=aff_label,
                )

                if np.abs(pref_fused).sum() > 1e-6:
                    raw_output = _3d.squeeze().cpu().numpy()
                    enhanced_raw = pc_memory_manager.apply_memory_to_output(
                        raw_output, pref_fused, alpha=alpha
                    )
                    _3d_enhanced = 1.0 / (1.0 + np.exp(-enhanced_raw))
                    _3d = torch.from_numpy(_3d_enhanced).unsqueeze(0).unsqueeze(-1).to(device)
                    both_used_count += 1
            except Exception:
                pass

            # 图像记忆增强 (特征平均)
            if img_mode == 'feature_average':
                try:
                    for b_idx, path in enumerate(img_paths):
                        if isinstance(path, str):
                            parts = path.split('_')
                            obj_name = parts[-3]
                            aff_name = parts[-2]
                            avg_feat = img_memory.get_average_feature(obj_name, aff_name)
                            if avg_feat is not None:
                                avg_feat_dev = avg_feat.unsqueeze(0).to(device)
                                F_i_avg, F_s_avg, F_e_avg = model.get_mask_feature(
                                    img_dev[b_idx:b_idx+1], avg_feat_dev,
                                    sub_box_dev[b_idx:b_idx+1], obj_box_dev[b_idx:b_idx+1],
                                    device
                                )
                                from torchvision.ops import roi_align
                                ROI_box = model.get_roi_box(1).to(device)
                                F_e_avg_aligned = roi_align(F_e_avg, ROI_box, output_size=(4, 4))
                                F_p_wise = model.point_encoder(point[b_idx:b_idx+1])
                                F_j_avg = model.JRA(F_i_avg, F_p_wise[-1][1])
                                affordance_avg = model.ARM(F_j_avg, F_s_avg, F_e_avg_aligned)
                                if use_text_emb and text_emb is not None:
                                    _3d_avg, _, _ = model.decoder(F_j_avg, affordance_avg, F_p_wise, text_emb[b_idx:b_idx+1])
                                else:
                                    _3d_avg, _, _ = model.decoder(F_j_avg, affordance_avg, F_p_wise)
                                _3d[b_idx] = 0.5 * _3d[b_idx] + 0.5 * _3d_avg[0]
                except Exception:
                    pass

            pred_num = _3d.shape[0]
            results[num:num + pred_num, :, :] = _3d.cpu()
            targets[num:num + pred_num, :, :] = label.cpu()
            num += pred_num

    elapsed = time.time() - start_time
    throughput = num / elapsed if elapsed > 0 else 0
    return _compute_metrics(results, targets, throughput, both_used_count, num)


def evaluate_baseline(model, val_loader, device, config,
                       use_text_emb=False, affordance_emb_tensor=None):
    """无记忆增强的基线评估"""
    model.eval()
    val_dataset = val_loader.dataset
    results = torch.zeros((len(val_dataset), config['N_raw'], 1))
    targets = torch.zeros((len(val_dataset), config['N_raw'], 1))

    num = 0
    start_time = time.time()

    with torch.no_grad():
        for batch_data in val_loader:
            img, point, label, img_paths, point_paths, sub_box, obj_box = batch_data
            point = point.float().to(device)
            label = label.float().unsqueeze(dim=-1).to(device)
            img_dev = img.to(device)
            sub_box_dev = sub_box.to(device)
            obj_box_dev = obj_box.to(device)
            batch_size = img_dev.size(0)

            if use_text_emb and affordance_emb_tensor is not None:
                text_emb = torch.zeros(batch_size, config['text_dim']).to(device)
                for b_idx, path in enumerate(img_paths):
                    if isinstance(path, str):
                        for aff_name in AFFORDANCE_LABELS:
                            if aff_name in path:
                                aff_idx = AFFORDANCE_LABELS.index(aff_name)
                                text_emb[b_idx] = affordance_emb_tensor[aff_idx]
                                break
                _3d, logits, to_KL = model(img_dev, point, sub_box_dev, obj_box_dev, text_emb)
            else:
                _3d, logits, to_KL = model(img_dev, point, sub_box_dev, obj_box_dev)

            pred_num = _3d.shape[0]
            results[num:num + pred_num, :, :] = _3d.cpu()
            targets[num:num + pred_num, :, :] = label.cpu()
            num += pred_num

    elapsed = time.time() - start_time
    throughput = num / elapsed if elapsed > 0 else 0
    return _compute_metrics(results, targets, throughput, 0, num)


def _compute_metrics(results, targets, throughput, memory_hits, num_samples):
    """计算评估指标"""
    results_np = results.numpy()
    targets_np = targets.numpy()

    sim_values = np.zeros(targets_np.shape[0])
    mae_values = np.zeros(targets_np.shape[0])
    auc_values = np.zeros(targets_np.shape[0])
    iou_values = np.zeros(targets_np.shape[0])
    iou_thres = np.linspace(0, 1, 20)
    targets_binary = (targets_np >= 0.5).astype(int)

    for i in range(targets_np.shape[0]):
        sim_values[i] = SIM(results_np[i], targets_np[i])
        mae_values[i] = np.sum(np.abs(results_np[i] - targets_np[i])) / results_np.shape[1]

        t_true = targets_binary[i]
        p_score = results_np[i]
        if np.sum(t_true) == 0:
            auc_values[i] = np.nan
            iou_values[i] = np.nan
        else:
            try:
                auc_values[i] = roc_auc_score(t_true.flatten(), p_score.flatten())
            except ValueError:
                auc_values[i] = np.nan
            temp_iou = []
            for thre in iou_thres:
                p_mask = (p_score >= thre).astype(int)
                intersect = np.sum(p_mask & t_true)
                union = np.sum(p_mask | t_true)
                temp_iou.append(1. * intersect / union if union > 0 else 0)
            iou_values[i] = np.mean(temp_iou)

    return {
        'AUC': np.nanmean(auc_values),
        'IOU': np.nanmean(iou_values),
        'SIM': np.mean(sim_values),
        'MAE': np.mean(mae_values),
        'throughput': throughput,
        'memory_hits': memory_hits,
        'num_samples': num_samples,
    }


# ============================================================================
# 主实验流程
# ============================================================================

def run_experiment(setting, data_dir, config, device,
                    iag_ckpt=None, textemb_ckpt=None,
                    glove_path=None):
    """运行一个setting下的全部记忆对比实验"""

    results = []

    for model_type in ['IAG', 'IAG_TextEmb']:
        print(f"\n{'#' * 90}")
        print(f"# Model: {model_type} | Setting: {setting}")
        print(f"{'#' * 90}")

        use_text_emb = (model_type == 'IAG_TextEmb')
        ckpt_path = iag_ckpt if model_type == 'IAG' else textemb_ckpt

        # 准备文本嵌入
        affordance_emb_tensor = None
        if use_text_emb:
            emb_matrix = prepare_text_embeddings(config, glove_path)
            affordance_emb_tensor = torch.tensor(emb_matrix, dtype=torch.float32).to(device)

        # 加载模型
        model = load_model(model_type, config, device, ckpt_path)

        # 加载验证数据
        data_path = os.path.join(data_dir, setting)
        val_dataset = PIAD('val', setting,
                           os.path.join(data_path, 'Point_Test.txt'),
                           os.path.join(data_path, 'Img_Test.txt'),
                           os.path.join(data_path, 'Box_Test.txt'))
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                                num_workers=0, shuffle=True)

        # ---- 基线 (无记忆) ----
        print(f"\n  [1/6] Baseline (no memory)...")
        baseline_result = evaluate_baseline(
            model, val_loader, device, config,
            use_text_emb=use_text_emb,
            affordance_emb_tensor=affordance_emb_tensor if use_text_emb else None
        )
        baseline_result['model'] = model_type
        baseline_result['setting'] = setting
        baseline_result['memory_type'] = 'none'
        baseline_result['image_mode'] = '-'
        results.append(baseline_result)
        print(f"    AUC={baseline_result['AUC']:.4f} IOU={baseline_result['IOU']:.4f} "
              f"SIM={baseline_result['SIM']:.4f} MAE={baseline_result['MAE']:.4f} "
              f"Throughput={baseline_result['throughput']:.2f}/s")

        # ---- 点云记忆 ----
        print(f"\n  [2/6] Point cloud memory only...")
        pc_manager = prefill_point_cloud_memory(
            model, data_dir, setting, device, config,
            use_text_emb=use_text_emb,
            affordance_emb_tensor=affordance_emb_tensor if use_text_emb else None
        )
        pc_result = evaluate_with_point_cloud_memory(
            model, val_loader, device, config,
            pc_manager, use_text_emb=use_text_emb,
            affordance_emb_tensor=affordance_emb_tensor if use_text_emb else None
        )
        pc_result['model'] = model_type
        pc_result['setting'] = setting
        pc_result['memory_type'] = 'point_cloud'
        pc_result['image_mode'] = '-'
        results.append(pc_result)
        print(f"    AUC={pc_result['AUC']:.4f} IOU={pc_result['IOU']:.4f} "
              f"SIM={pc_result['SIM']:.4f} MAE={pc_result['MAE']:.4f} "
              f"Throughput={pc_result['throughput']:.2f}/s")

        # ---- 图像记忆 (特征平均) ----
        print(f"\n  [3/6] Image memory (feature average)...")
        img_memory_feat = prefill_image_memory(
            model, data_dir, setting, device, config,
            use_text_emb=use_text_emb,
            affordance_emb_tensor=affordance_emb_tensor if use_text_emb else None
        )
        img_feat_result = evaluate_with_image_memory_feature_avg(
            model, val_loader, device, config,
            img_memory_feat, use_text_emb=use_text_emb,
            affordance_emb_tensor=affordance_emb_tensor if use_text_emb else None
        )
        img_feat_result['model'] = model_type
        img_feat_result['setting'] = setting
        img_feat_result['memory_type'] = 'image'
        img_feat_result['image_mode'] = 'feature_avg'
        results.append(img_feat_result)
        print(f"    AUC={img_feat_result['AUC']:.4f} IOU={img_feat_result['IOU']:.4f} "
              f"SIM={img_feat_result['SIM']:.4f} MAE={img_feat_result['MAE']:.4f} "
              f"Throughput={img_feat_result['throughput']:.2f}/s")

        # ---- 图像记忆 (热力图平均) ----
        print(f"\n  [4/6] Image memory (heatmap average)...")
        img_hm_result = evaluate_with_image_memory_heatmap_avg(
            model, val_loader, device, config,
            img_memory_feat, use_text_emb=use_text_emb,
            affordance_emb_tensor=affordance_emb_tensor if use_text_emb else None
        )
        img_hm_result['model'] = model_type
        img_hm_result['setting'] = setting
        img_hm_result['memory_type'] = 'image'
        img_hm_result['image_mode'] = 'heatmap_avg'
        results.append(img_hm_result)
        print(f"    AUC={img_hm_result['AUC']:.4f} IOU={img_hm_result['IOU']:.4f} "
              f"SIM={img_hm_result['SIM']:.4f} MAE={img_hm_result['MAE']:.4f} "
              f"Throughput={img_hm_result['throughput']:.2f}/s")

        # ---- 两者都用 (特征平均) ----
        print(f"\n  [5/6] Both memories (image: feature_avg)...")
        both_feat_result = evaluate_with_both_memories(
            model, val_loader, device, config,
            pc_manager, img_memory_feat, img_mode='feature_average',
            use_text_emb=use_text_emb,
            affordance_emb_tensor=affordance_emb_tensor if use_text_emb else None
        )
        both_feat_result['model'] = model_type
        both_feat_result['setting'] = setting
        both_feat_result['memory_type'] = 'both'
        both_feat_result['image_mode'] = 'feature_avg'
        results.append(both_feat_result)
        print(f"    AUC={both_feat_result['AUC']:.4f} IOU={both_feat_result['IOU']:.4f} "
              f"SIM={both_feat_result['SIM']:.4f} MAE={both_feat_result['MAE']:.4f} "
              f"Throughput={both_feat_result['throughput']:.2f}/s")

        # ---- 两者都用 (热力图平均) ----
        print(f"\n  [6/6] Both memories (image: heatmap_avg)...")
        both_hm_result = evaluate_with_both_memories(
            model, val_loader, device, config,
            pc_manager, img_memory_feat, img_mode='heatmap_average',
            use_text_emb=use_text_emb,
            affordance_emb_tensor=affordance_emb_tensor if use_text_emb else None
        )
        both_hm_result['model'] = model_type
        both_hm_result['setting'] = setting
        both_hm_result['memory_type'] = 'both'
        both_hm_result['image_mode'] = 'heatmap_avg'
        results.append(both_hm_result)
        print(f"    AUC={both_hm_result['AUC']:.4f} IOU={both_hm_result['IOU']:.4f} "
              f"SIM={both_hm_result['SIM']:.4f} MAE={both_hm_result['MAE']:.4f} "
              f"Throughput={both_hm_result['throughput']:.2f}/s")

    return results


def print_comparison_table(all_results):
    """打印记忆对比结果表格"""
    print("\n" + "=" * 120)
    print("Memory System Comparison Results")
    print("=" * 120)

    header = (f"{'Model':<16} {'Setting':<8} {'Memory':<14} {'ImgMode':<14} "
              f"{'AUC':>8} {'IOU':>8} {'SIM':>8} {'MAE':>8} {'Throughput':>12}")
    print(header)
    print("-" * 120)

    for r in all_results:
        line = (f"{r['model']:<16} {r['setting']:<8} {r['memory_type']:<14} "
                f"{r['image_mode']:<14} "
                f"{r['AUC']:>8.4f} {r['IOU']:>8.4f} {r['SIM']:>8.4f} "
                f"{r['MAE']:>8.4f} {r['throughput']:>10.2f}/s")
        print(line)

    print("=" * 120)

    # 保存结果
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_path = os.path.join(RESULTS_DIR, 'memory_comparison.txt')
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("Memory System Comparison Results\n")
        f.write("=" * 120 + "\n")
        f.write(header + "\n")
        f.write("-" * 120 + "\n")
        for r in all_results:
            line = (f"{r['model']:<16} {r['setting']:<8} {r['memory_type']:<14} "
                    f"{r['image_mode']:<14} "
                    f"{r['AUC']:>8.4f} {r['IOU']:>8.4f} {r['SIM']:>8.4f} "
                    f"{r['MAE']:>8.4f} {r['throughput']:>10.2f}/s")
            f.write(line + "\n")
        f.write("=" * 120 + "\n")
    print(f"\nResults saved to: {result_path}")


def main():
    parser = argparse.ArgumentParser(description='Memory System Comparison Experiment')
    parser.add_argument('--setting', type=str, default='Seen', choices=['Seen', 'Unseen', 'both'])
    parser.add_argument('--data_dir', type=str, default=os.path.join(PROJECT_ROOT, 'Data'))
    parser.add_argument('--iag_ckpt', type=str, default=None)
    parser.add_argument('--textemb_ckpt', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=0.3, help='Memory fusion alpha')
    parser.add_argument('--top_k', type=int, default=5, help='Memory retrieval top-k')
    parser.add_argument('--glove_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    config = DEFAULT_CONFIG.copy()
    config['batch_size'] = args.batch_size

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    settings = ['Seen', 'Unseen'] if args.setting == 'both' else [args.setting]
    all_results = []

    for setting in settings:
        results = run_experiment(
            setting=setting,
            data_dir=args.data_dir,
            config=config,
            device=device,
            iag_ckpt=args.iag_ckpt,
            textemb_ckpt=args.textemb_ckpt,
            glove_path=args.glove_path,
        )
        all_results.extend(results)

    if all_results:
        print_comparison_table(all_results)


if __name__ == '__main__':
    main()