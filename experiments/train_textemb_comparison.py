"""
IAG vs IAG_TextEmb Training & Performance Comparison
=====================================================
对比使用text_emb的IAG_TextEmb与原始IAG的性能。

使用方式:
    # Seen数据集训练+对比
    python experiments/train_textemb_comparison.py --setting Seen --data_dir ./Data

    # Unseen数据集训练+对比
    python experiments/train_textemb_comparison.py --setting Unseen --data_dir ./Data

    # 同时运行Seen和Unseen
    python experiments/train_textemb_comparison.py --setting both --data_dir ./Data

    # 仅评估(需要已训练好的checkpoint)
    python experiments/train_textemb_comparison.py --setting Seen --eval_only \
        --iag_ckpt ./ckpt/IAG-Seen/best.pt \
        --textemb_ckpt ./ckpt/IAG_TextEmb-Seen/best.pt

输出:
    - 训练日志和曲线保存到 ./ckpt/ 和 ./logs/
    - 最终对比结果表格打印到终端并保存到 ./experiments/results/
"""

import os
import sys
import argparse
import time
import random
import zipfile
import numpy as np

# 添加项目根路径
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

# ============================================================================
# 常量定义
# ============================================================================

AFFORDANCE_LABELS = [
    'grasp', 'contain', 'lift', 'open', 'lay', 'sit', 'support',
    'wrapgrasp', 'pour', 'move', 'display', 'push', 'listen',
    'wear', 'press', 'cut', 'stab'
]

CKPT_DIR = os.path.join(PROJECT_ROOT, 'ckpt')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
DATA_DIR = os.path.join(PROJECT_ROOT, 'Data')
GLOVE_DIR = os.path.join(PROJECT_ROOT, 'glove')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'experiments', 'results')

# 默认训练参数 (与config_seen.yaml一致)
DEFAULT_CONFIG = {
    'batch_size': 8,
    'lr': 0.0001,
    'Epoch': 80,
    'loss_cls': 0.3,
    'loss_kl': 0.5,
    'N_p': 64,
    'emb_dim': 512,
    'proj_dim': 512,
    'num_heads': 4,
    'N_raw': 2048,
    'num_affordance': 17,
    'pairing_num': 2,
    'text_dim': 300,
}


# ============================================================================
# GloVe 工具函数 (与train_textemb.py一致)
# ============================================================================

_PREBUILT_SEED_VECS = {
    'grasp': [0.23, -0.15, 0.42, 0.08, -0.31, 0.19, 0.05, -0.22, 0.14, 0.37],
    'contain': [-0.11, 0.33, -0.08, 0.25, 0.17, -0.29, 0.41, 0.06, -0.18, 0.22],
    'lift': [0.18, 0.09, -0.25, 0.33, -0.14, 0.28, -0.07, 0.15, 0.31, -0.21],
    'open': [-0.22, 0.17, 0.11, -0.34, 0.26, -0.08, 0.19, -0.12, 0.08, 0.35],
    'lay': [0.07, -0.28, 0.34, 0.12, -0.19, 0.23, -0.05, 0.31, -0.16, 0.09],
    'sit': [-0.15, 0.22, -0.11, 0.28, 0.09, -0.33, 0.17, -0.06, 0.24, -0.14],
    'support': [0.31, -0.07, 0.19, -0.22, 0.14, 0.08, -0.26, 0.33, 0.11, -0.29],
    'wrap': [-0.08, 0.25, -0.17, 0.06, 0.32, -0.11, 0.23, -0.19, 0.15, 0.28],
    'pour': [0.14, -0.21, 0.08, 0.37, -0.05, 0.19, -0.32, 0.07, 0.26, -0.13],
    'move': [0.26, 0.12, -0.29, -0.08, 0.21, 0.33, -0.14, 0.18, -0.07, 0.24],
    'display': [-0.17, 0.31, 0.06, -0.24, 0.15, -0.08, 0.27, -0.11, 0.34, 0.05],
    'push': [0.09, -0.16, 0.22, 0.11, -0.28, 0.07, 0.35, -0.23, 0.12, 0.19],
    'listen': [-0.24, 0.08, 0.33, -0.15, 0.19, -0.27, 0.06, 0.22, -0.09, 0.31],
    'wear': [0.15, 0.24, -0.06, 0.29, -0.12, 0.18, -0.21, 0.09, 0.37, -0.08],
    'press': [0.21, -0.13, 0.17, -0.09, 0.28, -0.15, 0.08, 0.34, -0.22, 0.11],
    'cut': [-0.07, 0.19, -0.31, 0.14, 0.06, 0.23, -0.18, -0.05, 0.29, 0.16],
    'stab': [0.32, -0.11, 0.05, 0.21, -0.24, 0.09, 0.17, -0.33, 0.08, -0.27],
}


def _split_compound_word(word):
    compound_map = {'wrapgrasp': ['wrap', 'grasp']}
    if word in compound_map:
        return compound_map[word]
    common_words = {
        'wrap', 'grasp', 'contain', 'lift', 'open', 'lay', 'sit',
        'support', 'pour', 'move', 'display', 'push', 'listen',
        'wear', 'press', 'cut', 'stab', 'hold', 'place', 'put',
    }
    sub_words = []
    remaining = word
    while remaining:
        found = False
        for end in range(len(remaining), 0, -1):
            prefix = remaining[:end]
            if prefix in common_words:
                sub_words.append(prefix)
                remaining = remaining[end:]
                found = True
                break
        if not found:
            sub_words.append(remaining)
            break
    return sub_words if len(sub_words) > 1 else [word]


def load_glove_embeddings(glove_path, target_words=None):
    embeddings = {}
    target_set = set(target_words) if target_words else None
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ')
            word = parts[0]
            if target_set is not None and word not in target_set:
                continue
            vector = np.array([float(x) for x in parts[1:]])
            embeddings[word] = vector
    return embeddings


def build_affordance_embeddings(glove_embeddings, affordance_labels, text_dim=300):
    emb_matrix = np.zeros((len(affordance_labels), text_dim))
    missing_words = []
    for i, label in enumerate(affordance_labels):
        if label in glove_embeddings:
            emb_matrix[i] = glove_embeddings[label]
        else:
            sub_words = _split_compound_word(label)
            vectors = []
            for w in sub_words:
                if w in glove_embeddings:
                    vectors.append(glove_embeddings[w])
                else:
                    missing_words.append(w)
            if vectors:
                emb_matrix[i] = np.mean(vectors, axis=0)
            else:
                emb_matrix[i] = np.random.randn(text_dim) * 0.1
    return emb_matrix, missing_words


def generate_fallback_embeddings(affordance_labels, text_dim=300, seed=42):
    np.random.seed(seed)
    embeddings = {}
    for label in affordance_labels:
        sub_words = _split_compound_word(label)
        seed_vecs = []
        for w in sub_words:
            if w in _PREBUILT_SEED_VECS:
                seed_vecs.append(np.array(_PREBUILT_SEED_VECS[w]))
        if seed_vecs:
            base_vec = np.mean(seed_vecs, axis=0)
        else:
            hash_val = hash(label) % (2**31)
            np.random.seed(hash_val)
            base_vec = np.random.randn(10) * 0.3
            np.random.seed(seed)
        np.random.seed(hash(label + '_proj') % (2**31))
        proj_matrix = np.random.randn(10, text_dim) / np.sqrt(text_dim)
        full_vec = base_vec @ proj_matrix
        np.random.seed(hash(label + '_noise') % (2**31))
        full_vec += np.random.randn(text_dim) * 0.02
        norm = np.linalg.norm(full_vec)
        if norm > 0:
            full_vec = full_vec / norm
        embeddings[label] = full_vec.astype(np.float32)
    np.random.seed(None)
    return embeddings


def prepare_text_embeddings(config, glove_path=None):
    """准备affordance文本嵌入矩阵"""
    text_dim = config['text_dim']
    if glove_path and os.path.exists(glove_path):
        glove_embeddings = load_glove_embeddings(glove_path, target_words=AFFORDANCE_LABELS)
        for label in AFFORDANCE_LABELS:
            sub_words = _split_compound_word(label)
            for w in sub_words:
                if w not in glove_embeddings:
                    glove_embeddings.update(load_glove_embeddings(glove_path, target_words=[w]))
        emb_matrix, _ = build_affordance_embeddings(glove_embeddings, AFFORDANCE_LABELS, text_dim)
    else:
        fallback_embs = generate_fallback_embeddings(AFFORDANCE_LABELS, text_dim=text_dim)
        emb_matrix = np.zeros((len(AFFORDANCE_LABELS), text_dim))
        for i, label in enumerate(AFFORDANCE_LABELS):
            emb_matrix[i] = fallback_embs.get(label, np.random.randn(text_dim) * 0.1)
    return emb_matrix


# ============================================================================
# 训练和评估函数
# ============================================================================

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU")
    return device


def train_one_epoch_iag(model, train_loader, criterion_hm, criterion_ce,
                        optimizer, device, config):
    """IAG模型训练一个epoch"""
    model.train()
    num_batches = len(train_loader)
    loss_sum = 0.0
    start_time = time.time()

    for i, (img, points, labels, logits_labels, sub_box, obj_box) in enumerate(train_loader):
        optimizer.zero_grad()
        temp_loss = 0.0

        for point, label, logits_label in zip(points, labels, logits_labels):
            point = point.float().to(device)
            label = label.float().unsqueeze(dim=-1).to(device)
            logits_label = logits_label.to(device)
            img_dev = img.to(device)
            sub_box_dev = sub_box.to(device)
            obj_box_dev = obj_box.to(device)

            _3d, logits, to_KL = model(img_dev, point, sub_box_dev, obj_box_dev)

            loss_hm = criterion_hm(_3d, label)
            loss_ce = criterion_ce(logits, logits_label)
            loss_kl = kl_div(to_KL[0], to_KL[1])

            temp_loss += loss_hm + config['loss_cls'] * loss_ce + config['loss_kl'] * loss_kl

        temp_loss.backward()
        optimizer.step()
        loss_sum += temp_loss.item()

    elapsed = time.time() - start_time
    mean_loss = loss_sum / (num_batches * config['pairing_num'])
    return mean_loss, elapsed


def train_one_epoch_textemb(model, train_loader, criterion_hm, criterion_ce,
                            optimizer, device, config, affordance_emb_tensor):
    """IAG_TextEmb模型训练一个epoch"""
    model.train()
    num_batches = len(train_loader)
    loss_sum = 0.0
    start_time = time.time()

    for i, (img, points, labels, logits_labels, sub_box, obj_box) in enumerate(train_loader):
        optimizer.zero_grad()
        temp_loss = 0.0

        for point, label, logits_label in zip(points, labels, logits_labels):
            point = point.float().to(device)
            label = label.float().unsqueeze(dim=-1).to(device)
            logits_label = logits_label.to(device)
            img_dev = img.to(device)
            sub_box_dev = sub_box.to(device)
            obj_box_dev = obj_box.to(device)

            text_emb = affordance_emb_tensor[logits_label]

            _3d, logits, to_KL = model(img_dev, point, sub_box_dev, obj_box_dev, text_emb)

            loss_hm = criterion_hm(_3d, label)
            loss_ce = criterion_ce(logits, logits_label)
            loss_kl = kl_div(to_KL[0], to_KL[1])

            temp_loss += loss_hm + config['loss_cls'] * loss_ce + config['loss_kl'] * loss_kl

        temp_loss.backward()
        optimizer.step()
        loss_sum += temp_loss.item()

    elapsed = time.time() - start_time
    mean_loss = loss_sum / (num_batches * config['pairing_num'])
    return mean_loss, elapsed


def evaluate_model(model, val_loader, criterion_hm, device, config,
                   affordance_emb_tensor=None, use_text_emb=False):
    """评估模型性能，返回AUC, IOU, SIM, MAE, 吞吐量"""
    model.eval()
    val_dataset = val_loader.dataset
    results = torch.zeros((len(val_dataset), config['N_raw'], 1))
    targets = torch.zeros((len(val_dataset), config['N_raw'], 1))

    val_loss_sum = 0.0
    total_mae = 0.0
    total_points = 0
    num = 0
    start_time = time.time()

    with torch.no_grad():
        for batch_data in val_loader:
            if use_text_emb:
                img, point, label, img_paths, point_paths, sub_box, obj_box = batch_data
            else:
                img, point, label, img_paths, point_paths, sub_box, obj_box = batch_data

            point = point.float().to(device)
            label = label.float().unsqueeze(dim=-1).to(device)
            img_dev = img.to(device)
            sub_box_dev = sub_box.to(device)
            obj_box_dev = obj_box.to(device)

            if use_text_emb and affordance_emb_tensor is not None:
                batch_size = img_dev.size(0)
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

            val_loss_hm = criterion_hm(_3d, label)
            val_loss_kl = kl_div(to_KL[0], to_KL[1])
            val_loss = val_loss_hm + config['loss_kl'] * val_loss_kl

            mae = torch.sum(torch.abs(_3d - label), dim=(0, 1))
            point_nums = _3d.shape[0] * _3d.shape[1]
            total_mae += mae.item()
            total_points += point_nums
            val_loss_sum += val_loss.item()

            pred_num = _3d.shape[0]
            results[num:num + pred_num, :, :] = _3d.cpu()
            targets[num:num + pred_num, :, :] = label.cpu()
            num += pred_num

    elapsed = time.time() - start_time
    throughput = num / elapsed if elapsed > 0 else 0

    if total_points == 0:
        return {'AUC': 0, 'IOU': 0, 'SIM': 0, 'MAE': 0, 'throughput': 0}

    mean_mae = total_mae / total_points

    results_np = results.numpy()
    targets_np = targets.numpy()

    # SIM
    sim_values = np.zeros(targets_np.shape[0])
    for i in range(targets_np.shape[0]):
        sim_values[i] = SIM(results_np[i], targets_np[i])
    sim = np.mean(sim_values)

    # AUC and IOU
    auc_values = np.zeros(targets_np.shape[0])
    iou_values = np.zeros(targets_np.shape[0])
    iou_thres = np.linspace(0, 1, 20)
    targets_binary = (targets_np >= 0.5).astype(int)

    for i in range(targets_np.shape[0]):
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

    auc = np.nanmean(auc_values)
    iou = np.nanmean(iou_values)

    return {
        'AUC': auc,
        'IOU': iou,
        'SIM': sim,
        'MAE': mean_mae,
        'throughput': throughput,
    }


def train_and_evaluate(model_type, setting, config, data_dir, device,
                       affordance_emb_tensor=None, resume_path=None,
                       glove_path=None):
    """
    训练并评估一个模型。

    Args:
        model_type: 'IAG' 或 'IAG_TextEmb'
        setting: 'Seen' 或 'Unseen'
        config: 训练配置
        data_dir: 数据目录
        device: 计算设备
        affordance_emb_tensor: 文本嵌入张量 (仅IAG_TextEmb需要)
        resume_path: 恢复训练的checkpoint路径
        glove_path: GloVe文件路径

    Returns:
        dict: 包含训练历史和最终评估结果
    """
    use_text_emb = (model_type == 'IAG_TextEmb')

    # 准备文本嵌入
    if use_text_emb and affordance_emb_tensor is None:
        emb_matrix = prepare_text_embeddings(config, glove_path)
        affordance_emb_tensor = torch.tensor(emb_matrix, dtype=torch.float32).to(device)

    # 加载数据
    data_path = os.path.join(data_dir, setting)
    if not os.path.exists(data_path):
        print(f"[Error] Data directory not found: {data_path}")
        return None

    train_dataset = PIAD('train', setting,
                         os.path.join(data_path, 'Point_Train.txt'),
                         os.path.join(data_path, 'Img_Train.txt'),
                         os.path.join(data_path, 'Box_Train.txt'),
                         config['pairing_num'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              num_workers=0, shuffle=True, drop_last=True)

    val_dataset = PIAD('val', setting,
                       os.path.join(data_path, 'Point_Test.txt'),
                       os.path.join(data_path, 'Img_Test.txt'),
                       os.path.join(data_path, 'Box_Test.txt'))
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                            num_workers=0, shuffle=True)

    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")

    # 初始化模型
    if use_text_emb:
        model = get_IAG_TextEmb(
            pre_train=False,
            N_p=config['N_p'], emb_dim=config['emb_dim'],
            proj_dim=config['proj_dim'], num_heads=config['num_heads'],
            N_raw=config['N_raw'], num_affordance=config['num_affordance'],
            text_dim=config['text_dim']
        )
    else:
        model = get_MyNet(
            pre_train=False,
            N_p=config['N_p'], emb_dim=config['emb_dim'],
            proj_dim=config['proj_dim'], num_heads=config['num_heads'],
            N_raw=config['N_raw'], num_affordance=config['num_affordance']
        )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # 损失函数
    criterion_hm = HM_Loss().to(device)
    criterion_ce = nn.CrossEntropyLoss().to(device)

    # 优化器和调度器
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['lr'],
        betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['Epoch'], eta_min=1e-6
    )

    # 恢复训练
    start_epoch = 0
    best_auc = 0.0
    history = {
        'train_loss': [], 'val_auc': [], 'val_iou': [],
        'val_sim': [], 'val_mae': [], 'epoch_time': []
    }

    if resume_path and os.path.exists(resume_path):
        print(f"  Resuming from: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_auc = checkpoint.get('best_auc', 0)
        history = checkpoint.get('history', history)

    # 训练循环
    model_name = f"{model_type}-{setting}"
    ckpt_dir = os.path.join(CKPT_DIR, model_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\n  Starting training: {model_name}")
    print(f"  Epochs: {config['Epoch']}, LR: {config['lr']}, Batch: {config['batch_size']}")

    for epoch in range(start_epoch, config['Epoch']):
        epoch_start = time.time()

        # 训练
        if use_text_emb:
            train_loss, train_time = train_one_epoch_textemb(
                model, train_loader, criterion_hm, criterion_ce,
                optimizer, device, config, affordance_emb_tensor
            )
        else:
            train_loss, train_time = train_one_epoch_iag(
                model, train_loader, criterion_hm, criterion_ce,
                optimizer, device, config
            )

        # 评估
        eval_results = evaluate_model(
            model, val_loader, criterion_hm, device, config,
            affordance_emb_tensor=affordance_emb_tensor if use_text_emb else None,
            use_text_emb=use_text_emb
        )

        scheduler.step()
        epoch_time = time.time() - epoch_start

        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_auc'].append(eval_results['AUC'])
        history['val_iou'].append(eval_results['IOU'])
        history['val_sim'].append(eval_results['SIM'])
        history['val_mae'].append(eval_results['MAE'])
        history['epoch_time'].append(epoch_time)

        print(f"  Epoch {epoch+1}/{config['Epoch']} | "
              f"Loss: {train_loss:.4f} | "
              f"AUC: {eval_results['AUC']:.4f} | "
              f"IOU: {eval_results['IOU']:.4f} | "
              f"SIM: {eval_results['SIM']:.4f} | "
              f"MAE: {eval_results['MAE']:.4f} | "
              f"Time: {epoch_time:.1f}s")

        # 保存最佳模型
        if eval_results['AUC'] > best_auc:
            best_auc = eval_results['AUC']
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'history': history,
                'best_auc': best_auc,
                'config': config,
                'model_type': model_type,
                'setting': setting,
            }
            torch.save(checkpoint, os.path.join(ckpt_dir, 'best.pt'))

    # 最终评估
    # 加载最佳模型
    best_ckpt_path = os.path.join(ckpt_dir, 'best.pt')
    if os.path.exists(best_ckpt_path):
        checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])

    final_results = evaluate_model(
        model, val_loader, criterion_hm, device, config,
        affordance_emb_tensor=affordance_emb_tensor if use_text_emb else None,
        use_text_emb=use_text_emb
    )

    print(f"\n  [{model_name}] Final Results:")
    print(f"    AUC: {final_results['AUC']:.4f}")
    print(f"    IOU: {final_results['IOU']:.4f}")
    print(f"    SIM: {final_results['SIM']:.4f}")
    print(f"    MAE: {final_results['MAE']:.4f}")
    print(f"    Throughput: {final_results['throughput']:.2f} samples/s")

    return {
        'model_type': model_type,
        'setting': setting,
        'history': history,
        'final_results': final_results,
        'total_params': total_params,
    }


def print_comparison_table(results_list):
    """打印对比结果表格"""
    print("\n" + "=" * 90)
    print("IAG vs IAG_TextEmb Performance Comparison")
    print("=" * 90)

    header = f"{'Model':<20} {'Setting':<10} {'AUC':>8} {'IOU':>8} {'SIM':>8} {'MAE':>8} {'Throughput':>12} {'Params':>12}"
    print(header)
    print("-" * 90)

    for r in results_list:
        fr = r['final_results']
        line = (f"{r['model_type']:<20} {r['setting']:<10} "
                f"{fr['AUC']:>8.4f} {fr['IOU']:>8.4f} {fr['SIM']:>8.4f} "
                f"{fr['MAE']:>8.4f} {fr['throughput']:>10.2f}/s {r['total_params']:>10,}")
        print(line)

    print("=" * 90)

    # 保存结果到文件
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_path = os.path.join(RESULTS_DIR, 'iag_vs_textemb.txt')
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("IAG vs IAG_TextEmb Performance Comparison\n")
        f.write("=" * 90 + "\n")
        f.write(header + "\n")
        f.write("-" * 90 + "\n")
        for r in results_list:
            fr = r['final_results']
            line = (f"{r['model_type']:<20} {r['setting']:<10} "
                    f"{fr['AUC']:>8.4f} {fr['IOU']:>8.4f} {fr['SIM']:>8.4f} "
                    f"{fr['MAE']:>8.4f} {fr['throughput']:>10.2f}/s {r['total_params']:>10,}")
            f.write(line + "\n")
        f.write("=" * 90 + "\n")
    print(f"\nResults saved to: {result_path}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='IAG vs IAG_TextEmb Training Comparison')
    parser.add_argument('--setting', type=str, default='Seen', choices=['Seen', 'Unseen', 'both'],
                        help='Dataset setting')
    parser.add_argument('--data_dir', type=str, default=os.path.join(PROJECT_ROOT, 'Data'),
                        help='Data directory')
    parser.add_argument('--epochs', type=int, default=80, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--glove_path', type=str, default=None,
                        help='Path to GloVe 300d embeddings file')
    parser.add_argument('--text_dim', type=int, default=300, help='Text embedding dimension')
    parser.add_argument('--eval_only', action='store_true', help='Evaluation only mode')
    parser.add_argument('--iag_ckpt', type=str, default=None,
                        help='IAG checkpoint for eval_only mode')
    parser.add_argument('--textemb_ckpt', type=str, default=None,
                        help='IAG_TextEmb checkpoint for eval_only mode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    set_seed(args.seed)

    config = DEFAULT_CONFIG.copy()
    config['Epoch'] = args.epochs
    config['batch_size'] = args.batch_size
    config['lr'] = args.lr
    config['text_dim'] = args.text_dim

    device = get_device()

    # 准备文本嵌入
    affordance_emb_tensor = None
    if not args.eval_only:
        emb_matrix = prepare_text_embeddings(config, args.glove_path)
        affordance_emb_tensor = torch.tensor(emb_matrix, dtype=torch.float32).to(device)

    settings = ['Seen', 'Unseen'] if args.setting == 'both' else [args.setting]
    all_results = []

    for setting in settings:
        print(f"\n{'#' * 90}")
        print(f"# Setting: {setting}")
        print(f"{'#' * 90}")

        # ---- IAG ----
        print(f"\n{'=' * 60}")
        print(f"Training IAG on {setting}")
        print(f"{'=' * 60}")
        if args.eval_only and args.iag_ckpt:
            iag_result = train_and_evaluate(
                'IAG', setting, config, args.data_dir, device,
                resume_path=args.iag_ckpt
            )
        elif not args.eval_only:
            iag_result = train_and_evaluate(
                'IAG', setting, config, args.data_dir, device
            )
        else:
            print("  [Skip] No IAG checkpoint provided for eval_only mode")
            iag_result = None

        # ---- IAG_TextEmb ----
        print(f"\n{'=' * 60}")
        print(f"Training IAG_TextEmb on {setting}")
        print(f"{'=' * 60}")
        if args.eval_only and args.textemb_ckpt:
            textemb_result = train_and_evaluate(
                'IAG_TextEmb', setting, config, args.data_dir, device,
                affordance_emb_tensor=affordance_emb_tensor,
                resume_path=args.textemb_ckpt
            )
        elif not args.eval_only:
            textemb_result = train_and_evaluate(
                'IAG_TextEmb', setting, config, args.data_dir, device,
                affordance_emb_tensor=affordance_emb_tensor,
                glove_path=args.glove_path
            )
        else:
            print("  [Skip] No IAG_TextEmb checkpoint provided for eval_only mode")
            textemb_result = None

        if iag_result:
            all_results.append(iag_result)
        if textemb_result:
            all_results.append(textemb_result)

    # 打印对比结果
    if all_results:
        print_comparison_table(all_results)
    else:
        print("\nNo results to compare. Please provide checkpoints for eval_only mode.")


if __name__ == '__main__':
    main()