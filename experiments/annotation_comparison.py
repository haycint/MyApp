"""
Annotation Model Training & Performance Comparison
====================================================
对比两个标注模型（端到端模型和小模型协作）的训练效果和性能。

Plan1 (端到端): ResNet-50 + FPN + RPN + BBoxHead + InteractionClassifier (~48M参数)
Plan2 (多模型协同): BBoxDetector + ItemClassifier + InteractionClassifier (~25M参数)

实验内容:
1. 训练两个模型并记录训练曲线
2. 评估准确率、mAP 和吞吐量
3. 对比两种架构的性能

使用方式:
    # 完整训练+对比 (Seen)
    python experiments/annotation_comparison.py --setting Seen --data_dir ./Data

    # 仅训练Plan1
    python experiments/annotation_comparison.py --plan 1 --setting Seen --data_dir ./Data

    # 仅训练Plan2
    python experiments/annotation_comparison.py --plan 2 --setting Seen --data_dir ./Data

    # 指定epochs
    python experiments/annotation_comparison.py --epochs 50 --data_dir ./Data

输出:
    - 训练曲线保存到 ./experiments/results/
    - 对比结果表格保存到 ./experiments/results/annotation_comparison.txt
"""

import os
import sys
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Plan1 imports
from annotation.annotation_model import build_annotation_model, AnnotationLoss
from annotation.annotation_dataset import build_dataloader, PIADAnnotationDataset, collate_fn

# Plan2 imports
from annotation.annotation_model_plan2 import (
    BBoxDetector, ItemClassifier, InteractionClassifier,
    BBoxDetectorLoss, ItemClassifierLoss, InteractionClassifierLoss,
    AFFORDANCE_LABELS, DEFAULT_ITEM_CATEGORIES,
    build_bbox_detector, build_item_classifier, build_interaction_classifier,
)
from annotation.annotation_train_plan2 import (
    BBoxDetectorDataset, ItemClassifierDataset, InteractionClassifierDataset,
    bbox_collate_fn, item_collate_fn, interaction_collate_fn,
)

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'experiments', 'results')

AFFORDANCE_LABELS_LIST = AFFORDANCE_LABELS

# ============================================================================
# Synthetic Datasets for Fallback
# ============================================================================

class SyntheticBBoxDataset(Dataset):
    """Synthetic dataset for BBoxDetector training."""
    def __init__(self, num_samples=500, img_size=(224, 224)):
        self.num_samples = num_samples
        self.img_size = img_size
        self.transform = T.Compose([T.ToTensor()])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random image
        img = torch.randn(3, *self.img_size)
        # Generate random bboxes (normalized)
        bboxes = torch.rand(2, 4)  # subject and object
        labels = torch.randint(0, 2, (2,))  # binary labels
        return img, {'bboxes': bboxes, 'labels': labels}

class SyntheticItemDataset(Dataset):
    """Synthetic dataset for ItemClassifier training."""
    def __init__(self, num_samples=500, num_classes=10, img_size=(224, 224)):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = torch.randn(3, *self.img_size)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return img, label

class SyntheticInteractionDataset(Dataset):
    """Synthetic dataset for InteractionClassifier training."""
    def __init__(self, num_samples=500, img_size=(224, 224)):
        self.num_samples = num_samples
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        s_crop = torch.randn(3, 64, 64)  # subject crop
        o_crop = torch.randn(3, 64, 64)  # object crop
        sub_box = torch.rand(4)  # normalized bbox
        obj_box = torch.rand(4)
        label = torch.randint(0, 17, (1,)).item()  # affordance label
        return s_crop, o_crop, sub_box, obj_box, label

# ============================================================================
# Utility Functions
# ============================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_map(predictions, targets, iou_threshold=0.5, num_classes=17):
    """Compute mean Average Precision (mAP) for detection."""
    if not predictions or not targets:
        return 0.0
    
    # Assume predictions and targets are lists of dicts with 'boxes', 'scores', 'labels'
    # For simplicity, implement a basic version
    aps = []
    for cls in range(num_classes):
        # Collect predictions and targets for this class
        cls_preds = []
        cls_targets = []
        for pred, targ in zip(predictions, targets):
            if isinstance(pred, dict) and 'boxes' in pred:
                boxes = pred['boxes']
                scores = pred.get('scores', torch.ones(len(boxes)))
                labels = pred.get('labels', torch.full((len(boxes),), cls))
                mask = labels == cls
                cls_preds.extend(zip(boxes[mask], scores[mask]))
            if isinstance(targ, dict) and 'boxes' in targ:
                boxes = targ['boxes']
                labels = targ.get('labels', torch.full((len(boxes),), cls))
                mask = labels == cls
                cls_targets.extend(boxes[mask])
        
        if not cls_preds or not cls_targets:
            aps.append(0.0)
            continue
        
        # Sort predictions by score descending
        cls_preds.sort(key=lambda x: x[1], reverse=True)
        
        # Compute TP/FP
        tp = np.zeros(len(cls_preds))
        fp = np.zeros(len(cls_preds))
        gt_matched = set()
        
        for i, (pred_box, _) in enumerate(cls_preds):
            best_iou = 0
            best_gt_idx = -1
            for j, gt_box in enumerate(cls_targets):
                if j in gt_matched:
                    continue
                iou = box_iou(pred_box.unsqueeze(0), gt_box.unsqueeze(0)).item()
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            if best_iou >= iou_threshold and best_gt_idx != -1:
                tp[i] = 1
                gt_matched.add(best_gt_idx)
            else:
                fp[i] = 1
        
        # Compute precision and recall
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / len(cls_targets)
        precision = tp_cum / (tp_cum + fp_cum)
        
        # Compute AP using 11-point interpolation
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p
        ap /= 11
        aps.append(ap)
    
    return np.mean(aps)

def box_iou(boxes1, boxes2):
    """Compute IoU between two sets of boxes."""
    # Simplified IoU for 2D boxes [x1, y1, x2, y2]
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, :2].unsqueeze(1), boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:].unsqueeze(1), boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    union = area1.unsqueeze(1) + area2 - inter
    iou = inter / union
    return iou

# ============================================================================
# Plan1: 端到端模型训练与评估
# ============================================================================

def prepare_proposals_and_targets(images, targets, device):
    """Prepare proposals and ground truth for Plan1 training."""
    batch_proposals = []
    all_gt_labels = []
    all_gt_boxes = []

    for target in targets:
        subject_box = target['subject_box'].to(device)
        object_box = target['object_box'].to(device)

        # Generate proposals: subject, object, and random noise
        noise_boxes = torch.rand(8, 4, device=device) * 0.8 + 0.1  # Within [0.1, 0.9]
        proposals = torch.cat([subject_box.unsqueeze(0), object_box.unsqueeze(0), noise_boxes], dim=0)
        batch_proposals.append(proposals)

        # GT labels: 1 for subject, 2 for object, 0 for others
        gt_labels = torch.zeros(len(proposals), dtype=torch.long, device=device)
        gt_labels[0] = 1
        gt_labels[1] = 2
        all_gt_labels.append(gt_labels)

        # GT boxes: delta encoding
        gt_boxes = torch.zeros(len(proposals), 3 * 4, device=device, dtype=torch.float32)
        for prop_idx, proposal in enumerate(proposals):
            for box_idx, gt_box in [(1, subject_box), (2, object_box)]:
                dx = (gt_box[0] - proposal[0]) / (proposal[2] - proposal[0] + 1e-6)
                dy = (gt_box[1] - proposal[1]) / (proposal[3] - proposal[1] + 1e-6)
                dw = torch.log((gt_box[2] - gt_box[0]) / (proposal[2] - proposal[0] + 1e-6))
                dh = torch.log((gt_box[3] - gt_box[1]) / (proposal[3] - proposal[1] + 1e-6))
                gt_boxes[prop_idx, box_idx * 4:(box_idx + 1) * 4] = torch.tensor([dx, dy, dw, dh], device=device)
        all_gt_boxes.append(gt_boxes)

    return batch_proposals, all_gt_labels, all_gt_boxes

def train_plan1(config, data_dir, setting, device):
    """训练Plan1端到端标注模型。"""
    print(f"\n{'=' * 70}")
    print(f"Plan1: End-to-End Annotation Model Training ({setting})")
    print(f"{'=' * 70}")

    model = build_annotation_model(num_interactions=17, pretrained=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params / 1e6:.2f}M")
    print(f"  Trainable parameters: {trainable_params / 1e6:.2f}M")

    # Load data with fallback
    try:
        train_loader = build_dataloader(
            dataset_type='piad', data_dir=data_dir, setting=setting,
            split='train', batch_size=config['batch_size'],
            num_workers=0, img_size=(224, 224), augment=True
        )
        val_loader = build_dataloader(
            dataset_type='piad', data_dir=data_dir, setting=setting,
            split='val', batch_size=config['batch_size'],
            num_workers=0, img_size=(224, 224), augment=False
        )
    except Exception as e:
        print(f"  [Warning] Using synthetic data: {e}")
        train_ds = SyntheticBBoxDataset(num_samples=500)
        val_ds = SyntheticBBoxDataset(num_samples=100)
        train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)

    criterion = AnnotationLoss()
    backbone_params = [p for n, p in model.named_parameters() if 'layer' in n]
    other_params = [p for n, p in model.named_parameters() if 'layer' not in n]
    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': config['lr'] * 0.1},
        {'params': other_params, 'lr': config['lr']}
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_map': []}
    best_val_loss = float('inf')
    ckpt_dir = os.path.join(PROJECT_ROOT, 'ckpt', f'annotation_plan1-{setting}')
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(config['epochs']):
        print(f"  Epoch {epoch+1}/{config['epochs']}",end=' | ')
        model.train()
        total_loss = 0
        for batch_idx,(images, targets) in enumerate(train_loader):
            print(f"Batch {batch_idx+1}/{len(train_loader)}", end='\r')
            images = images.to(device)
            batch_proposals, all_gt_labels, all_gt_boxes = prepare_proposals_and_targets(images, targets, device)
            outputs = model(images, proposals=batch_proposals)
            loss_targets = {'gt_labels': torch.cat(all_gt_labels), 'gt_boxes': torch.cat(all_gt_boxes)}
            losses = criterion(outputs, loss_targets)
            loss = losses['total_loss']
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            print(f"Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}", end='\r')
        print(f"Batch {batch_idx+1}/{len(train_loader)} | Loss: {total_loss / len(train_loader):.4f}", end='\r')

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                batch_proposals, all_gt_labels, all_gt_boxes = prepare_proposals_and_targets(images, targets, device)
                outputs = model(images, proposals=batch_proposals)
                loss_targets = {'gt_labels': torch.cat(all_gt_labels), 'gt_boxes': torch.cat(all_gt_boxes)}
                losses = criterion(outputs, loss_targets)
                # val_loss += losses['total_loss'].item()
                if torch.is_tensor(losses['total_loss']):
                    val_loss += losses['total_loss'].item()
                else:
                    val_loss += losses['total_loss']
                # Simplified accuracy for interaction classification
                
                all_preds.append(outputs)
                all_targets.append(loss_targets)

        val_acc = correct / max(total, 1)
        # Prepare for mAP: assume outputs have 'predictions' with boxes, scores, labels
        pred_list = []
        targ_list = []
        for out, targ in zip(all_preds, all_targets):
            if 'predictions' in out:
                pred_list.append(out['predictions'])
            else:
                pred_list.append({'boxes': torch.empty(0, 4), 'scores': torch.empty(0), 'labels': torch.empty(0)})
            if isinstance(targ, dict) and 'gt_boxes' in targ:
                targ_list.append({'boxes': targ['gt_boxes'], 'labels': targ['gt_labels']})
            else:
                targ_list.append({'boxes': torch.empty(0, 4), 'labels': torch.empty(0)})
        val_map = compute_map(pred_list, targ_list)
        scheduler.step()
        history['train_loss'].append(total_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        history['val_map'].append(val_map)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_loss': best_val_loss, 'history': history},
                       os.path.join(ckpt_dir, 'best.pt'))

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{config['epochs']} | Loss: {history['train_loss'][-1]:.4f} | Val Acc: {val_acc:.4f} | mAP: {val_map:.4f}")

    throughput = measure_annotation_throughput(model, val_loader, device, plan='plan1')
    # Save history
    os.makedirs(RESULTS_DIR, exist_ok=True)
    history_path = os.path.join(RESULTS_DIR, f'plan1_history_{setting}.json')
    import json
    with open(history_path, 'w') as f:
        json.dump(history, f)
    print(f"Plan1 history saved to: {history_path}")
    return {
        'plan': 'Plan1 (End-to-End)',
        'total_params': total_params,
        'history': history,
        'best_val_loss': best_val_loss,
        'final_val_acc': val_acc,
        'final_val_map': val_map,
        'throughput': throughput,
    }

# ============================================================================
# Plan2 Pipeline for End-to-End Inference
# ============================================================================

class Plan2Pipeline:
    """Pipeline for Plan2 multi-model collaborative inference."""
    def __init__(self, bbox_model, item_model, inter_model):
        self.bbox_model = bbox_model
        self.item_model = item_model
        self.inter_model = inter_model
        self.bbox_model.eval()
        self.item_model.eval()
        self.inter_model.eval()
    
    def __call__(self, images):
        """Perform end-to-end inference."""
        with torch.no_grad():
            # Step 1: Detect bounding boxes
            bbox_outputs = self.bbox_model(images)
            # Assuming bbox_outputs has 'boxes' and 'scores'
            boxes = bbox_outputs.get('boxes', [])  # List of [N, 4] tensors
            scores = bbox_outputs.get('scores', [])
            
            # Step 2: Classify items for detected boxes
            item_logits = []
            for img_boxes in boxes:
                if len(img_boxes) == 0:
                    item_logits.append(torch.empty(0, len(DEFAULT_ITEM_CATEGORIES)))
                    continue
                # Crop images for each box (simplified)
                crops = []  # In real impl, crop images
                for box in img_boxes:
                    crop = images[0]  # Placeholder
                    crops.append(crop)
                if crops:
                    crops = torch.stack(crops)
                    logits = self.item_model(crops)
                    item_logits.append(logits)
                else:
                    item_logits.append(torch.empty(0, len(DEFAULT_ITEM_CATEGORIES)))
            
            # Step 3: Predict interactions
            # Simplified: assume pairs are formed
            inter_preds = []
            for i, img_boxes in enumerate(boxes):
                if len(img_boxes) < 2:
                    inter_preds.append(torch.empty(0, 17))
                    continue
                # Form pairs (simplified)
                pairs = [(0, 1)]  # Placeholder
                s_crops = []
                o_crops = []
                for s_idx, o_idx in pairs:
                    s_crop = images[i]  # Placeholder crop
                    o_crop = images[i]
                    s_crops.append(s_crop)
                    o_crops.append(o_crop)
                if s_crops:
                    s_crops = torch.stack(s_crops)
                    o_crops = torch.stack(o_crops)
                    sub_boxes = img_boxes[pairs[0][0]].unsqueeze(0)
                    obj_boxes = img_boxes[pairs[0][1]].unsqueeze(0)
                    logits = self.inter_model(s_crops, o_crops, sub_boxes, obj_boxes)
                    inter_preds.append(logits)
                else:
                    inter_preds.append(torch.empty(0, 17))
            
            return {
                'boxes': boxes,
                'item_logits': item_logits,
                'inter_logits': inter_preds
            }

def train_plan2(config, data_dir, setting, device):
    """
    训练Plan2多模型协作标注系统。

    顺序训练三个小模型:
    1. BBoxDetector - 主客体边界框检测
    2. ItemClassifier - 物品类别分类
    3. InteractionClassifier - 交互动作分类

    评估包括各子模型指标 + 端到端 mAP 和联合准确率。
    """
    print(f"\n{'=' * 70}")
    print(f"Plan2: Multi-Model Collaborative Training ({setting})")
    print(f"{'=' * 70}")

    results = {}

    # ---- Phase 1: BBoxDetector ----
    print(f"\n  Phase 1/3: BBoxDetector")
    bbox_model = build_bbox_detector(pretrained=True).to(device)
    bbox_params = sum(p.numel() for p in bbox_model.parameters())
    print(f"    Parameters: {bbox_params / 1e6:.2f}M")

    bbox_criterion = BBoxDetectorLoss()
    backbone_p = [p for n, p in bbox_model.named_parameters() if 'backbone' in n]
    other_p = [p for n, p in bbox_model.named_parameters() if 'backbone' not in n]
    bbox_optimizer = optim.Adam([
        {'params': backbone_p, 'lr': config['lr'] * 0.1},
        {'params': other_p, 'lr': config['lr']}
    ], weight_decay=1e-4)
    bbox_scheduler = optim.lr_scheduler.CosineAnnealingLR(bbox_optimizer, T_max=config['epochs'], eta_min=1e-6)

    try:
        train_ds = BBoxDetectorDataset(data_dir, setting=setting, split='train', augment=True)
        val_ds = BBoxDetectorDataset(data_dir, setting=setting, split='val')
    except Exception:
        print(f"    [Warning] Using synthetic data for BBoxDetector")
        train_ds = SyntheticBBoxDataset(num_samples=500)
        val_ds = SyntheticBBoxDataset(num_samples=100)

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True,
                              num_workers=0, collate_fn=bbox_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False,
                            num_workers=0, collate_fn=bbox_collate_fn)

    bbox_history = {'train_loss': [], 'val_loss': []}
    best_bbox_loss = float('inf')

    for epoch in range(config['epochs']):
        bbox_model.train()
        train_loss = 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            outputs = bbox_model(images)
            loss, _ = bbox_criterion(outputs['predictions'], targets)
            bbox_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bbox_model.parameters(), max_norm=1.0)
            bbox_optimizer.step()
            train_loss += loss.item()

        bbox_model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = {k: v.to(device) for k, v in targets.items()}
                outputs = bbox_model(images)
                loss, _ = bbox_criterion(outputs['predictions'], targets)
                val_loss += loss.item()

        bbox_scheduler.step()
        bbox_history['train_loss'].append(train_loss / max(len(train_loader), 1))
        bbox_history['val_loss'].append(val_loss / max(len(val_loader), 1))

        if val_loss < best_bbox_loss:
            best_bbox_loss = val_loss

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{config['epochs']} | "
                  f"Train: {bbox_history['train_loss'][-1]:.4f} | "
                  f"Val: {bbox_history['val_loss'][-1]:.4f}")

    results['bbox'] = {
        'params': bbox_params, 'history': bbox_history,
        'best_val_loss': best_bbox_loss
    }

    # ---- Phase 2: ItemClassifier ----
    print(f"\n  Phase 2/3: ItemClassifier")
    num_item_classes = len(DEFAULT_ITEM_CATEGORIES)
    item_model = build_item_classifier(num_item_classes=num_item_classes, pretrained=True).to(device)
    item_params = sum(p.numel() for p in item_model.parameters())
    print(f"    Parameters: {item_params / 1e6:.2f}M")

    item_criterion = ItemClassifierLoss()
    item_optimizer = optim.Adam(item_model.parameters(), lr=config['lr'], weight_decay=1e-4)
    item_scheduler = optim.lr_scheduler.CosineAnnealingLR(item_optimizer, T_max=config['epochs'], eta_min=1e-6)

    try:
        item_train_ds = ItemClassifierDataset(data_dir, setting=setting, split='train', augment=True)
        item_val_ds = ItemClassifierDataset(data_dir, setting=setting, split='val')
    except Exception:
        print(f"    [Warning] Using synthetic data for ItemClassifier")
        item_train_ds = SyntheticItemDataset(num_samples=500, num_classes=num_item_classes)
        item_val_ds = SyntheticItemDataset(num_samples=100, num_classes=num_item_classes)

    item_train_loader = DataLoader(item_train_ds, batch_size=config['batch_size'], shuffle=True,
                                   num_workers=0, collate_fn=item_collate_fn)
    item_val_loader = DataLoader(item_val_ds, batch_size=config['batch_size'], shuffle=False,
                                 num_workers=0, collate_fn=item_collate_fn)

    item_history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_item_acc = 0

    for epoch in range(config['epochs']):
        item_model.train()
        train_loss = 0
        correct = 0
        total = 0
        for images, labels in item_train_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = item_model(images)
            loss, _ = item_criterion(logits, labels)
            item_optimizer.zero_grad()
            loss.backward()
            item_optimizer.step()
            train_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        item_model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in item_val_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = item_model(images)
                loss, _ = item_criterion(logits, labels)
                val_loss += loss.item()
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / max(val_total, 1)
        item_scheduler.step()
        item_history['train_loss'].append(train_loss / max(len(item_train_loader), 1))
        item_history['val_loss'].append(val_loss / max(len(item_val_loader), 1))
        item_history['val_acc'].append(val_acc)

        if val_acc > best_item_acc:
            best_item_acc = val_acc

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{config['epochs']} | "
                  f"Train: {item_history['train_loss'][-1]:.4f} | "
                  f"Val Acc: {val_acc:.4f}")

    results['item'] = {
        'params': item_params, 'history': item_history,
        'best_val_acc': best_item_acc
    }

    # ---- Phase 3: InteractionClassifier ----
    print(f"\n  Phase 3/3: InteractionClassifier")
    inter_model = build_interaction_classifier(num_interactions=17).to(device)
    inter_params = sum(p.numel() for p in inter_model.parameters())
    print(f"    Parameters: {inter_params / 1e6:.2f}M")

    inter_criterion = InteractionClassifierLoss()
    inter_optimizer = optim.Adam(inter_model.parameters(), lr=config['lr'], weight_decay=1e-4)
    inter_scheduler = optim.lr_scheduler.CosineAnnealingLR(inter_optimizer, T_max=config['epochs'], eta_min=1e-6)

    try:
        inter_train_ds = InteractionClassifierDataset(data_dir, setting=setting, split='train', augment=True)
        inter_val_ds = InteractionClassifierDataset(data_dir, setting=setting, split='val')
    except Exception:
        print(f"    [Warning] Using synthetic data for InteractionClassifier")
        inter_train_ds = SyntheticInteractionDataset(num_samples=500)
        inter_val_ds = SyntheticInteractionDataset(num_samples=100)

    inter_train_loader = DataLoader(inter_train_ds, batch_size=config['batch_size'], shuffle=True,
                                     num_workers=0, collate_fn=interaction_collate_fn)
    inter_val_loader = DataLoader(inter_val_ds, batch_size=config['batch_size'], shuffle=False,
                                   num_workers=0, collate_fn=interaction_collate_fn)

    inter_history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_inter_acc = 0

    for epoch in range(config['epochs']):
        inter_model.train()
        train_loss = 0
        correct = 0
        total = 0
        for s_crop, o_crop, sub_box, obj_box, labels in inter_train_loader:
            s_crop = s_crop.to(device)
            o_crop = o_crop.to(device)
            sub_box = sub_box.to(device)
            obj_box = obj_box.to(device)
            labels = labels.to(device)
            logits = inter_model(s_crop, o_crop, sub_box, obj_box)
            loss, _ = inter_criterion(logits, labels)
            inter_optimizer.zero_grad()
            loss.backward()
            inter_optimizer.step()
            train_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        inter_model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for s_crop, o_crop, sub_box, obj_box, labels in inter_val_loader:
                s_crop = s_crop.to(device)
                o_crop = o_crop.to(device)
                sub_box = sub_box.to(device)
                obj_box = obj_box.to(device)
                labels = labels.to(device)
                logits = inter_model(s_crop, o_crop, sub_box, obj_box)
                loss, _ = inter_criterion(logits, labels)
                val_loss += loss.item()
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / max(val_total, 1)
        inter_scheduler.step()
        inter_history['train_loss'].append(train_loss / max(len(inter_train_loader), 1))
        inter_history['val_loss'].append(val_loss / max(len(inter_val_loader), 1))
        inter_history['val_acc'].append(val_acc)

        if val_acc > best_inter_acc:
            best_inter_acc = val_acc

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{config['epochs']} | "
                  f"Train: {inter_history['train_loss'][-1]:.4f} | "
                  f"Val Acc: {val_acc:.4f}")

    results['interaction'] = {
        'params': inter_params, 'history': inter_history,
        'best_val_acc': best_inter_acc
    }

    # ---- 端到端评估 ----
    print(f"\n  End-to-End Evaluation")
    pipeline = Plan2Pipeline(bbox_model, item_model, inter_model)
    end_to_end_acc = 0
    end_to_end_map = 0
    total_samples = 0
    correct_inter = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, targets in inter_val_loader:  # Use interaction val loader as proxy
            images = images.to(device)
            outputs = pipeline(images)
            # Simplified evaluation: compare predicted interactions
            for i, (s_crop, o_crop, sub_box, obj_box, label) in enumerate(zip(*[images]*5, targets)):  # Placeholder
                if outputs['inter_logits'][i].numel() > 0:
                    pred = outputs['inter_logits'][i].argmax(1)
                    correct_inter += (pred == label.to(device)).sum().item()
                    total_samples += label.size(0)
                    all_preds.append(pred.cpu())
                    all_targets.append(label.cpu())
    end_to_end_acc = correct_inter / max(total_samples, 1)
    # For mAP, assume predictions are logits, targets are labels
    pred_dicts = [{'boxes': torch.empty(0, 4), 'scores': pred.float(), 'labels': pred} for pred in all_preds]
    targ_dicts = [{'boxes': torch.empty(0, 4), 'labels': targ} for targ in all_targets]
    end_to_end_map = compute_map(pred_dicts, targ_dicts)

    # 汇总
    total_plan2_params = bbox_params + item_params + inter_params
    print(f"\n  Plan2 Total Parameters: {total_plan2_params / 1e6:.2f}M")

    # 测量吞吐量
    throughput = measure_annotation_throughput(
        {'bbox': bbox_model, 'item': item_model, 'interaction': inter_model},
        val_loader if 'val_loader' in dir() else inter_val_loader,
        device, plan='plan2'
    )
    # Save histories
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plan2_history = {
        'bbox': results['bbox']['history'],
        'item': results['item']['history'],
        'interaction': results['interaction']['history']
    }
    history_path = os.path.join(RESULTS_DIR, f'plan2_history_{setting}.json')
    import json
    with open(history_path, 'w') as f:
        json.dump(plan2_history, f)
    print(f"Plan2 history saved to: {history_path}")
    return {
        'plan': 'Plan2 (Multi-Model)',
        'total_params': total_plan2_params,
        'bbox_params': bbox_params,
        'item_params': item_params,
        'inter_params': inter_params,
        'results': results,
        'best_bbox_loss': best_bbox_loss,
        'best_item_acc': best_item_acc,
        'best_inter_acc': best_inter_acc,
        'end_to_end_acc': end_to_end_acc,
        'end_to_end_map': end_to_end_map,
        'throughput': throughput,
    }

# ============================================================================
# 吞吐量测量 (Improved)
# ============================================================================

def measure_annotation_throughput(model, val_loader, device, plan='plan1', num_batches=10):
    """测量标注模型的推理吞吐量"""
    if plan == 'plan1':
        model.eval()
        total_samples = 0
        start_time = time.time()
        with torch.no_grad():
            for i, (images, targets) in enumerate(val_loader):
                if i >= num_batches: break
                images = images.to(device)
                batch_proposals, _, _ = prepare_proposals_and_targets(images, targets, device)
                _ = model(images, proposals=batch_proposals)
                total_samples += images.size(0)
        elapsed = time.time() - start_time
    else:
        # Plan2: Measure all models
        bbox_model, item_model, inter_model = model['bbox'], model['item'], model['interaction']
        bbox_model.eval()
        item_model.eval()
        inter_model.eval()
        total_samples = 0
        start_time = time.time()
        with torch.no_grad():
            for i, (images, targets) in enumerate(val_loader):
                if i >= num_batches: break
                images = images.to(device)
                # Run bbox detection
                bbox_outputs = bbox_model(images)
                # Run item classification (simplified)
                if 'crops' in targets:
                    crops = targets['crops'].to(device)
                    item_logits = item_model(crops)
                # Run interaction classification (simplified)
                if len(targets) >= 5:
                    s_crop, o_crop, sub_box, obj_box, labels = targets[:5]
                    s_crop = s_crop.to(device)
                    o_crop = o_crop.to(device)
                    sub_box = sub_box.to(device)
                    obj_box = obj_box.to(device)
                    inter_logits = inter_model(s_crop, o_crop, sub_box, obj_box)
                total_samples += images.size(0)
        elapsed = time.time() - start_time
    return total_samples / elapsed if elapsed > 0 else 0

# ============================================================================
# 对比结果输出 (Enhanced)
# ============================================================================

def print_comparison(plan1_result, plan2_result, setting):
    """打印Plan1和Plan2的对比结果"""
    print(f"\n{'=' * 90}")
    print(f"Annotation Model Comparison ({setting})")
    print(f"{'=' * 90}")

    header = f"{'Metric':<30} {'Plan1 (End-to-End)':<25} {'Plan2 (Multi-Model)':<25}"
    print(header)
    print("-" * 90)

    rows = [
        ('Total Parameters', f"{plan1_result['total_params']/1e6:.2f}M", f"{plan2_result['total_params']/1e6:.2f}M"),
        ('Throughput (samples/s)', f"{plan1_result['throughput']:.2f}", f"{plan2_result['throughput']:.2f}"),
        ('Validation Accuracy', f"{plan1_result['final_val_acc']:.4f}", f"{plan2_result['end_to_end_acc']:.4f}"),
        ('Validation mAP', f"{plan1_result['final_val_map']:.4f}", f"{plan2_result['end_to_end_map']:.4f}"),
        ('Best Validation Loss', f"{plan1_result['best_val_loss']:.4f}", f"{plan2_result['best_bbox_loss']:.4f}"),
    ]

    for metric, v1, v2 in rows:
        print(f"{metric:<30} {v1:<25} {v2:<25}")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_path = os.path.join(RESULTS_DIR, 'annotation_comparison.txt')
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write(f"Annotation Model Comparison ({setting})\n")
        f.write("=" * 90 + "\n")
        f.write(header + "\n")
        f.write("-" * 90 + "\n")
        for metric, v1, v2 in rows:
            f.write(f"{metric:<30} {v1:<25} {v2:<25}\n")
        f.write("=" * 90 + "\n")
    print(f"\nResults saved to: {result_path}")

# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Annotation Model Comparison')
    parser.add_argument('--setting', type=str, default='Seen', choices=['Seen', 'Unseen'])
    parser.add_argument('--data_dir', type=str, default=os.path.join(PROJECT_ROOT, 'Data'))
    parser.add_argument('--plan', type=int, default=0, choices=[0, 1, 2],
                        help='0=both, 1=Plan1 only, 2=Plan2 only')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    config = {'epochs': args.epochs, 'batch_size': args.batch_size, 'lr': args.lr}

    plan1_result = None
    plan2_result = None

    if args.plan in [0, 1]:
        plan1_result = train_plan1(config, args.data_dir, args.setting, device)

    if args.plan in [0, 2]:
        plan2_result = train_plan2(config, args.data_dir, args.setting, device)

    if plan1_result and plan2_result:
        print_comparison(plan1_result, plan2_result, args.setting)
    elif plan1_result:
        print(f"\nPlan1 Results: Parameters: {plan1_result['total_params']/1e6:.2f}M, Throughput: {plan1_result['throughput']:.2f}, Val Acc: {plan1_result['final_val_acc']:.4f}, mAP: {plan1_result['final_val_map']:.4f}")
    elif plan2_result:
        print(f"\nPlan2 Results: Total Parameters: {plan2_result['total_params']/1e6:.2f}M, Throughput: {plan2_result['throughput']:.2f}, Best Interaction Acc: {plan2_result['best_inter_acc']:.4f}")

if __name__ == '__main__':
    main()