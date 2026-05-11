"""
Training Script for Annotation Model
标注模型训练脚本

功能:
- 训练主体/客体检测
- 训练交互动作分类
- 支持两种模式：
  1. 原始模型训练（随机初始化权重）
  2. ImageNet 预训练权重训练（自动下载或从本地文件加载）
- 使用 PIAD Seen 数据集
- 支持断点续训
- 训练曲线可视化
- 模型保存与加载

用法:
  # 原始模型训练（随机初始化，无预训练权重）
  python train_annotation.py --mode scratch --data_dir ./Data --setting Seen

  # 使用 ImageNet 预训练权重（自动下载）
  python train_annotation.py --mode imagenet --data_dir ./Data --setting Seen

  # 使用本地 ImageNet 预训练权重文件
  python train_annotation.py --mode imagenet --img_model_path ./ckpt/resnet50.pth --data_dir ./Data --setting Seen

  # 使用配置文件
  python train_annotation.py --config config_annotation.yaml

  # 断点续训
  python train_annotation.py --mode imagenet --resume ./checkpoints/annotation/best.pt
"""

import os
import sys
import random
import yaml
import argparse
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from annotation_model import build_annotation_model, AnnotationLoss
from annotation_dataset import build_dataloader, AFFORDANCE_LABELS

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AnnotationTrainer:
    """
    标注模型训练器

    支持两种训练模式：
    1. scratch: 从随机初始化的权重开始训练
    2. imagenet: 使用 ImageNet 预训练的 ResNet-50 作为 backbone
    """

    def __init__(self, config):
        """
        Args:
            config: 配置字典
        """
        self.config = config
        self.device = torch.device(
            config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )

        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_cls_loss': [],
            'train_bbox_loss': [],
            'train_interaction_loss': [],
            'val_cls_acc': [],
            'val_interaction_acc': [],
            'lr': []
        }

        # 创建保存目录
        self.save_dir = config.get('save_dir', './checkpoints/annotation')
        os.makedirs(self.save_dir, exist_ok=True)

        # 初始化组件
        self._init_model()
        self._init_dataloaders()
        self._init_loss()
        self._init_optimizer()

        # 日志文件
        log_name = datetime.now().strftime('%Y%m%d_%H%M%S') + '-annotation'
        self.log_file_path = os.path.join(self.save_dir, f"{log_name}.txt")

        print(f"Trainer initialized on {self.device}")
        print(f"Training mode: {config.get('mode', 'imagenet')}")
        print(f"Dataset setting: {config.get('setting', 'Seen')}")

    def _init_model(self):
        """初始化模型"""
        mode = self.config.get('mode', 'imagenet')

        if mode == 'imagenet':
            # ImageNet 预训练模式
            img_model_path = self.config.get('img_model_path', None)
            if img_model_path and os.path.exists(img_model_path):
                # 从本地文件加载 ImageNet 预训练权重
                print(f"[Model] Loading ImageNet pretrained weights from: {img_model_path}")
                self.model = build_annotation_model(
                    num_interactions=self.config.get('num_interactions', 17),
                    pretrained=False,  # 不使用 torchvision 自动下载
                    img_model_path=img_model_path
                )
            else:
                # 使用 torchvision 自动下载 ImageNet 预训练权重
                print("[Model] Using ImageNet pretrained weights (auto-download via torchvision)")
                self.model = build_annotation_model(
                    num_interactions=self.config.get('num_interactions', 17),
                    pretrained=True,
                    img_model_path=None
                )
        else:
            # 原始模式：随机初始化
            print("[Model] Using randomly initialized weights (no pretrained)")
            self.model = build_annotation_model(
                num_interactions=self.config.get('num_interactions', 17),
                pretrained=False,
                img_model_path=None
            )

        self.model = self.model.to(self.device)

        # 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters()
                               if p.requires_grad)
        print(f"Total parameters: {total_params / 1e6:.2f}M")
        print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    def _init_dataloaders(self):
        """初始化数据加载器"""
        dataset_type = self.config.get('dataset_type', 'piad')
        data_dir = self.config.get('data_dir')
        setting = self.config.get('setting', 'Seen')

        print(f"[Data] Dataset type: {dataset_type}, Setting: {setting}")
        print(f"[Data] Data directory: {data_dir}")

        self.train_loader = build_dataloader(
            dataset_type=dataset_type,
            data_dir=data_dir,
            setting=setting,
            split='train',
            batch_size=self.config.get('batch_size', 8),
            num_workers=self.config.get('num_workers', 0),
            img_size=tuple(self.config.get('img_size', [224, 224])),
            augment=True
        )

        self.val_loader = build_dataloader(
            dataset_type=dataset_type,
            data_dir=data_dir,
            setting=setting,
            split='test',
            batch_size=self.config.get('batch_size', 8),
            num_workers=self.config.get('num_workers', 0),
            img_size=tuple(self.config.get('img_size', [224, 224])),
            augment=False
        )

    def _init_loss(self):
        """初始化损失函数"""
        self.criterion = AnnotationLoss()

    def _init_optimizer(self):
        """初始化优化器"""
        mode = self.config.get('mode', 'imagenet')
        base_lr = self.config.get('lr', 1e-4)

        if mode == 'imagenet':
            # ImageNet 预训练模式：backbone 使用更小的学习率
            backbone_params = []
            other_params = []

            for name, param in self.model.named_parameters():
                if any(layer in name for layer in
                       ['layer1', 'layer2', 'layer3', 'layer4',
                        'conv1', 'bn1']):
                    backbone_params.append(param)
                else:
                    other_params.append(param)

            print(f"[Optimizer] Backbone params: {len(backbone_params)}, "
                  f"Other params: {len(other_params)}")
            print(f"[Optimizer] Backbone LR: {base_lr * 0.1}, "
                  f"Other LR: {base_lr}")

            self.optimizer = optim.Adam([
                {'params': backbone_params, 'lr': base_lr * 0.1},
                {'params': other_params, 'lr': base_lr}
            ], weight_decay=self.config.get('weight_decay', 1e-4))
        else:
            # 原始模式：所有参数使用相同学习率
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=base_lr,
                weight_decay=self.config.get('weight_decay', 1e-4)
            )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.get('epochs', 100),
            eta_min=1e-6
        )

    def _log(self, message):
        """打印并写入日志"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')

    def _prepare_proposals_and_targets(self, targets, num_noise_boxes=10):
        """
        准备训练/验证用的 proposals 和目标数据

        使用真实边界框作为正样本 proposals，
        添加噪声 boxes 作为负样本 proposals

        Args:
            targets: batch 中的目标列表
            num_noise_boxes: 每张图片添加的噪声负样本数量

        Returns:
            batch_proposals, all_gt_labels, all_gt_boxes, all_gt_interactions
        """
        batch_proposals = []
        all_gt_labels = []
        all_gt_boxes = []
        all_gt_interactions = []

        for i, target in enumerate(targets):
            # 使用真实边界框作为 proposals
            subject_box = target['subject_box'].to(self.device)
            object_box = target['object_box'].to(self.device)
            interaction = target['interaction'].to(self.device)

            # 添加噪声 boxes 作为负样本
            noise_boxes = torch.randn(num_noise_boxes, 4, device=self.device) * 50 + 100
            noise_boxes = torch.clamp(noise_boxes, 0, 224)

            proposals = torch.cat([
                subject_box.unsqueeze(0),
                object_box.unsqueeze(0),
                noise_boxes
            ], dim=0)
            batch_proposals.append(proposals)

            # 标签: 0=bg, 1=subject, 2=object
            gt_labels = torch.zeros(len(proposals), dtype=torch.long,
                                    device=self.device)
            gt_labels[0] = 1  # subject
            gt_labels[1] = 2  # object
            all_gt_labels.append(gt_labels)

            # 计算边界框回归目标
            # 形状 [num_proposals, num_classes * 4] = [N, 12]
            gt_boxes = torch.zeros(len(proposals), 3 * 4,
                                   device=self.device, dtype=torch.float32)

            gt_subject_box = target['subject_box'].to(self.device)
            gt_object_box = target['object_box'].to(self.device)

            for prop_idx, proposal in enumerate(proposals):
                prop_w = proposal[2] - proposal[0]
                prop_h = proposal[3] - proposal[1]
                prop_ctr_x = proposal[0] + 0.5 * prop_w
                prop_ctr_y = proposal[1] + 0.5 * prop_h

                # 计算与主体框的偏移
                gt_w = gt_subject_box[2] - gt_subject_box[0]
                gt_h = gt_subject_box[3] - gt_subject_box[1]
                gt_ctr_x = gt_subject_box[0] + 0.5 * gt_w
                gt_ctr_y = gt_subject_box[1] + 0.5 * gt_h

                dx_sub = (gt_ctr_x - prop_ctr_x) / (prop_w + 1e-6)
                dy_sub = (gt_ctr_y - prop_ctr_y) / (prop_h + 1e-6)
                dw_sub = torch.log(gt_w / (prop_w + 1e-6) + 1e-6)
                dh_sub = torch.log(gt_h / (prop_h + 1e-6) + 1e-6)
                gt_boxes[prop_idx, 1 * 4:2 * 4] = torch.tensor(
                    [dx_sub, dy_sub, dw_sub, dh_sub], device=self.device
                )

                # 计算与客体框的偏移
                gt_w = gt_object_box[2] - gt_object_box[0]
                gt_h = gt_object_box[3] - gt_object_box[1]
                gt_ctr_x = gt_object_box[0] + 0.5 * gt_w
                gt_ctr_y = gt_object_box[1] + 0.5 * gt_h

                dx_obj = (gt_ctr_x - prop_ctr_x) / (prop_w + 1e-6)
                dy_obj = (gt_ctr_y - prop_ctr_y) / (prop_h + 1e-6)
                dw_obj = torch.log(gt_w / (prop_w + 1e-6) + 1e-6)
                dh_obj = torch.log(gt_h / (prop_h + 1e-6) + 1e-6)
                gt_boxes[prop_idx, 2 * 4:3 * 4] = torch.tensor(
                    [dx_obj, dy_obj, dw_obj, dh_obj], device=self.device
                )

            all_gt_boxes.append(gt_boxes)

            # 交互类别目标（每张图片一个）
            all_gt_interactions.append(interaction)

        return batch_proposals, all_gt_labels, all_gt_boxes, all_gt_interactions

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        cls_loss_sum = 0
        bbox_loss_sum = 0
        interaction_loss_sum = 0
        num_batches = len(self.train_loader)

        if num_batches == 0:
            self._log("[Warning] Training dataloader is empty!")
            return 0.0

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)

            # 准备 proposals 和目标
            (batch_proposals, all_gt_labels,
             all_gt_boxes, all_gt_interactions) = \
                self._prepare_proposals_and_targets(targets)

            # 前向传播
            outputs = self.model(images, proposals=batch_proposals)

            # 准备损失目标
            loss_targets = {
                'gt_labels': torch.cat(all_gt_labels),
                'gt_boxes': torch.cat(all_gt_boxes),
                'gt_interactions': torch.stack(all_gt_interactions)
            }

            # 计算交互分类损失（训练模式下模型不直接输出interaction_logits，
            # 需要用检测到的subject和object特征计算）
            self._compute_interaction_loss(outputs, targets, loss_targets)

            # 计算损失
            losses = self.criterion(outputs, loss_targets)
            loss = losses['total_loss']

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # 记录损失
            total_loss += loss.item()
            if 'cls_loss' in losses:
                cls_loss_sum += losses['cls_loss'].item()
            if 'bbox_loss' in losses:
                bbox_loss_sum += losses['bbox_loss'].item()
            if 'interaction_loss' in losses:
                interaction_loss_sum += losses['interaction_loss'].item()

            if batch_idx % 10 == 0:
                self._log(
                    f"  Batch {batch_idx}/{num_batches} | "
                    f"Loss: {loss.item():.4f} | "
                    f"cls: {losses.get('cls_loss', torch.tensor(0)).item():.4f} | "
                    f"bbox: {losses.get('bbox_loss', torch.tensor(0)).item():.4f} | "
                    f"inter: {losses.get('interaction_loss', torch.tensor(0)).item():.4f}"
                )

        avg_loss = total_loss / num_batches
        avg_cls_loss = cls_loss_sum / num_batches
        avg_bbox_loss = bbox_loss_sum / num_batches
        avg_interaction_loss = interaction_loss_sum / num_batches

        self.history['train_loss'].append(avg_loss)
        self.history['train_cls_loss'].append(avg_cls_loss)
        self.history['train_bbox_loss'].append(avg_bbox_loss)
        self.history['train_interaction_loss'].append(avg_interaction_loss)

        return avg_loss

    def _compute_interaction_loss(self, outputs, targets, loss_targets):
        """
        计算交互分类损失

        训练模式下，使用 roi_features 和 proposals 中
        的 subject/object 位置来计算交互分类
        """
        if 'roi_features' not in outputs or 'proposals' not in outputs:
            return

        roi_features = outputs['roi_features']
        proposals = outputs['proposals']
        p3_feature = outputs.get('p3_feature', None)

        # 获取 ground truth 交互标签
        gt_interactions = loss_targets['gt_interactions']

        # 对每个样本，取 subject (index 0) 和 object (index 1) 的 ROI 特征
        all_logits = []
        all_labels = []

        offset = 0
        for batch_idx, props in enumerate(proposals):
            n_props = len(props)
            if n_props < 2:
                offset += n_props
                continue

            # subject 是第 0 个 proposal，object 是第 1 个
            s_feat = roi_features[offset + 0:offset + 1]  # [1, C, 7, 7]
            o_feat = roi_features[offset + 1:offset + 2]  # [1, C, 7, 7]

            # 计算空间关系
            s_box = props[0]
            o_box = props[1]

            s_center = (s_box[:2] + s_box[2:]) / 2
            o_center = (o_box[:2] + o_box[2:]) / 2
            s_size = s_box[2:] - s_box[:2]
            o_size = o_box[2:] - o_box[:2]

            spatial = torch.cat([
                (o_center - s_center) / (s_size + 1e-6),
                torch.log(o_size / (s_size + 1e-6) + 1e-6)
            ]).unsqueeze(0)

            # 交互分类
            logits = self.model.interaction_classifier(s_feat, o_feat, spatial)
            all_logits.append(logits)
            all_labels.append(gt_interactions[batch_idx])

            offset += n_props

        if all_logits:
            interaction_logits = torch.cat(all_logits, dim=0)
            interaction_labels = torch.stack(all_labels)
            loss_targets['gt_interactions'] = interaction_labels
            outputs['interaction_logits'] = interaction_logits

    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        total_cls_loss = 0
        total_bbox_loss = 0
        total_interaction_loss = 0
        correct_cls = 0
        correct_interaction = 0
        total_samples = 0
        num_batches = len(self.val_loader)

        if num_batches == 0:
            self._log("[Warning] Validation dataloader is empty!")
            return 0.0, 0.0, 0.0

        with torch.no_grad():
            for images, targets in enumerate(self.val_loader) if \
                    isinstance(self.val_loader, enumerate) else \
                    [(i, batch) for i, batch in enumerate(self.val_loader)]:
                images = images.to(self.device)

                # 准备验证用数据
                (batch_proposals, all_gt_labels,
                 all_gt_boxes, all_gt_interactions) = \
                    self._prepare_proposals_and_targets(targets)

                # 前向传播
                outputs = self.model(images, proposals=batch_proposals)

                # 准备损失目标
                loss_targets = {
                    'gt_labels': torch.cat(all_gt_labels),
                    'gt_boxes': torch.cat(all_gt_boxes),
                    'gt_interactions': torch.stack(all_gt_interactions)
                }

                # 计算交互分类损失
                self._compute_interaction_loss(outputs, targets, loss_targets)

                # 计算损失
                losses = self.criterion(outputs, loss_targets)

                # 累积损失
                for key, val in losses.items():
                    if isinstance(val, torch.Tensor):
                        losses[key] = val.item()

                total_loss += losses.get('total_loss', 0)
                total_cls_loss += losses.get('cls_loss', 0)
                total_bbox_loss += losses.get('bbox_loss', 0)
                total_interaction_loss += losses.get('interaction_loss', 0)

                # 计算分类准确率
                if 'cls_scores' in outputs:
                    pred_labels = outputs['cls_scores'].argmax(dim=1)
                    gt_labels = loss_targets['gt_labels']
                    correct_cls += (pred_labels == gt_labels).sum().item()
                    total_samples += len(gt_labels)

                # 计算交互分类准确率
                if 'interaction_logits' in outputs and \
                        outputs['interaction_logits'] is not None:
                    pred_interactions = outputs['interaction_logits'].argmax(dim=1)
                    gt_inter = loss_targets['gt_interactions']
                    if len(pred_interactions) == len(gt_inter):
                        correct_interaction += \
                            (pred_interactions == gt_inter).sum().item()

        # 计算平均指标
        avg_loss = total_loss / max(num_batches, 1)
        avg_cls_loss = total_cls_loss / max(num_batches, 1)
        avg_bbox_loss = total_bbox_loss / max(num_batches, 1)
        avg_interaction_loss = total_interaction_loss / max(num_batches, 1)
        cls_accuracy = correct_cls / max(total_samples, 1)
        interaction_accuracy = correct_interaction / max(num_batches, 1)

        # 记录历史
        self.history['val_loss'].append(avg_loss)
        self.history['val_cls_acc'].append(cls_accuracy)
        self.history['val_interaction_acc'].append(interaction_accuracy)

        return avg_loss, cls_accuracy, interaction_accuracy

    def train(self, epochs, save_every=5):
        """
        完整训练流程

        Args:
            epochs: 训练轮数
            save_every: 每隔多少epoch保存一次
        """
        self._log(f"\n{'=' * 60}")
        self._log(f"Starting training for {epochs} epochs")
        self._log(f"Mode: {self.config.get('mode', 'imagenet')}")
        self._log(f"Setting: {self.config.get('setting', 'Seen')}")
        self._log(f"Save directory: {self.save_dir}")
        self._log(f"{'=' * 60}")

        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch

            # 训练
            train_loss = self.train_epoch()

            # 验证
            val_loss, cls_acc, inter_acc = self.validate()

            # 更新学习率
            current_lr = self.optimizer.param_groups[-1]['lr']
            self.scheduler.step()

            # 记录学习率
            self.history['lr'].append(current_lr)

            # 打印日志
            self._log(
                f"\nEpoch {epoch + 1}/{epochs} | "
                f"LR: {current_lr:.6f}\n"
                f"  Train Loss: {train_loss:.4f}\n"
                f"  Val Loss: {val_loss:.4f} | "
                f"Cls Acc: {cls_acc:.4f} | "
                f"Inter Acc: {inter_acc:.4f}"
            )

            # 保存最佳模型
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self._save_checkpoint('best.pt', epoch)
                self._log(f"  ★ Saved best model (loss: {val_loss:.4f})")

            # 定期保存
            if (epoch + 1) % save_every == 0:
                self._save_checkpoint(f'epoch_{epoch + 1}.pt', epoch)

        # 保存最终模型
        self._save_checkpoint('final.pt', epochs - 1)

        # 保存训练曲线
        self._save_curves()

        self._log(f"\nTraining completed! Best loss: {self.best_loss:.4f}")

    def _save_checkpoint(self, filename, epoch):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'history': self.history,
            'config': self.config
        }
        path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.history = checkpoint.get('history', self.history)

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    def _save_curves(self):
        """保存训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        mode = self.config.get('mode', 'imagenet')
        setting = self.config.get('setting', 'Seen')
        title_prefix = f"Annotation ({mode}, {setting})"

        if len(self.history['train_loss']) > 0:
            axes[0, 0].plot(self.history['train_loss'],
                            label='Train Loss', color='blue')
            if len(self.history['val_loss']) > 0:
                axes[0, 0].plot(self.history['val_loss'],
                                label='Val Loss', color='red')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title(f'{title_prefix} - Loss')
            axes[0, 0].legend(loc='best')
            axes[0, 0].grid(True)

        if len(self.history['train_cls_loss']) > 0:
            axes[0, 1].plot(self.history['train_cls_loss'],
                            label='Cls Loss', color='blue')
            axes[0, 1].plot(self.history['train_bbox_loss'],
                            label='BBox Loss', color='orange')
            axes[0, 1].plot(self.history['train_interaction_loss'],
                            label='Inter Loss', color='green')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].set_title(f'{title_prefix} - Component Losses')
            axes[0, 1].legend(loc='best')
            axes[0, 1].grid(True)

        if len(self.history['val_cls_acc']) > 0:
            axes[1, 0].plot(self.history['val_cls_acc'],
                            label='Cls Acc', color='blue')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].set_title(f'{title_prefix} - Detection Accuracy')
            axes[1, 0].legend(loc='best')
            axes[1, 0].grid(True)

        if len(self.history['val_interaction_acc']) > 0:
            axes[1, 1].plot(self.history['val_interaction_acc'],
                            label='Inter Acc', color='green')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].set_title(f'{title_prefix} - Interaction Accuracy')
            axes[1, 1].legend(loc='best')
            axes[1, 1].grid(True)

        plt.tight_layout()
        curve_path = os.path.join(self.save_dir, 'training_curves.png')
        plt.savefig(curve_path, dpi=150, bbox_inches='tight')
        plt.close()

        self._log(f"Training curves saved: {curve_path}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train Annotation Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 原始模型训练（随机初始化，无预训练权重）
  python train_annotation.py --mode scratch --data_dir ./Data --setting Seen

  # 使用 ImageNet 预训练权重（自动下载）
  python train_annotation.py --mode imagenet --data_dir ./Data --setting Seen

  # 使用本地 ImageNet 预训练权重文件
  python train_annotation.py --mode imagenet --img_model_path ./ckpt/resnet50.pth --data_dir ./Data

  # 使用配置文件
  python train_annotation.py --config config_annotation.yaml

  # 断点续训
  python train_annotation.py --mode imagenet --resume ./checkpoints/annotation/best.pt
        """
    )

    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML file')
    parser.add_argument('--mode', type=str, default='imagenet',
                        choices=['scratch', 'imagenet'],
                        help='Training mode: scratch (random init) or imagenet (pretrained)')
    parser.add_argument('--setting', type=str, default='Seen',
                        choices=['Seen', 'Unseen'],
                        help='Dataset setting')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='PIAD data directory (containing Seen/ and Unseen/ subdirs)')
    parser.add_argument('--save_dir', type=str,
                        default='./checkpoints/annotation',
                        help='Save directory')
    parser.add_argument('--dataset_type', type=str, default='piad',
                        choices=['piad', 'custom', 'synthetic'],
                        help='Dataset type')
    parser.add_argument('--img_model_path', type=str, default=None,
                        help='Path to local pretrained ResNet-50 weights (.pth)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 加载配置
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        # 确保关键参数从命令行覆盖配置文件
        if args.data_dir:
            config['data_dir'] = args.data_dir
        if args.mode != 'imagenet':
            config['mode'] = args.mode
        if args.setting != 'Seen':
            config['setting'] = args.setting
    else:
        config = {
            'mode': args.mode,
            'setting': args.setting,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'device': args.device,
            'data_dir': args.data_dir,
            'save_dir': args.save_dir,
            'dataset_type': args.dataset_type,
            'num_interactions': 17,
            'img_size': [224, 224],
            'num_workers': 0,
            'pretrained': args.mode == 'imagenet',
            'img_model_path': args.img_model_path,
            'weight_decay': 1e-4
        }

    # 如果是 imagenet 模式，设置 save_dir 包含模式信息
    if config.get('mode', 'imagenet') == 'imagenet':
        config['save_dir'] = os.path.join(
            config.get('save_dir', './checkpoints/annotation'),
            'imagenet'
        )
    else:
        config['save_dir'] = os.path.join(
            config.get('save_dir', './checkpoints/annotation'),
            'scratch'
        )

    # 创建训练器
    trainer = AnnotationTrainer(config)

    # 断点续训
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # 开始训练
    trainer.train(config.get('epochs', args.epochs),
                  save_every=config.get('save_every', 5))


if __name__ == "__main__":
    main()
