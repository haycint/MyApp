"""
Training Script for Annotation Model
标注模型训练脚本

功能:
- 训练主体/客体检测
- 训练交互动作分类
- 支持断点续训
- 训练曲线可视化
- 模型保存与加载
"""

import os
import sys
import random
import yaml
import argparse
import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from annotation_model import build_annotation_model, AnnotationLoss
from annotation_dataset import build_dataloader


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
    """
    
    def __init__(self, config):
        """
        Args:
            config: 配置字典
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_cls_loss': [],# 分类损失
            'train_bbox_loss': [],# 边框损失
            'val_cls_acc': []
        }
        
        # 创建保存目录
        self.save_dir = config.get('save_dir', './checkpoints/annotation')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 初始化组件
        self._init_model()
        self._init_dataloaders()
        self._init_loss()
        self._init_optimizer()
        
        # TensorBoard
        log_dir = os.path.join(self.save_dir, 'logs', datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.writer = SummaryWriter(log_dir)
        
        print(f"Trainer initialized on {self.device}")
    
    def _init_model(self):
        """初始化模型"""
        self.model = build_annotation_model(
            num_interactions=self.config.get('num_interactions', 17),
            pretrained=self.config.get('pretrained', True)
        )
        self.model = self.model.to(self.device)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params / 1e6:.2f}M")
        print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    def _init_dataloaders(self):
        """初始化数据加载器"""
        dataset_type = self.config.get('dataset_type', 'synthetic')
        data_dir = self.config.get('data_dir')
        
        self.train_loader = build_dataloader(
            dataset_type=dataset_type,
            data_dir=data_dir,
            split='train',
            batch_size=self.config.get('batch_size', 8),
            num_workers=self.config.get('num_workers', 4),
            img_size=self.config.get('img_size', (224, 224)),
            augment=True
        )
        
        self.val_loader = build_dataloader(
            dataset_type=dataset_type,
            data_dir=data_dir,
            split='val',
            batch_size=self.config.get('batch_size', 8),
            num_workers=self.config.get('num_workers', 4),
            img_size=self.config.get('img_size', (224, 224)),
            augment=False
        )
    
    def _init_loss(self):
        """初始化损失函数"""
        self.criterion = AnnotationLoss()
    
    def _init_optimizer(self):
        """初始化优化器"""
        # 不同层使用不同学习率
        backbone_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'layer' in name or 'conv1' in name or 'bn1' in name:
                backbone_params.append(param)
            else:
                other_params.append(param)
        
        self.optimizer = optim.Adam([
            {'params': backbone_params, 'lr': self.config.get('lr', 1e-4) * 0.1},
            {'params': other_params, 'lr': self.config.get('lr', 1e-4)}
        ], weight_decay=self.config.get('weight_decay', 1e-4))
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.get('epochs', 100),
            eta_min=1e-6
        )
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        cls_loss_sum = 0
        bbox_loss_sum = 0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            
            # 准备目标数据
            # 合并批次中的所有proposals
            batch_proposals = []
            all_gt_labels = []
            all_gt_boxes = []
            
            for i, target in enumerate(targets):
                # 使用真实边界框作为proposals
                subject_box = target['subject_box'].to(self.device)
                object_box = target['object_box'].to(self.device)
                
                # 添加一些噪声boxes作为负样本
                noise_boxes = torch.randn(10, 4, device=self.device) * 50 + 100
                noise_boxes = torch.clamp(noise_boxes, 0, 224)
                
                proposals = torch.cat([subject_box.unsqueeze(0), object_box.unsqueeze(0), noise_boxes], dim=0)
                batch_proposals.append(proposals)
                
                # 标签: 0=bg, 1=subject, 2=object
                gt_labels = torch.zeros(len(proposals), dtype=torch.long, device=self.device)
                gt_labels[0] = 1  # subject
                gt_labels[1] = 2  # object
                all_gt_labels.append(gt_labels)
                
                # 边界框偏移目标
                # 1. 初始化目标张量，形状为 [num_proposals, num_classes * 4] 即 [N, 12]，与模型输出匹配
                gt_boxes = torch.zeros(len(proposals), 3 * 4, device=self.device, dtype=torch.float32) # 3个类别 * 4个坐标

                # 2. 获取当前图片对应的两个真实框（主体和客体）
                # target 是当前图片的标注字典
                gt_subject_box = target['subject_box'].to(self.device)  # 形状 [4]
                gt_object_box = target['object_box'].to(self.device)    # 形状 [4]

                # 3. 为每个proposal计算与两个真实框的偏移量 (dx, dy, dw, dh)
                # 公式参考模型中的 _apply_deltas 方法（或标准R-CNN）
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
                    gt_boxes[prop_idx, 1 * 4:2 * 4] = torch.tensor([dx_sub, dy_sub, dw_sub, dh_sub]) # 填入“主体”类别(索引1)对应的位置
                    
                    # 计算与客体框的偏移
                    gt_w = gt_object_box[2] - gt_object_box[0]
                    gt_h = gt_object_box[3] - gt_object_box[1]
                    gt_ctr_x = gt_object_box[0] + 0.5 * gt_w
                    gt_ctr_y = gt_object_box[1] + 0.5 * gt_h
                    
                    dx_obj = (gt_ctr_x - prop_ctr_x) / (prop_w + 1e-6)
                    dy_obj = (gt_ctr_y - prop_ctr_y) / (prop_h + 1e-6)
                    dw_obj = torch.log(gt_w / (prop_w + 1e-6) + 1e-6)
                    dh_obj = torch.log(gt_h / (prop_h + 1e-6) + 1e-6)
                    gt_boxes[prop_idx, 2 * 4:3 * 4] = torch.tensor([dx_obj, dy_obj, dw_obj, dh_obj]) # 填入“客体”类别(索引2)对应的位置
                    
                    # 注意：背景类别(索引0)对应的4个偏移量保持为0

                all_gt_boxes.append(gt_boxes)
            
            # 前向传播
            outputs = self.model(images, proposals=batch_proposals)
            
            # 准备损失目标
            loss_targets = {
                'gt_labels': torch.cat(all_gt_labels),
                'gt_boxes': torch.cat(all_gt_boxes)
            }
            
            # 计算损失
            losses = self.criterion(outputs, loss_targets)
            loss = losses['total_loss']
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            print("losses:",losses)
            # 记录损失
            total_loss += loss.item()
            if 'cls_loss' in losses:
                cls_loss_sum += losses['cls_loss'].item()
            if 'bbox_loss' in losses:
                bbox_loss_sum += losses['bbox_loss'].item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cls': f'{losses.get("cls_loss", torch.tensor(0)).item():.4f}',
                'bbox': f'{losses.get("bbox_loss", torch.tensor(0)).item():.4f}'
            })
        
        avg_loss = total_loss / num_batches
        avg_cls_loss = cls_loss_sum / num_batches
        avg_bbox_loss = bbox_loss_sum / num_batches
        
        self.history['train_loss'].append(avg_loss)
        self.history['train_cls_loss'].append(avg_cls_loss)
        self.history['train_bbox_loss'].append(avg_bbox_loss)
        
        return avg_loss
    
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        total_cls_loss = 0
        total_bbox_loss = 0
        correct = 0
        total = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc='Validating'):
                images = images.to(self.device)
                
                # 准备验证用的目标数据 - 与训练时一致，包含噪声box
                batch_proposals = []
                all_gt_labels = []
                all_gt_boxes = []
                
                for i, target in enumerate(targets):
                    # 使用真实边界框作为proposals，并添加噪声boxes作为负样本
                    subject_box = target['subject_box'].to(self.device)
                    object_box = target['object_box'].to(self.device)
                    
                    # 添加一些噪声boxes作为负样本（与训练时一致）
                    noise_boxes = torch.randn(10, 4, device=self.device) * 50 + 100
                    noise_boxes = torch.clamp(noise_boxes, 0, 224)
                    
                    proposals = torch.cat([subject_box.unsqueeze(0), object_box.unsqueeze(0), noise_boxes], dim=0)
                    batch_proposals.append(proposals)
                    
                    # 标签: 0=bg, 1=subject, 2=object
                    gt_labels = torch.zeros(len(proposals), dtype=torch.long, device=self.device)
                    gt_labels[0] = 1  # subject
                    gt_labels[1] = 2  # object
                    all_gt_labels.append(gt_labels)
                    
                    # 计算边界框回归目标
                    gt_boxes = torch.zeros(len(proposals), 3 * 4, device=self.device, dtype=torch.float32)
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
                        gt_boxes[prop_idx, 1 * 4:2 * 4] = torch.tensor([dx_sub, dy_sub, dw_sub, dh_sub], device=self.device)
                        
                        # 计算与客体框的偏移
                        gt_w = gt_object_box[2] - gt_object_box[0]
                        gt_h = gt_object_box[3] - gt_object_box[1]
                        gt_ctr_x = gt_object_box[0] + 0.5 * gt_w
                        gt_ctr_y = gt_object_box[1] + 0.5 * gt_h
                        
                        dx_obj = (gt_ctr_x - prop_ctr_x) / (prop_w + 1e-6)
                        dy_obj = (gt_ctr_y - prop_ctr_y) / (prop_h + 1e-6)
                        dw_obj = torch.log(gt_w / (prop_w + 1e-6) + 1e-6)
                        dh_obj = torch.log(gt_h / (prop_h + 1e-6) + 1e-6)
                        gt_boxes[prop_idx, 2 * 4:3 * 4] = torch.tensor([dx_obj, dy_obj, dw_obj, dh_obj], device=self.device)
                    
                    all_gt_boxes.append(gt_boxes)
                
                # 前向传播
                outputs = self.model(images, proposals=batch_proposals)
                
                # 计算损失
                loss_targets = {
                    'gt_labels': torch.cat(all_gt_labels),
                    'gt_boxes': torch.cat(all_gt_boxes)
                }
                losses = self.criterion(outputs, loss_targets)
                print("losses:",losses)
                # 累积损失 - 确保处理所有可能的返回类型
                if losses.get('total_loss') is not None:
                    total_loss_val = losses['total_loss']
                    if isinstance(total_loss_val, torch.Tensor):
                        total_loss += total_loss_val.item()
                    elif isinstance(total_loss_val, (int, float)):
                        total_loss += float(total_loss_val)
                
                if losses.get('cls_loss') is not None:
                    cls_loss_val = losses['cls_loss']
                    if isinstance(cls_loss_val, torch.Tensor):
                        total_cls_loss += cls_loss_val.item()
                    elif isinstance(cls_loss_val, (int, float)):
                        total_cls_loss += float(cls_loss_val)
                
                if losses.get('bbox_loss') is not None:
                    bbox_loss_val = losses['bbox_loss']
                    if isinstance(bbox_loss_val, torch.Tensor):
                        total_bbox_loss += bbox_loss_val.item()
                    elif isinstance(bbox_loss_val, (int, float)):
                        total_bbox_loss += float(bbox_loss_val)
                
                # 计算交互分类准确率
                if 'interaction_logits' in outputs and outputs['interaction_logits'] is not None:
                    pred_interactions = outputs['interaction_logits'].argmax(dim=1)
                    if len(pred_interactions) > 0 and len(targets) > 0:
                        gt_interaction = targets[0]['interaction'].to(self.device)
                        if pred_interactions[0] == gt_interaction:
                            correct += 1
                        total += 1
        
        # 计算平均指标
        avg_loss = total_loss / max(num_batches, 1)
        avg_cls_loss = total_cls_loss / max(num_batches, 1)
        avg_bbox_loss = total_bbox_loss / max(num_batches, 1)
        accuracy = correct / max(total, 1) if total > 0 else 0
        
        # 记录历史
        self.history['val_loss'].append(avg_loss)
        self.history['val_cls_acc'].append(accuracy)
        
        return avg_loss, accuracy
    
    def train(self, epochs, save_every=5):
        """
        完整训练流程
        
        Args:
            epochs: 训练轮数
            save_every: 每隔多少epoch保存一次
        """
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Save directory: {self.save_dir}")
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_loss, val_acc = self.validate()
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录到TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[-1]['lr'], epoch)
            
            # 打印日志
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Accuracy: {val_acc:.4f}")
            
            # 保存最佳模型
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self._save_checkpoint('best.pt', epoch)
                print(f"  Saved best model (loss: {val_loss:.4f})")
            
            # 定期保存
            if (epoch + 1) % save_every == 0:
                self._save_checkpoint(f'epoch_{epoch + 1}.pt', epoch)
        
        # 保存最终模型
        self._save_checkpoint('final.pt', epochs - 1)
        print(f"\nTraining completed! Best loss: {self.best_loss:.4f}")
        
        # 关闭TensorBoard
        self.writer.close()
    
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
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.history = checkpoint.get('history', self.history)
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")


def main():
    parser = argparse.ArgumentParser(description='Train Annotation Model')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--data_dir', type=str, default=None, help='Data directory')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/annotation', help='Save directory')
    parser.add_argument('--dataset_type', type=str, default='synthetic', help='Dataset type')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载配置
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'epochs': args.epochs,# 100
            'batch_size': args.batch_size,# 8
            'lr': args.lr,# 1e-4
            'device': args.device,
            'data_dir': args.data_dir,# ./Data
            'save_dir': args.save_dir,# ./checkpoints/annotation
            'dataset_type': args.dataset_type,# PIAD
            'num_interactions': 17,# 17类交互动作
            'img_size': (224, 224),
            'num_workers': 4,
            'pretrained': True,
            'weight_decay': 1e-4
        }
    
    # 创建训练器
    trainer = AnnotationTrainer(config)
    
    # 断点续训
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    trainer.train(config.get('epochs', args.epochs))


if __name__ == "__main__":
    main()# cmd in MyApp
