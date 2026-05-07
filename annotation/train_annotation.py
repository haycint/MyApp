"""
Training Script for Annotation Model
标注模型训练脚本
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
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from annotation_model import build_annotation_model, AnnotationLoss
from annotation_dataset import build_dataloader


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AnnotationTrainer:
    def __init__(self, config):
        self.config = config
        set_seed(config.get('seed', 42))
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.save_dir = config.get('save_dir', './checkpoints/annotation')
        os.makedirs(self.save_dir, exist_ok=True)

        self._init_model()
        self._init_dataloaders()
        self._init_loss()
        self._init_optimizer()

        self.writer = SummaryWriter(os.path.join(self.save_dir, 'logs', datetime.now().strftime('%Y%m%d_%H%M%S')))
        self.best_loss = float('inf')
        self.current_epoch = 0

        print(f'Trainer initialized on {self.device}')

    def _init_model(self):
        self.model = build_annotation_model(
            num_interactions=self.config.get('num_interactions', 17),
            pretrained=self.config.get('pretrained', True)
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Total parameters: {total_params / 1e6:.2f}M')
        print(f'Trainable parameters: {trainable_params / 1e6:.2f}M')

    def _init_dataloaders(self):
        self.train_loader = build_dataloader(
            dataset_type=self.config.get('dataset_type', 'piad'),
            data_dir=self.config.get('data_dir'),
            split='train',
            batch_size=self.config.get('batch_size', 8),
            num_workers=self.config.get('num_workers', 4),
            img_size=tuple(self.config.get('img_size', [224, 224])),
            augment=True
        )
        self.val_loader = build_dataloader(
            dataset_type=self.config.get('dataset_type', 'piad'),
            data_dir=self.config.get('data_dir'),
            split='val',
            batch_size=self.config.get('batch_size', 8),
            num_workers=self.config.get('num_workers', 4),
            img_size=tuple(self.config.get('img_size', [224, 224])),
            augment=False
        )

    def _init_loss(self):
        self.criterion = AnnotationLoss()

    def _init_optimizer(self):
        backbone_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
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

    def _prepare_targets(self, targets):
        subject_boxes = torch.stack([t['subject_box'] for t in targets], dim=0).to(self.device)
        object_boxes = torch.stack([t['object_box'] for t in targets], dim=0).to(self.device)
        interactions = torch.stack([t['interaction'] for t in targets], dim=0).to(self.device)
        return {
            'subject_boxes': subject_boxes,
            'object_boxes': object_boxes,
            'interaction': interactions
        }

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1} [Train]')

        for images, targets in pbar:
            images = images.to(self.device)
            target_data = self._prepare_targets(targets)
            outputs = self.model(images)
            losses = self.criterion(outputs, target_data)
            loss = losses['total_loss']

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'interaction': f'{losses.get("interaction_loss", torch.tensor(0.0)).item():.4f}',
                'subject_bbox': f'{losses.get("subject_bbox_loss", torch.tensor(0.0)).item():.4f}',
                'object_bbox': f'{losses.get("object_bbox_loss", torch.tensor(0.0)).item():.4f}'
            })

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.writer.add_scalar('train/loss', avg_loss, self.current_epoch)
        return avg_loss

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc='Validating'):
                images = images.to(self.device)
                target_data = self._prepare_targets(targets)
                outputs = self.model(images)
                losses = self.criterion(outputs, target_data)
                total_loss += losses['total_loss'].item()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.writer.add_scalar('val/loss', avg_loss, self.current_epoch)
        return avg_loss

    def save_checkpoint(self, name):
        path = os.path.join(self.save_dir, name)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'epoch': self.current_epoch
        }, path)
        print(f'Saved checkpoint: {path}')

    def train(self):
        epochs = self.config.get('epochs', 100)
        save_every = self.config.get('save_every', 5)

        for epoch in range(epochs):
            self.current_epoch = epoch
            train_loss = self.train_epoch()
            val_loss = self.validate()
            self.scheduler.step()

            print(f'Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}')

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint('best.pt')

            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'epoch_{epoch + 1}.pt')


def parse_args():
    parser = argparse.ArgumentParser(description='Train annotation model')
    parser.add_argument('--config', default='config_annotation.yaml', help='Path to YAML config file')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    trainer = AnnotationTrainer(config)
    trainer.train()
