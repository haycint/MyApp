"""
Training module for IAGNet
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import random
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from model.iagnet import get_IAGNet
from utils.loss import HM_Loss, kl_div
from utils.eval import SIM
from data_utils.dataset import PIAD


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_yaml(path):
    with open(path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    print("==============================================================================================================================")
    print("[Admin log]config:",config)
    print("==============================================================================================================================")
    return config


class Trainer:
    """IAGNet Trainer class"""

    def __init__(self, config_path, data_dir, ckpt_dir, use_gpu=True,
                 loss_cls=0.3, loss_kl=0.5, log_callback=None):
        self.config = read_yaml(config_path)
        self.data_dir = data_dir
        self.ckpt_dir = ckpt_dir
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.loss_cls = loss_cls
        self.loss_kl = loss_kl
        self.log_callback = log_callback

        self.device = torch.device("cuda:0" if self.use_gpu else "cpu")
        self._update_data_paths()

        self.history = {
            'train_loss': [], 'val_loss': [], 'val_auc': [],
            'val_iou': [], 'val_sim': [], 'val_mae': []
        }

        self.current_epoch = 0
        self.best_auc = 0
        self.is_training = False
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.criterion_hm = None
        self.criterion_ce = None
        self.optimizer = None
        self.scheduler = None

    def _update_data_paths(self):
        setting = self.config['Setting']
        self.config['img_train'] = os.path.join(self.data_dir, setting, 'Img_Train.txt')
        self.config['img_test'] = os.path.join(self.data_dir, setting, 'Img_Test.txt')
        self.config['point_train'] = os.path.join(self.data_dir, setting, 'Point_Train.txt')
        self.config['point_test'] = os.path.join(self.data_dir, setting, 'Point_Test.txt')
        self.config['box_train'] = os.path.join(self.data_dir, setting, 'Box_Train.txt')
        self.config['box_test'] = os.path.join(self.data_dir, setting, 'Box_Test.txt')

    def _log(self, message):
        print(message)
        if self.log_callback:
            self.log_callback(message)

    def init_model(self):
        self._log("Initializing model...")
        self.model = get_IAGNet(
            img_model_path=self.config.get('res18_pre'),
            pre_train=self.config.get('res18_pre') is not None,
            N_p=self.config['N_p'], emb_dim=self.config['emb_dim'],
            proj_dim=self.config['proj_dim'], num_heads=self.config['num_heads'],
            N_raw=self.config['N_raw'], num_affordance=self.config['num_affordance']
        )
        self.model = self.model.to(self.device)
        self._log(f"Model initialized on {self.device}")

    def init_dataloaders(self):
        self._log("Loading training data...")
        train_dataset = PIAD('train', self.config['Setting'],
                             self.config['point_train'], self.config['img_train'],
                             self.config['box_train'], self.config['pairing_num'])
        self.train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'],
                                       num_workers=0, shuffle=True, drop_last=True)
        self._log(f"Training samples: {len(train_dataset)}")

        self._log("Loading validation data...")
        val_dataset = PIAD('val', self.config['Setting'],
                           self.config['point_test'], self.config['img_test'],
                           self.config['box_test'])
        self.val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'],
                                     num_workers=0, shuffle=True)
        self._log(f"Validation samples: {len(val_dataset)}")

    def init_loss(self):
        self.criterion_hm = HM_Loss().to(self.device)
        self.criterion_ce = nn.CrossEntropyLoss().to(self.device)

    def init_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'],
                                          betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config['Epoch'], eta_min=1e-6)

    def train_epoch(self):
        self.model.train()
        num_batches = len(self.train_loader)
        loss_sum = 0

        for i, (img, points, labels, logits_labels, sub_box, obj_box) in enumerate(self.train_loader):
            if not self.is_training:
                break

            self.optimizer.zero_grad()
            temp_loss = 0

            for point, label, logits_label in zip(points, labels, logits_labels):
                point, label = point.float(), label.float()
                label = label.unsqueeze(dim=-1)

                if self.use_gpu:
                    img = img.to(self.device)
                    point = point.to(self.device)
                    label = label.to(self.device)
                    logits_label = logits_label.to(self.device)
                    sub_box = sub_box.to(self.device)
                    obj_box = obj_box.to(self.device)

                _3d, logits, to_KL = self.model(img, point, sub_box, obj_box)

                loss_hm = self.criterion_hm(_3d, label)
                loss_ce = self.criterion_ce(logits, logits_label)
                loss_kl = kl_div(to_KL[0], to_KL[1])

                temp_loss += loss_hm + self.loss_cls * loss_ce + self.loss_kl * loss_kl

            temp_loss.backward()
            self.optimizer.step()
            loss_sum += temp_loss.item()

            if i % 10 == 0:
                self._log(f"Epoch {self.current_epoch} | Batch {i}/{num_batches} | Loss: {temp_loss.item():.4f}")

        mean_loss = loss_sum / (num_batches * self.config['pairing_num'])
        self.history['train_loss'].append(mean_loss)
        return mean_loss

    def validate(self):
        self.model.eval()
        val_dataset = self.val_loader.dataset
        results = torch.zeros((len(val_dataset), 2048, 1))
        targets = torch.zeros((len(val_dataset), 2048, 1))

        num_batches = len(self.val_loader)
        val_loss_sum = 0
        total_mae = 0
        total_points = 0
        num = 0

        with torch.no_grad():
            for i, (img, point, label, _, _, sub_box, obj_box) in enumerate(self.val_loader):
                point, label = point.float(), label.float()
                label = label.unsqueeze(dim=-1)

                if self.use_gpu:
                    img = img.to(self.device)
                    point = point.to(self.device)
                    label = label.to(self.device)
                    sub_box = sub_box.to(self.device)
                    obj_box = obj_box.to(self.device)

                _3d, logits, to_KL = self.model(img, point, sub_box, obj_box)

                val_loss_hm = self.criterion_hm(_3d, label)
                val_loss_kl = kl_div(to_KL[0], to_KL[1])
                val_loss = val_loss_hm + self.loss_kl * val_loss_kl

                mae, point_nums = self._evaluating(_3d, label)
                total_mae += mae.item()
                total_points += point_nums
                val_loss_sum += val_loss.item()

                pred_num = _3d.shape[0]
                results[num:num + pred_num, :, :] = _3d.cpu()
                targets[num:num + pred_num, :, :] = label.cpu()
                num += pred_num

        val_mean_loss = val_loss_sum / num_batches
        mean_mae = total_mae / total_points

        results_np = results.numpy()
        targets_np = targets.numpy()

        # Calculate SIM
        sim_values = np.zeros(targets_np.shape[0])
        for i in range(targets_np.shape[0]):
            sim_values[i] = SIM(results_np[i], targets_np[i])
        sim = np.mean(sim_values)

        # Calculate AUC and IOU
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
                auc_values[i] = roc_auc_score(t_true.flatten(), p_score.flatten())
                temp_iou = []
                for thre in iou_thres:
                    p_mask = (p_score >= thre).astype(int)
                    intersect = np.sum(p_mask & t_true)
                    union = np.sum(p_mask | t_true)
                    temp_iou.append(1. * intersect / union if union > 0 else 0)
                iou_values[i] = np.mean(temp_iou)

        auc = np.nanmean(auc_values)
        iou = np.nanmean(iou_values)

        # Update history
        self.history['val_loss'].append(val_mean_loss)
        self.history['val_auc'].append(auc)
        self.history['val_iou'].append(iou)
        self.history['val_sim'].append(sim)
        self.history['val_mae'].append(mean_mae)

        # Save best model
        if auc > self.best_auc:
            self.best_auc = auc
            self._save_checkpoint('best.pt', self.current_epoch)

        return val_mean_loss, auc, iou, sim, mean_mae

    def _evaluating(self, pred, label):
        mae = torch.sum(torch.abs(pred - label), dim=(0, 1))
        points_num = pred.shape[0] * pred.shape[1]
        return mae, points_num

    def train(self, num_epochs, progress_callback=None):
        self.is_training = True
        self._log(f"Starting training for {num_epochs} epochs...")

        for epoch in range(self.current_epoch, num_epochs):
            if not self.is_training:
                break

            self.current_epoch = epoch
            self._log(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

            # Training
            train_loss = self.train_epoch()
            self._log(f"Train Loss: {train_loss:.4f}")

            # Validation
            val_loss, auc, iou, sim, mae = self.validate()
            self._log(f"Val Loss: {val_loss:.4f} | AUC: {auc:.4f} | IOU: {iou:.4f} | SIM: {sim:.4f} | MAE: {mae:.4f}")

            self.scheduler.step()

            if progress_callback:
                progress_callback(epoch + 1, num_epochs, train_loss, val_loss, auc, iou, sim, mae)

        self._log("Training completed!")
        return self.history

    def stop_training(self):
        self.is_training = False

    def _save_checkpoint(self, filename, epoch):
        os.makedirs(self.ckpt_dir, exist_ok=True)
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch,
            'history': self.history,
            'config': self.config,
            'best_auc': self.best_auc
        }
        path = os.path.join(self.ckpt_dir, filename)
        torch.save(checkpoint, path)

    def save_model(self, model_name):
        self._save_checkpoint(f'{model_name}.pt', self.current_epoch)
        self._log(f"Model saved as {model_name}.pt")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.init_model()
        self.model.load_state_dict(checkpoint['model'])

        if 'optimizer' in checkpoint:
            self.init_optimizer()
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        if 'epoch' in checkpoint:
            self.current_epoch = checkpoint['epoch'] + 1
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        if 'best_auc' in checkpoint:
            self.best_auc = checkpoint['best_auc']

        self._log(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint.get('config', {}).get('Setting', 'Seen')

    def get_history(self):
        return self.history

    def plot_and_save_curves(self, save_dir, model_name):
        os.makedirs(save_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        if len(self.history['train_loss']) > 0:
            axes[0, 0].plot(self.history['train_loss'], label='Train Loss', color='blue')
            if len(self.history['val_loss']) > 0:
                axes[0, 0].plot(self.history['val_loss'], label='Val Loss', color='red')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Loss Curve')
            axes[0, 0].legend()
            axes[0, 0].grid(True)

        if len(self.history['val_auc']) > 0:
            axes[0, 1].plot(self.history['val_auc'], label='Val AUC', color='green')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('AUC')
            axes[0, 1].set_title('AUC Curve')
            axes[0, 1].legend()
            axes[0, 1].grid(True)

        if len(self.history['val_iou']) > 0:
            axes[1, 0].plot(self.history['val_iou'], label='Val IOU', color='orange')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('IOU')
            axes[1, 0].set_title('IOU Curve')
            axes[1, 0].legend()
            axes[1, 0].grid(True)

        if len(self.history['val_sim']) > 0:
            axes[1, 1].plot(self.history['val_sim'], label='Val SIM', color='purple')
            if len(self.history['val_mae']) > 0:
                ax2 = axes[1, 1].twinx()
                ax2.plot(self.history['val_mae'], label='Val MAE', color='brown', linestyle='--')
                ax2.set_ylabel('MAE', color='brown')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('SIM')
            axes[1, 1].set_title('SIM and MAE Curves')
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        plt.tight_layout()
        curve_path = os.path.join(save_dir, f'{model_name}-loss.png')
        plt.savefig(curve_path, dpi=150, bbox_inches='tight')
        plt.close()

        return curve_path
