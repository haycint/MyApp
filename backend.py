import os
import sys
import json
import time
import threading
import numpy as np
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.MyNet import get_MyNet
from data_utils.dataset import PIAD, PIADInference
from utils.loss import HM_Loss, kl_div
from utils.eval import SIM
import torch.nn.functional as F

# Constants
CKPT_DIR = os.path.join(os.path.dirname(__file__), 'ckpt')
BREAK_POINT_DIR = os.path.join(os.path.dirname(__file__), 'break_point')
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'Data')

AFFORDANCE_LABELS = [
    'grasp', 'contain', 'lift', 'open', 'lay', 'sit', 'support',
    'wrapgrasp', 'pour', 'move', 'display', 'push', 'listen',
    'wear', 'press', 'cut', 'stab'
]


def ensure_dir(path):
    """Ensure directory exists"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def get_model_name(setting):
    """Generate model name based on current time and setting"""
    now = datetime.now()
    return f"{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}-{setting}-model"


def get_log_name(setting):
    """Generate log name based on current time and setting"""
    now = datetime.now()
    return f"{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}-{setting}-log"


def get_breakpoint_name(model_name, epoch):
    """Generate breakpoint name"""
    return f"{model_name}-{epoch}"


def get_available_models():
    """Get list of available trained models"""
    ensure_dir(CKPT_DIR)
    models = []
    for f in os.listdir(CKPT_DIR):
        if f.endswith('.pt') and 'model' in f:
            model_path = os.path.join(CKPT_DIR, f)
            stat = os.stat(model_path)
            mtime = datetime.fromtimestamp(stat.st_mtime)
            models.append({
                'name': f.replace('.pt', ''),
                'path': model_path,
                'mtime': mtime,
                'size': stat.st_size / (1024 * 1024)
            })
    return sorted(models, key=lambda x: x['mtime'], reverse=True)


def get_available_breakpoints(setting=None):
    """Get list of available breakpoints"""
    print(setting)
    ensure_dir(BREAK_POINT_DIR)
    breakpoints = []
    for f in os.listdir(BREAK_POINT_DIR):
        if f.endswith('.pt'):
            parts = f.replace('.pt', '').split('-')
            print("parts:", parts)
            print("len(parts):", len(parts))
            if len(parts) >= 6:
                # 解析设置
                try:
                    # 假设文件名格式: YYYY-MM-DD-HH-MM-setting-epoch
                    bp_setting = parts[5]  # 第6部分是设置
                    epoch_str = parts[-1]  # 最后一部分是epoch
                    print("bp_setting:", bp_setting, "epoch_str:", epoch_str)
                    if setting and bp_setting != setting:
                        continue
                        
                    bp_path = os.path.join(BREAK_POINT_DIR, f)
                    stat = os.stat(bp_path)
                    mtime = datetime.fromtimestamp(stat.st_mtime)
                    
                    # 验证epoch是否为数字
                    epoch = 0
                    if epoch_str.isdigit():
                        epoch = int(epoch_str)
                    else:
                        # 如果不是数字，可能是没有epoch编号
                        epoch = 0
                        bp_setting = parts[-1]  # 重新设置
                    
                    breakpoints.append({
                        'name': f.replace('.pt', ''),
                        'path': bp_path,
                        'mtime': mtime,
                        'setting': bp_setting,
                        'epoch': epoch,
                        'size': stat.st_size / (1024 * 1024)
                    })
                    print("bp_path:", bp_path, "mtime:", mtime, "setting:", bp_setting, "epoch:", epoch)
                except (IndexError, ValueError) as e:
                    print(f"Warning: Failed to parse breakpoint file {f}: {e}")
                    continue
    
    return sorted(breakpoints, key=lambda x: x['mtime'], reverse=True)

def get_available_logs():
    """Get list of available log files"""
    ensure_dir(LOG_DIR)
    logs = []
    for f in os.listdir(LOG_DIR):
        if f.endswith('.txt'):
            log_path = os.path.join(LOG_DIR, f)
            stat = os.stat(log_path)
            mtime = datetime.fromtimestamp(stat.st_mtime)
            # Parse setting from filename
            parts = f.replace('.txt', '').split('-')
            log_setting = parts[4] if len(parts) > 5 else 'Unknown'
            logs.append({
                'name': f.replace('.txt', ''),
                'path': log_path,
                'mtime': mtime,
                'setting': log_setting,
                'size': stat.st_size / 1024
            })
    return sorted(logs, key=lambda x: x['mtime'], reverse=True)


def find_associated_files(model_name):
    """Find log and curve files associated with a model"""
    base_name = model_name.replace('-model', '')
    log_file = None
    loss_file = None

    ensure_dir(CKPT_DIR)
    for f in os.listdir(CKPT_DIR):
        if base_name in f:
            if f.endswith('-log.txt'):
                log_file = os.path.join(CKPT_DIR, f)
            elif f.endswith('-loss.png'):
                loss_file = os.path.join(CKPT_DIR, f)

    # Also check logs directory
    ensure_dir(LOG_DIR)
    for f in os.listdir(LOG_DIR):
        if base_name.replace('-model', '') in f and f.endswith('.txt'):
            log_file = os.path.join(LOG_DIR, f)

    return log_file, loss_file


class TrainingState:
    """Training state management with file-based logging"""

    def __init__(self, max_logs=1000):
        self.max_logs = max_logs
        self._lock = threading.Lock()
        self.log_file = None
        self.log_file_path = None
        self.reset()

    def reset(self):
        with self._lock:
            self.is_training = False
            self.stop_requested = False
            self.current_epoch = 0
            self.total_epochs = 0
            self.train_loss = 0.0
            self.val_loss = 0.0
            self.auc = 0.0
            self.iou = 0.0
            self.sim = 0.0
            self.mae = 0.0
            self.lr = 0.0
            self.history = {
                'train_loss': [], 'val_loss': [],
                'val_auc': [], 'val_iou': [],
                'val_sim': [], 'val_mae': []
            }
            self.logs = deque(maxlen=self.max_logs)
            self.model_name = ""
            self.setting = "Seen"
            self.start_time = None
            self.error_message = None
            self.log_file_path = None
            self.log_file = None
            self._stop_flag = threading.Event()  # Stop signal

    def request_stop(self):
        """Request training to stop"""
        self._stop_flag.set()
        self.stop_requested = True

    def is_stop_requested(self):
        """Check if stop was requested"""
        return self._stop_flag.is_set()

    def clear_stop(self):
        """Clear stop flag"""
        self._stop_flag.clear()
        self.stop_requested = False

    def init_log_file(self, log_name):
        """Initialize log file"""
        ensure_dir(LOG_DIR)
        self.log_file_path = os.path.join(LOG_DIR, f"{log_name}.txt")
        self.log_file = open(self.log_file_path, 'a', encoding='utf-8')

    def close_log_file(self):
        """Close log file"""
        if self.log_file:
            self.log_file.close()
            self.log_file = None

    def add_log(self, message):
        """Add log entry - writes to both memory and file"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {message}"

        with self._lock:
            self.logs.append(log_entry)

            # Write to file
            if self.log_file:
                try:
                    self.log_file.write(log_entry + '\n')
                    self.log_file.flush()
                except:
                    pass

    def get_logs(self):
        """Get logs from memory"""
        with self._lock:
            return list(self.logs)

    def load_logs_from_file(self, log_path):
        """Load logs from a log file"""
        if os.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8') as f:
                return f.read()
        return ""

    def to_dict(self):
        with self._lock:
            return {
                'is_training': self.is_training,
                'current_epoch': self.current_epoch,
                'total_epochs': self.total_epochs,
                'train_loss': self.train_loss,
                'val_loss': self.val_loss,
                'auc': self.auc,
                'iou': self.iou,
                'sim': self.sim,
                'mae': self.mae,
                'lr': self.lr,
                'history': self.history.copy(),
                'model_name': self.model_name,
                'setting': self.setting,
                'error_message': self.error_message,
                'log_file_path': self.log_file_path
            }

    def from_dict(self, data):
        """Load state from dictionary"""
        with self._lock:
            self.is_training = data.get('is_training', False)
            self.current_epoch = data.get('current_epoch', 0)
            self.total_epochs = data.get('total_epochs', 0)
            self.train_loss = data.get('train_loss', 0.0)
            self.val_loss = data.get('val_loss', 0.0)
            self.auc = data.get('auc', 0.0)
            self.iou = data.get('iou', 0.0)
            self.sim = data.get('sim', 0.0)
            self.mae = data.get('mae', 0.0)
            self.lr = data.get('lr', 0.0)
            self.history = data.get('history', {
                'train_loss': [], 'val_loss': [],
                'val_auc': [], 'val_iou': [],
                'val_sim': [], 'val_mae': []
            })
            self.model_name = data.get('model_name', '')
            self.setting = data.get('setting', 'Seen')
            self.error_message = data.get('error_message')
            self.log_file_path = data.get('log_file_path')


class TrainerBackend:
    """Backend trainer with checkpoint management"""

    def __init__(self, state):
        self.state = state
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.criterion_hm = None
        self.criterion_ce = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.config = None
        self._stop_flag = threading.Event()
        self.breakpoint_files = []

    def setup(self, setting, data_dir, epochs, batch_size, lr,
              use_gpu=True, loss_cls=0.3, loss_kl=0.5):
        """Setup training environment"""
        self.device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
        self.state.add_log(f"Using device: {self.device}")

        self.config = {
            'Setting': setting,
            'batch_size': batch_size,
            'lr': lr,
            'Epoch': epochs,
            'loss_cls': loss_cls,
            'loss_kl': loss_kl,
            'N_p': 64,
            'emb_dim': 512,
            'proj_dim': 512,
            'num_heads': 4,
            'N_raw': 2048,
            'num_affordance': 17,
            'pairing_num': 2
        }

        data_path = os.path.join(data_dir, setting)

        required_files = ['Point_Train.txt', 'Img_Train.txt', 'Box_Train.txt',
                          'Point_Test.txt', 'Img_Test.txt', 'Box_Test.txt']
        for f in required_files:
            if not os.path.exists(os.path.join(data_path, f)):
                raise FileNotFoundError(f"Data file not found: {os.path.join(data_path, f)}")

        self.state.add_log("Loading training data...")
        train_dataset = PIAD('train', setting,
                             os.path.join(data_path, 'Point_Train.txt'),
                             os.path.join(data_path, 'Img_Train.txt'),
                             os.path.join(data_path, 'Box_Train.txt'),
                             self.config['pairing_num'])
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                       num_workers=4, shuffle=True, drop_last=True)
        self.state.add_log(f"Training samples: {len(train_dataset)}")

        self.state.add_log("Loading validation data...")
        val_dataset = PIAD('val', setting,
                           os.path.join(data_path, 'Point_Test.txt'),
                           os.path.join(data_path, 'Img_Test.txt'),
                           os.path.join(data_path, 'Box_Test.txt'))
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                     num_workers=4, shuffle=True)
        self.state.add_log(f"Validation samples: {len(val_dataset)}")

        self.state.add_log("Initializing model...")
        self.model = get_MyNet(
            pre_train=False,
            N_p=self.config['N_p'],
            emb_dim=self.config['emb_dim'],
            proj_dim=self.config['proj_dim'],
            num_heads=self.config['num_heads'],
            N_raw=self.config['N_raw'],
            num_affordance=self.config['num_affordance']
        )
        self.model = self.model.to(self.device)
        self.state.add_log(f"Model initialized on {self.device}")

        self.criterion_hm = HM_Loss().to(self.device)
        self.criterion_ce = nn.CrossEntropyLoss().to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr,
            betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )

        return True


    def setup(self, setting, data_dir, epochs, batch_size, lr,
          use_gpu=True, loss_cls=0.3, loss_kl=0.5, few_shot=0):
        """Setup training environment
        Args:
            few_shot: 如果是Unseen设置，这个参数表示每个类别的少样本数量
                    0表示不使用few-shot（原始行为），>0表示使用few-shot
        """
        self.device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
        self.state.add_log(f"Using device: {self.device}")
        
        self.config = {
            'Setting': setting,
            'batch_size': batch_size,
            'lr': lr,
            'Epoch': epochs,
            'loss_cls': loss_cls,
            'loss_kl': loss_kl,
            'N_p': 64,
            'emb_dim': 512,
            'proj_dim': 512,
            'num_heads': 4,
            'N_raw': 2048,
            'num_affordance': 17,
            'pairing_num': 2,
            'few_shot': few_shot
        }
        
        data_path = os.path.join(data_dir, setting)
        
        # 检查文件是否存在
        if setting == 'Unseen' and few_shot > 0:
            # Few-shot学习模式：使用测试集进行训练
            self.state.add_log(f"Using Few-shot learning for Unseen dataset ({few_shot} shots per class)")
            
            # 检查测试文件
            test_files = ['Point_Test.txt', 'Img_Test.txt', 'Box_Test.txt']
            for f in test_files:
                if not os.path.exists(os.path.join(data_path, f)):
                    raise FileNotFoundError(f"Test file not found: {os.path.join(data_path, f)}")
            
            # 创建Few-shot数据集
            from data_utils.dataset import PIADUnseenFewShot
            
            self.state.add_log("Creating Few-shot training dataset...")
            train_dataset = PIADUnseenFewShot(
                run_type='train',
                setting_type=setting,
                point_path=os.path.join(data_path, 'Point_Test.txt'),
                img_path=os.path.join(data_path, 'Img_Test.txt'),
                box_path=os.path.join(data_path, 'Box_Test.txt'),
                shot_num=few_shot
            )
            
            self.state.add_log("Creating Few-shot validation dataset...")
            val_dataset = PIADUnseenFewShot(
                run_type='test',
                setting_type=setting,
                point_path=os.path.join(data_path, 'Point_Test.txt'),
                img_path=os.path.join(data_path, 'Img_Test.txt'),
                box_path=os.path.join(data_path, 'Box_Test.txt'),
                shot_num=few_shot
            )
            
        else:
            # 原始训练模式
            required_files = ['Point_Train.txt', 'Img_Train.txt', 'Box_Train.txt',
                            'Point_Test.txt', 'Img_Test.txt', 'Box_Test.txt']
            for f in required_files:
                if not os.path.exists(os.path.join(data_path, f)):
                    raise FileNotFoundError(f"Data file not found: {os.path.join(data_path, f)}")
            
            self.state.add_log("Loading training data...")
            train_dataset = PIAD('train', setting,
                                os.path.join(data_path, 'Point_Train.txt'),
                                os.path.join(data_path, 'Img_Train.txt'),
                                os.path.join(data_path, 'Box_Train.txt'),
                                self.config['pairing_num'])
            
            self.state.add_log("Loading validation data...")
            val_dataset = PIAD('val', setting,
                            os.path.join(data_path, 'Point_Test.txt'),
                            os.path.join(data_path, 'Img_Test.txt'),
                            os.path.join(data_path, 'Box_Test.txt'))
        
        # 创建数据加载器
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                    num_workers=4, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                    num_workers=4, shuffle=True)
        
        self.state.add_log(f"Training samples: {len(train_dataset)}")
        self.state.add_log(f"Validation samples: {len(val_dataset)}")
        
        # 模型初始化等后续代码保持不变...
        self.state.add_log("Initializing model...")
        self.model = get_MyNet(
            pre_train=False,
            N_p=self.config['N_p'],
            emb_dim=self.config['emb_dim'],
            proj_dim=self.config['proj_dim'],
            num_heads=self.config['num_heads'],
            N_raw=self.config['N_raw'],
            num_affordance=self.config['num_affordance']
        )
        self.model = self.model.to(self.device)
        self.state.add_log(f"Model initialized on {self.device}")
        
        self.criterion_hm = HM_Loss().to(self.device)
        self.criterion_ce = nn.CrossEntropyLoss().to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr,
            betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )
        
        return True


    def load_breakpoint(self, breakpoint_path):
        """Load from breakpoint"""
        self.state.add_log(f"Loading breakpoint: {breakpoint_path}")
        checkpoint = torch.load(breakpoint_path, map_location=self.device,weights_only=False)

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

        self.state.current_epoch = checkpoint['epoch'] + 1
        self.state.history = checkpoint.get('history', self.state.history)

        self.state.add_log(f"Resumed from epoch {self.state.current_epoch}")
        return True

    def save_breakpoint(self, epoch):
        """Save checkpoint every 5 epochs"""
        ensure_dir(BREAK_POINT_DIR)
        bp_name = get_breakpoint_name(self.state.model_name, epoch)
        bp_path = os.path.join(BREAK_POINT_DIR, f"{bp_name}.pt")

        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch,
            'history': self.state.history,
            'config': self.config,
            'model_name': self.state.model_name,
            'setting': self.state.setting,
            'log_file_path': self.state.log_file_path
        }

        torch.save(checkpoint, bp_path)
        self.breakpoint_files.append(bp_path)
        self.state.add_log(f"Breakpoint saved: {bp_name}")

    def save_final_model(self):
        """Save final model"""
        ensure_dir(CKPT_DIR)
        model_path = os.path.join(CKPT_DIR, f"{self.state.model_name}.pt")

        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.state.current_epoch,
            'history': self.state.history,
            'config': self.config,
            'model_name': self.state.model_name,
            'setting': self.state.setting,
            'log_file_path': self.state.log_file_path
        }

        torch.save(checkpoint, model_path)
        self.state.add_log(f"Model saved: {self.state.model_name}.pt")

        self.delete_breakpoints()

    def delete_breakpoints(self):
        """Delete all breakpoints for this model"""
        for bp_path in self.breakpoint_files:
            if os.path.exists(bp_path):
                os.remove(bp_path)
                self.state.add_log(f"Deleted breakpoint: {os.path.basename(bp_path)}")
        self.breakpoint_files = []

    def should_stop(self):
        """Check if training should stop"""
        return self._stop_flag.is_set() or self.state.is_stop_requested()

    def stop(self):
        """Request training to stop"""
        self._stop_flag.set()
        self.state.request_stop()
        self.state.add_log("Stop signal sent - training will stop after current batch")

    def train_epoch(self):
        """Train one epoch"""
        self.model.train()
        num_batches = len(self.train_loader)
        loss_sum = 0

        for i, (img, points, labels, logits_labels, sub_box, obj_box) in enumerate(self.train_loader):
            if self.should_stop():
                return None

            self.optimizer.zero_grad()
            temp_loss = 0

            for point, label, logits_label in zip(points, labels, logits_labels):
                point, label = point.float(), label.float()
                label = label.unsqueeze(dim=-1)

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

                temp_loss += loss_hm + self.config['loss_cls'] * loss_ce + self.config['loss_kl'] * loss_kl

            temp_loss.backward()
            self.optimizer.step()
            loss_sum += temp_loss.item()

            if i % 10 == 0:
                self.state.add_log(f"Epoch {self.state.current_epoch} | Batch {i}/{num_batches} | Loss: {temp_loss.item():.4f}")

        mean_loss = loss_sum / (num_batches * self.config['pairing_num'])
        return mean_loss

    def validate(self):
        """Validate the model"""
        self.model.eval()
        val_dataset = self.val_loader.dataset
        results = torch.zeros((len(val_dataset), 2048, 1))
        targets = torch.zeros((len(val_dataset), 2048, 1))

        val_loss_sum = 0
        total_mae = 0
        total_points = 0
        num = 0

        with torch.no_grad():
            for i, (img, point, label, _, _, sub_box, obj_box) in enumerate(self.val_loader):
                if self.should_stop():
                    break

                point, label = point.float(), label.float()
                label = label.unsqueeze(dim=-1)

                img = img.to(self.device)
                point = point.to(self.device)
                label = label.to(self.device)
                sub_box = sub_box.to(self.device)
                obj_box = obj_box.to(self.device)

                _3d, logits, to_KL = self.model(img, point, sub_box, obj_box)

                val_loss_hm = self.criterion_hm(_3d, label)
                val_loss_kl = kl_div(to_KL[0], to_KL[1])
                val_loss = val_loss_hm + self.config['loss_kl'] * val_loss_kl

                mae, point_nums = self._evaluating(_3d, label)
                total_mae += mae.item()
                total_points += point_nums
                val_loss_sum += val_loss.item()

                pred_num = _3d.shape[0]
                results[num:num + pred_num, :, :] = _3d.cpu()
                targets[num:num + pred_num, :, :] = label.cpu()
                num += pred_num

        if total_points == 0:
            return 0, 0, 0, 0, 0

        val_mean_loss = val_loss_sum / len(self.val_loader)
        mean_mae = total_mae / total_points

        results_np = results.numpy()
        targets_np = targets.numpy()

        sim_values = np.zeros(targets_np.shape[0])
        for i in range(targets_np.shape[0]):
            sim_values[i] = SIM(results_np[i], targets_np[i])
        sim = np.mean(sim_values)

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

        return val_mean_loss, auc, iou, sim, mean_mae

    def _evaluating(self, pred, label):
        mae = torch.sum(torch.abs(pred - label), dim=(0, 1))
        points_num = pred.shape[0] * pred.shape[1]
        return mae, points_num

    def save_curves(self):
        """Save training curves"""
        ensure_dir(CKPT_DIR)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        if len(self.state.history['train_loss']) > 0:
            axes[0, 0].plot(self.state.history['train_loss'], label='Train Loss', color='blue')
            if len(self.state.history['val_loss']) > 0:
                axes[0, 0].plot(self.state.history['val_loss'], label='Val Loss', color='red')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Loss Curve')
            axes[0, 0].legend()
            axes[0, 0].grid(True)

        if len(self.state.history['val_auc']) > 0:
            axes[0, 1].plot(self.state.history['val_auc'], label='Val AUC', color='green')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('AUC')
            axes[0, 1].set_title('AUC Curve')
            axes[0, 1].legend()
            axes[0, 1].grid(True)

        if len(self.state.history['val_iou']) > 0:
            axes[1, 0].plot(self.state.history['val_iou'], label='Val IOU', color='orange')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('IOU')
            axes[1, 0].set_title('IOU Curve')
            axes[1, 0].legend()
            axes[1, 0].grid(True)

        if len(self.state.history['val_sim']) > 0:
            axes[1, 1].plot(self.state.history['val_sim'], label='Val SIM', color='purple')
            if len(self.state.history['val_mae']) > 0:
                ax2 = axes[1, 1].twinx()
                ax2.plot(self.state.history['val_mae'], label='Val MAE', color='brown', linestyle='--')
                ax2.set_ylabel('MAE', color='brown')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('SIM')
            axes[1, 1].set_title('SIM and MAE Curves')
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        plt.tight_layout()
        curve_path = os.path.join(CKPT_DIR, f"{self.state.model_name.replace('-model', '')}-loss.png")
        plt.savefig(curve_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.state.add_log(f"Training curves saved: {os.path.basename(curve_path)}")
        return curve_path


def run_training(state, setting, data_dir, epochs, batch_size, lr,
                 use_gpu=True, start_from_breakpoint=None, model_name=None, log_name=None):
    """Run training in a separate thread"""
    backend = TrainerBackend(state)

    # Reset state and initialize
    state.reset()
    state.is_training = True
    state.total_epochs = epochs
    state.setting = setting
    state.model_name = model_name or get_model_name(setting)
    state.start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    state.stop_requested = False

    # Initialize log file
    if not log_name:
        log_name = get_log_name(setting)
    state.init_log_file(log_name)

    try:
        backend.setup(setting, data_dir, epochs, batch_size, lr, use_gpu)

        if start_from_breakpoint and start_from_breakpoint != "Pure":
            backend.load_breakpoint(start_from_breakpoint)

        for epoch in range(state.current_epoch, epochs):
            if backend.should_stop():
                state.add_log("Training stopped by user request")
                break

            state.current_epoch = epoch
            state.lr = backend.optimizer.state_dict()['param_groups'][0]['lr']
            state.add_log(f"\n=== Epoch {epoch + 1}/{epochs} ===")
            state.add_log(f"Learning rate: {state.lr:.6f}")

            train_loss = backend.train_epoch()
            if train_loss is None:
                state.add_log("Training interrupted")
                break

            state.train_loss = train_loss
            state.history['train_loss'].append(train_loss)
            state.add_log(f"Train Loss: {train_loss:.4f}")

            val_loss, auc, iou, sim, mae = backend.validate()
            state.val_loss = val_loss
            state.auc = auc
            state.iou = iou
            state.sim = sim
            state.mae = mae

            state.history['val_loss'].append(val_loss)
            state.history['val_auc'].append(auc)
            state.history['val_iou'].append(iou)
            state.history['val_sim'].append(sim)
            state.history['val_mae'].append(mae)

            state.add_log(f"Val Loss: {val_loss:.4f} | AUC: {auc:.4f} | IOU: {iou:.4f} | SIM: {sim:.4f} | MAE: {mae:.4f}")

            if (epoch + 1) % 5 == 0 and (epoch + 1) < epochs:
                backend.save_breakpoint(epoch + 1)

            backend.scheduler.step()

        if not backend.should_stop():
            state.add_log("\n" + "=" * 50)
            state.add_log("Training completed successfully!")
            backend.save_final_model()
            backend.save_curves()
        else:
            state.add_log("\nTraining stopped - checkpoints preserved")

    except Exception as e:
        state.error_message = str(e)
        state.add_log(f"ERROR: {str(e)}")
        import traceback
        state.add_log(traceback.format_exc())

    finally:
        state.is_training = False
        state.close_log_file()

    return state.to_dict()


def run_few_shot_training(state, setting, data_dir, epochs, batch_size, lr,
                          use_gpu=True, start_from_breakpoint=None, 
                          model_name=None, log_name=None, few_shot=0):
    """Run training with few-shot support"""
    backend = TrainerBackend(state)
    
    # Reset state and initialize
    print("setting:",setting, "few_shot:", few_shot,"epochs:", epochs)
    state.reset()
    state.is_training = True
    state.total_epochs = epochs
    state.setting = setting
    state.model_name = model_name or get_model_name(setting)
    state.start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    state.stop_requested = False
    
    # 添加few-shot信息到模型名称
    if few_shot > 0:
        state.model_name = f"{state.model_name.replace('-model', '')}-fewshot{few_shot}-model"
    
    # Initialize log file
    if not log_name:
        log_name = get_log_name(setting)
        if few_shot > 0:
            log_name = f"{log_name.replace('-log', '')}-fewshot{few_shot}-log"
    state.init_log_file(log_name)
    
    try:
        # 使用修改后的setup方法
        backend.setup(setting, data_dir, epochs, batch_size, lr, use_gpu, few_shot=few_shot)
        
        if start_from_breakpoint and start_from_breakpoint != "Pure":
            backend.load_breakpoint(start_from_breakpoint)

        # 确保从正确的epoch开始
        start_epoch = state.current_epoch

        # 如果当前epoch已经达到或超过总epochs，自动扩展
        if start_epoch >= epochs:
            # 自动扩展20个epochs
            epochs = start_epoch + epochs
            state.total_epochs = epochs
            backend.config['Epoch'] = epochs
            state.add_log(f"Automatically extended training to {epochs} epochs")
        
        for epoch in range(state.current_epoch, epochs):
            if backend.should_stop():
                state.add_log("Training stopped by user request")
                break
            
            state.current_epoch = epoch
            state.lr = backend.optimizer.state_dict()['param_groups'][0]['lr']
            state.add_log(f"\n=== Epoch {epoch + 1}/{epochs} ===")
            state.add_log(f"Learning rate: {state.lr:.6f}")
            if few_shot > 0:
                state.add_log(f"Few-shot setting: {few_shot} samples per class")
            
            train_loss = backend.train_epoch()
            if train_loss is None:
                state.add_log("Training interrupted")
                break
            
            state.train_loss = train_loss
            state.history['train_loss'].append(train_loss)
            state.add_log(f"Train Loss: {train_loss:.4f}")
            
            val_loss, auc, iou, sim, mae = backend.validate()
            state.val_loss = val_loss
            state.auc = auc
            state.iou = iou
            state.sim = sim
            state.mae = mae
            
            state.history['val_loss'].append(val_loss)
            state.history['val_auc'].append(auc)
            state.history['val_iou'].append(iou)
            state.history['val_sim'].append(sim)
            state.history['val_mae'].append(mae)
            
            state.add_log(f"Val Loss: {val_loss:.4f} | AUC: {auc:.4f} | IOU: {iou:.4f} | SIM: {sim:.4f} | MAE: {mae:.4f}")
            
            if (epoch + 1) % 5 == 0 and (epoch + 1) < epochs:
                backend.save_breakpoint(epoch + 1)
            
            backend.scheduler.step()
        
        if not backend.should_stop():
            state.add_log("\n" + "=" * 50)
            state.add_log("Training completed successfully!")
            backend.save_final_model()
            backend.save_curves()
        else:
            state.add_log("\nTraining stopped - checkpoints preserved")
    
    except Exception as e:
        state.error_message = str(e)
        state.add_log(f"ERROR: {str(e)}")
        import traceback
        state.add_log(traceback.format_exc())
    
    finally:
        state.is_training = False
        state.close_log_file()
    
    return state.to_dict()


class InferenceBackend:
    """Backend for inference"""

    def __init__(self):
        self.model = None
        self.device = None
        self.setting = None
        self.affordance_labels = AFFORDANCE_LABELS

    def load_model(self, model_path):
        """Load model for inference"""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(model_path, map_location=self.device,weights_only=False)
        self.setting = checkpoint.get('config', {}).get('Setting', 'Seen')

        self.model = get_MyNet(
            pre_train=False,
            N_p=64, emb_dim=512, proj_dim=512,
            num_heads=4, N_raw=2048, num_affordance=17
        )
        self.model.load_state_dict(checkpoint['model'])
        self.model = self.model.to(self.device)
        self.model.eval()

        return self.setting

    def predict(self, img, point, sub_box, obj_box):
        """Run inference on a single sample"""
        with torch.no_grad():
            if isinstance(point, np.ndarray):
                point = torch.from_numpy(point).float()
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img).float()
            if isinstance(sub_box, np.ndarray):
                sub_box = torch.from_numpy(sub_box).float()
            if isinstance(obj_box, np.ndarray):
                obj_box = torch.from_numpy(obj_box).float()


            img_batch = img.unsqueeze(0).to(self.device)
            point_batch = point.unsqueeze(0).to(self.device)
            sub_box_batch = sub_box.unsqueeze(0).to(self.device)
            obj_box_batch = obj_box.unsqueeze(0).to(self.device)

            pred, _, _ = self.model(img_batch, point_batch, sub_box_batch, obj_box_batch)
            pred = pred.cpu().squeeze().numpy()

        return pred
    def predict_1(self, img, point, sub_box, obj_box):
        """Run inference on a single sample"""
        with torch.no_grad():
            if isinstance(point, np.ndarray):
                point = torch.from_numpy(point).float()
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img).float()
            if isinstance(sub_box, np.ndarray):
                sub_box = torch.from_numpy(sub_box).float()
            if isinstance(obj_box, np.ndarray):
                obj_box = torch.from_numpy(obj_box).float()
            
            img_batch = img.unsqueeze(0).to(self.device)
            point_batch = point.unsqueeze(0).to(self.device)
            sub_box_batch = sub_box.unsqueeze(0).to(self.device)
            obj_box_batch = obj_box.unsqueeze(0).to(self.device)
            
            # 修改：同时返回分割结果和分类结果
            pred, logits, _ = self.model(img_batch, point_batch, sub_box_batch, obj_box_batch)
            
            # 计算softmax概率
            probabilities = F.softmax(logits, dim=1)
            
            # 获取预测的类别索引和概率
            pred_class = torch.argmax(probabilities, dim=1)
            pred_prob = torch.max(probabilities, dim=1).values
            
            return {
                'point_cloud_pred': pred.cpu().squeeze().numpy(),  # 3D点云分割预测
                'class_logits': logits.cpu().squeeze().numpy(),    # 分类logits
                'class_probabilities': probabilities.cpu().squeeze().numpy(),  # 类别概率
                'predicted_class': pred_class.cpu().item(),        # 预测的类别索引
                'predicted_class_name': self.affordance_labels[pred_class.cpu().item()],  # 预测的类别名称
                'confidence': pred_prob.cpu().item()               # 预测置信度
            }
