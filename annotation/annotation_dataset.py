"""
Annotation Dataset for Training Auto-Labeling Model
标注模型训练数据集

数据格式：
- 图像文件
- 标注文件 (JSON格式，包含主体、客体边界框和动作类型)
"""

import os
import json
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# 可供性类别列表
AFFORDANCE_LABELS = [
    'grasp', 'contain', 'lift', 'open', 'lay', 'sit', 'support',
    'wrapgrasp', 'pour', 'move', 'display', 'push', 'listen',
    'wear', 'press', 'cut', 'stab'
]


class AnnotationDataset(Dataset):
    """
    标注模型训练数据集
    
    数据结构:
    root_dir/
        images/
            img_001.jpg
            img_002.jpg
            ...
        annotations/
            img_001.json
            img_002.json
            ...
    
    JSON格式:
    {
        "image_width": 640,
        "image_height": 480,
        "subject_box": [x1, y1, x2, y2],
        "object_box": [x1, y1, x2, y2],
        "interaction": "grasp"
    }
    """
    
    def __init__(
        self,
        root_dir,
        split='train',
        transform=None,
        img_size=(224, 224),
        augment=False
    ):
        """
        Args:
            root_dir: 数据根目录
            split: 'train' 或 'val'
            transform: 图像变换
            img_size: 输出图像尺寸
            augment: 是否数据增强
        """
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == 'train')
        
        # 图像变换
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        
        # 数据增强变换
        self.augmentation = transforms.Compose([
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
        ])
        
        # 加载数据列表
        self.data_list = self._load_data_list()
        
        print(f"Loaded {len(self.data_list)} samples for {split}")
    
    def _load_data_list(self):
        """加载数据文件列表"""
        data_list = []
        
        images_dir = os.path.join(self.root_dir, 'images')
        annotations_dir = os.path.join(self.root_dir, 'annotations')
        
        if not os.path.exists(images_dir) or not os.path.exists(annotations_dir):
            print(f"Warning: Directory not found - {images_dir} or {annotations_dir}")
            return data_list
        
        # 获取所有图像文件
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in image_files:
            img_path = os.path.join(images_dir, img_file)
            
            # 对应的标注文件
            ann_file = os.path.splitext(img_file)[0] + '.json'
            ann_path = os.path.join(annotations_dir, ann_file)
            
            if os.path.exists(ann_path):
                data_list.append({
                    'image_path': img_path,
                    'annotation_path': ann_path
                })
        
        # 划分训练/验证集
        random.seed(42)
        random.shuffle(data_list)
        split_idx = int(len(data_list) * 0.8)
        
        if self.split == 'train':
            return data_list[:split_idx]
        else:
            return data_list[split_idx:]
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Returns:
            image: [3, H, W] 图像张量
            target: dict
                - 'subject_box': [4] 主体边界框
                - 'object_box': [4] 客体边界框
                - 'interaction': int 动作类别索引
                - 'original_size': [2] 原始图像尺寸
        """
        data = self.data_list[idx]
        
        # ============ 加载图像 ============
        image = Image.open(data['image_path']).convert('RGB')
        original_size = image.size  # (W, H)
        
        # ============ 加载标注 ============
        with open(data['annotation_path'], 'r') as f:
            annotation = json.load(f)
        
        subject_box = torch.tensor(annotation['subject_box'], dtype=torch.float32)
        object_box = torch.tensor(annotation['object_box'], dtype=torch.float32)
        interaction = annotation['interaction']
        
        # 转换交互类型为索引
        if isinstance(interaction, str):
            interaction_idx = AFFORDANCE_LABELS.index(interaction)
        else:
            interaction_idx = interaction
        
        # ============ 数据增强 ============
        if self.augment:
            image, subject_box, object_box = self._apply_augmentation(
                image, subject_box, object_box, original_size
            )
        
        # ============ 缩放边界框 ============
        scale_x = self.img_size[0] / original_size[0]
        scale_y = self.img_size[1] / original_size[1]
        
        subject_box[0] *= scale_x
        subject_box[2] *= scale_x
        subject_box[1] *= scale_y
        subject_box[3] *= scale_y
        
        object_box[0] *= scale_x
        object_box[2] *= scale_x
        object_box[1] *= scale_y
        object_box[3] *= scale_y
        
        # ============ 图像变换 ============
        image = self.transform(image)
        
        target = {
            'subject_box': subject_box,
            'object_box': object_box,
            'interaction': torch.tensor(interaction_idx, dtype=torch.long),
            'original_size': torch.tensor(original_size),
            'image_path': data['image_path']
        }
        
        return image, target
    
    def _apply_augmentation(self, image, subject_box, object_box, original_size):
        """应用数据增强"""
        # 水平翻转
        if random.random() > 0.5:
            image = transforms.functional.hflip(image)
            w = original_size[0]
            subject_box[0], subject_box[2] = w - subject_box[2], w - subject_box[0]
            object_box[0], object_box[2] = w - object_box[2], w - object_box[0]
        
        # 颜色抖动
        image = self.augmentation(image)
        
        return image, subject_box, object_box


class PIADAnnotationDataset(Dataset):
    """
    从PIAD数据集构建标注训练数据集
    
    直接使用现有的PIAD数据格式
    """
    
    def __init__(
        self,
        data_dir,
        setting='Seen',
        split='train',
        img_size=(224, 224),
        augment=False
    ):
        """
        Args:
            data_dir: PIAD数据目录
            setting: 'Seen' 或 'Unseen'
            split: 'train' 或 'test'
            img_size: 输出图像尺寸
            augment: 是否数据增强
        """
        self.data_dir = data_dir
        self.setting = setting
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == 'train')
        
        # 图像变换
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # 加载数据列表
        self.img_files, self.box_files = self._load_file_lists()
        
        print(f"Loaded {len(self.img_files)} samples for {setting} {split}")
    
    def _load_file_lists(self):
        """加载文件列表"""
        setting_dir = os.path.join(self.data_dir, self.setting)
        
        if self.split == 'train':
            img_list_file = os.path.join(setting_dir, 'Img_Train.txt')
            box_list_file = os.path.join(setting_dir, 'Box_Train.txt')
        else:
            img_list_file = os.path.join(setting_dir, 'Img_Test.txt')
            box_list_file = os.path.join(setting_dir, 'Box_Test.txt')
        
        img_files = self._read_file_list(img_list_file)
        box_files = self._read_file_list(box_list_file)
        
        return img_files, box_files
    
    def _read_file_list(self, path):
        """读取文件列表"""
        files = []
        if os.path.exists(path):
            with open(path, 'r') as f:
                files = [line.strip() for line in f.readlines()]
        return files
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        """
        img_path = self.img_files[idx]
        box_path = self.box_files[idx]
        
        # ============ 加载图像 ============
        image = Image.open(img_path).convert('RGB')
        original_size = image.size
        
        # ============ 加载边界框标注 ============
        with open(box_path, 'r') as f:
            box_data = json.load(f)
        
        subject_box = None
        object_box = None
        
        for shape in box_data.get('shapes', []):
            if shape['label'] == 'subject':
                points = shape['points']
                subject_box = torch.tensor([
                    min(points[0][0], points[1][0]),
                    min(points[0][1], points[1][1]),
                    max(points[0][0], points[1][0]),
                    max(points[0][1], points[1][1])
                ], dtype=torch.float32)
            elif shape['label'] == 'object':
                points = shape['points']
                object_box = torch.tensor([
                    min(points[0][0], points[1][0]),
                    min(points[0][1], points[1][1]),
                    max(points[0][0], points[1][0]),
                    max(points[0][1], points[1][1])
                ], dtype=torch.float32)
        
        # 如果没有标注，使用默认值
        if subject_box is None:
            subject_box = torch.tensor([0, 0, 50, 50], dtype=torch.float32)
        if object_box is None:
            object_box = torch.tensor([50, 50, 150, 150], dtype=torch.float32)
        
        # ============ 提取交互类型 ============
        # 从文件名中提取
        interaction = self._extract_interaction(img_path)
        
        # ============ 数据增强 ============
        if self.augment:
            image, subject_box, object_box = self._apply_augmentation(
                image, subject_box, object_box, original_size
            )
        
        # ============ 缩放边界框 ============
        scale_x = self.img_size[0] / original_size[0]
        scale_y = self.img_size[1] / original_size[1]
        
        subject_box[0] *= scale_x
        subject_box[2] *= scale_x
        subject_box[1] *= scale_y
        subject_box[3] *= scale_y
        
        object_box[0] *= scale_x
        object_box[2] *= scale_x
        object_box[1] *= scale_y
        object_box[3] *= scale_y
        
        # ============ 图像变换 ============
        image = self.transform(image)
        
        target = {
            'subject_box': subject_box,
            'object_box': object_box,
            'interaction': torch.tensor(interaction, dtype=torch.long),
            'original_size': torch.tensor(original_size),
            'image_path': img_path
        }
        
        return image, target
    
    def _extract_interaction(self, path):
        """从文件名提取交互类型"""
        filename = os.path.basename(path)
        parts = filename.split('_')
        
        # 假设格式: xxx_objectname_interaction_xxx.jpg
        for i, part in enumerate(parts):
            if part.lower() in [aff.lower() for aff in AFFORDANCE_LABELS]:
                return AFFORDANCE_LABELS.index(part.lower())
        
        # 默认返回0 (grasp)
        return 0
    
    def _apply_augmentation(self, image, subject_box, object_box, original_size):
        """应用数据增强"""
        # 水平翻转
        if random.random() > 0.5:
            image = transforms.functional.hflip(image)
            w = original_size[0]
            subject_box[0], subject_box[2] = w - subject_box[2], w - subject_box[0]
            object_box[0], object_box[2] = w - object_box[2], w - object_box[0]
        
        return image, subject_box, object_box


class SyntheticAnnotationDataset(Dataset):
    """
    合成数据集，用于在没有真实标注数据时进行测试/演示
    """
    
    def __init__(self, num_samples=1000, img_size=(224, 224)):
        self.num_samples = num_samples
        self.img_size = img_size
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 随机生成图像
        image = Image.fromarray(
            np.random.randint(0, 255, (*self.img_size, 3), dtype=np.uint8)
        )
        image = self.transform(image)
        
        # 随机生成边界框
        subject_box = torch.tensor([
            random.uniform(10, 100),
            random.uniform(10, 100),
            random.uniform(100, 200),
            random.uniform(100, 200)
        ], dtype=torch.float32)
        
        object_box = torch.tensor([
            random.uniform(10, 100),
            random.uniform(10, 100),
            random.uniform(100, 200),
            random.uniform(100, 200)
        ], dtype=torch.float32)
        
        # 随机交互类型
        interaction = random.randint(0, 16)
        
        target = {
            'subject_box': subject_box,
            'object_box': object_box,
            'interaction': torch.tensor(interaction, dtype=torch.long),
            'original_size': torch.tensor(self.img_size),
            'image_path': f'synthetic_{idx}'
        }
        
        return image, target


def collate_fn(batch):
    """
    自定义批处理函数
    """
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    images = torch.stack(images, dim=0)
    
    return images, targets


def build_dataloader(
    dataset_type,
    data_dir=None,
    setting='Seen',
    split='train',
    batch_size=8,
    num_workers=4,
    img_size=(224, 224),
    augment=False
):
    """
    构建数据加载器
    
    Args:
        dataset_type: 'piad', 'custom', 或 'synthetic'
        data_dir: 数据目录
        setting: 'Seen' 或 'Unseen'
        split: 'train' 或 'val'/'test'
        batch_size: 批量大小
        num_workers: 工作线程数
        img_size: 图像尺寸
        augment: 是否数据增强
    """
    if dataset_type == 'piad':
        dataset = PIADAnnotationDataset(
            data_dir=data_dir,
            setting=setting,
            split=split,
            img_size=img_size,
            augment=augment
        )
    elif dataset_type == 'custom':
        dataset = AnnotationDataset(
            root_dir=data_dir,
            split=split,
            img_size=img_size,
            augment=augment
        )
    else:  # synthetic
        dataset = SyntheticAnnotationDataset(
            num_samples=1000,
            img_size=img_size
        )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return dataloader


if __name__ == "__main__":
    # 测试合成数据集
    dataset = SyntheticAnnotationDataset(num_samples=100)
    print(f"Dataset size: {len(dataset)}")
    
    image, target = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Subject box: {target['subject_box']}")
    print(f"Object box: {target['object_box']}")
    print(f"Interaction: {target['interaction']}")
