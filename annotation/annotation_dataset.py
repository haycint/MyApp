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
import re
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


def _default_transform(img_size):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


class AnnotationDataset(Dataset):
    """
    通用标注数据集
    """

    def __init__(
        self,
        root_dir,
        split='train',
        transform=None,
        img_size=(224, 224),
        augment=False
    ):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == 'train')

        self.transform = transform if transform is not None else _default_transform(img_size)
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

        self.data_list = self._load_data_list()
        print(f"Loaded {len(self.data_list)} samples for {split}")

    def _load_data_list(self):
        data_list = []
        images_dir = os.path.join(self.root_dir, 'images')
        annotations_dir = os.path.join(self.root_dir, 'annotations')

        if not os.path.isdir(images_dir) or not os.path.isdir(annotations_dir):
            print(f"Warning: Directory not found - {images_dir} or {annotations_dir}")
            return data_list

        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for img_file in image_files:
            img_path = os.path.join(images_dir, img_file)
            ann_file = os.path.splitext(img_file)[0] + '.json'
            ann_path = os.path.join(annotations_dir, ann_file)
            if os.path.exists(ann_path):
                data_list.append({
                    'image_path': img_path,
                    'annotation_path': ann_path
                })

        random.seed(42)
        random.shuffle(data_list)
        split_idx = int(len(data_list) * 0.8)
        return data_list[:split_idx] if self.split == 'train' else data_list[split_idx:]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        image = Image.open(data['image_path']).convert('RGB')
        original_size = image.size

        with open(data['annotation_path'], 'r', encoding='utf-8') as f:
            annotation = json.load(f)

        subject_box = torch.tensor(annotation['subject_box'], dtype=torch.float32)
        object_box = torch.tensor(annotation['object_box'], dtype=torch.float32)
        interaction = annotation['interaction']

        if isinstance(interaction, str):
            interaction_idx = AFFORDANCE_LABELS.index(interaction)
        else:
            interaction_idx = int(interaction)

        if self.augment:
            image, subject_box, object_box = self._apply_augmentation(image, subject_box, object_box, original_size)

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
        if random.random() > 0.5:
            image = transforms.functional.hflip(image)
            w = original_size[0]
            subject_box[0], subject_box[2] = w - subject_box[2], w - subject_box[0]
            object_box[0], object_box[2] = w - object_box[2], w - object_box[0]
        image = self.augmentation(image)
        return image, subject_box, object_box


class PIADAnnotationDataset(Dataset):
    """
    使用 PIAD 数据集训练标注模型
    """

    def __init__(
        self,
        data_dir,
        setting='Seen',
        split='train',
        img_size=(224, 224),
        augment=False
    ):
        self.data_dir = data_dir
        self.setting = setting
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == 'train')

        self.transform = _default_transform(img_size)
        self.augmentation = transforms.Compose([
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

        self.img_files, self.box_files = self._load_file_lists()
        print(f"Loaded {len(self.img_files)} samples for {setting} {split}")

    def _load_file_lists(self):
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
        files = []
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                files = [line.strip() for line in f if line.strip()]
        return files

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        box_path = self.box_files[idx]
        image = Image.open(img_path).convert('RGB')
        original_size = image.size

        with open(box_path, 'r', encoding='utf-8') as f:
            box_data = json.load(f)

        subject_box = None
        object_box = None
        for shape in box_data.get('shapes', []):
            pts = shape.get('points', [])
            if len(pts) >= 2:
                x1 = min(pts[0][0], pts[1][0])
                y1 = min(pts[0][1], pts[1][1])
                x2 = max(pts[0][0], pts[1][0])
                y2 = max(pts[0][1], pts[1][1])
                if shape.get('label') == 'subject':
                    subject_box = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
                elif shape.get('label') == 'object':
                    object_box = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)

        if subject_box is None:
            subject_box = torch.tensor([0.0, 0.0, 50.0, 50.0], dtype=torch.float32)
        if object_box is None:
            object_box = torch.tensor([50.0, 50.0, 150.0, 150.0], dtype=torch.float32)

        interaction = self._extract_interaction(img_path)

        if self.augment:
            image, subject_box, object_box = self._apply_augmentation(image, subject_box, object_box, original_size)

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
        filename = os.path.basename(path)
        parts = re.split(r'[_\-.]', filename.lower())
        for part in parts:
            if part in [label.lower() for label in AFFORDANCE_LABELS]:
                return AFFORDANCE_LABELS.index(part)
        return 0

    def _apply_augmentation(self, image, subject_box, object_box, original_size):
        if random.random() > 0.5:
            image = transforms.functional.hflip(image)
            w = original_size[0]
            subject_box[0], subject_box[2] = w - subject_box[2], w - subject_box[0]
            object_box[0], object_box[2] = w - object_box[2], w - object_box[0]
        return image, subject_box, object_box


class SyntheticAnnotationDataset(Dataset):
    """
    合成数据集，用于测试和调试
    """

    def __init__(self, num_samples=1000, img_size=(224, 224)):
        self.num_samples = num_samples
        self.img_size = img_size
        self.transform = _default_transform(img_size)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = Image.fromarray(
            np.random.randint(0, 255, (*self.img_size, 3), dtype=np.uint8)
        )
        image = self.transform(image)

        subject_box = torch.tensor([
            random.uniform(10, 80),
            random.uniform(10, 80),
            random.uniform(120, 214),
            random.uniform(120, 214)
        ], dtype=torch.float32)

        object_box = torch.tensor([
            random.uniform(20, 90),
            random.uniform(120, 180),
            random.uniform(140, 210),
            random.uniform(170, 214)
        ], dtype=torch.float32)

        interaction = random.randint(0, len(AFFORDANCE_LABELS) - 1)

        target = {
            'subject_box': subject_box,
            'object_box': object_box,
            'interaction': torch.tensor(interaction, dtype=torch.long),
            'original_size': torch.tensor(self.img_size),
            'image_path': f'synthetic_{idx}.jpg'
        }
        return image, target


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
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
    else:
        dataset = SyntheticAnnotationDataset(
            num_samples=1000,
            img_size=img_size
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )


if __name__ == '__main__':
    dataset = SyntheticAnnotationDataset(num_samples=10)
    print(f'Dataset size: {len(dataset)}')
    image, target = dataset[0]
    print('Image shape:', image.shape)
    print('Subject box:', target['subject_box'])
    print('Object box:', target['object_box'])
    print('Interaction:', target['interaction'])
'''
Path('annotation/annotation_dataset.new.py').write_text(content, encoding='utf-8')'''