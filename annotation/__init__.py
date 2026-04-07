"""
Annotation Module for IAGNet Application
图像标注模块

提供自动化的图像标注功能，包括：
- 主体检测
- 客体检测
- 动作类型识别
"""

from .annotation_model import (
    build_annotation_model,
    AnnotationModel,
    AnnotationLoss
)

from .annotation_dataset import (
    AnnotationDataset,
    PIADAnnotationDataset,
    SyntheticAnnotationDataset,
    build_dataloader,
    AFFORDANCE_LABELS
)

__all__ = [
    'build_annotation_model',
    'AnnotationModel',
    'AnnotationLoss',
    'AnnotationDataset',
    'PIADAnnotationDataset',
    'SyntheticAnnotationDataset',
    'build_dataloader',
    'AFFORDANCE_LABELS'
]
