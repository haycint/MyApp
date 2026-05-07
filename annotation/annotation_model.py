"""
Annotation Model for 2D Image Labeling
自动标注模型：识别主体、客体边界框及动作类型
"""

import torch
import torch.nn as nn
from torchvision import models


class BoxPredictor(nn.Module):
    def __init__(self, in_features, hidden=1024):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden, 4)
        )

    def forward(self, features, image_size):
        x = torch.sigmoid(self.fc(features))
        width, height = image_size
        cx = x[:, 0] * width
        cy = x[:, 1] * height
        w = x[:, 2] * width
        h = x[:, 3] * height

        x1 = torch.clamp(cx - w * 0.5, min=0.0, max=width)
        y1 = torch.clamp(cy - h * 0.5, min=0.0, max=height)
        x2 = torch.clamp(cx + w * 0.5, min=0.0, max=width)
        y2 = torch.clamp(cy + h * 0.5, min=0.0, max=height)

        return torch.stack([x1, y1, x2, y2], dim=1)


class AnnotationModel(nn.Module):
    """
    简化的 2D 图像标注模型
    """

    def __init__(self, num_interactions=17, pretrained=True):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.interaction_head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_interactions)
        )
        self.subject_box_head = BoxPredictor(2048, hidden=1024)
        self.object_box_head = BoxPredictor(2048, hidden=1024)
        self.score_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
            nn.Sigmoid()
        )

    def forward(self, images):
        features = self.backbone(images)
        pooled = self.avgpool(features).flatten(1)
        image_size = (images.shape[-1], images.shape[-2])

        subject_boxes = self.subject_box_head(pooled, image_size)
        object_boxes = self.object_box_head(pooled, image_size)
        interaction_logits = self.interaction_head(pooled)
        scores = self.score_head(pooled)

        return {
            'subject_boxes': subject_boxes,
            'object_boxes': object_boxes,
            'subject_scores': scores[:, 0],
            'object_scores': scores[:, 1],
            'interaction_logits': interaction_logits
        }


def build_annotation_model(num_interactions=17, pretrained=True):
    return AnnotationModel(num_interactions=num_interactions, pretrained=pretrained)


class AnnotationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bbox_loss = nn.SmoothL1Loss()
        self.interaction_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        losses = {}
        if 'subject_boxes' in outputs and 'subject_boxes' in targets:
            losses['subject_bbox_loss'] = self.bbox_loss(outputs['subject_boxes'], targets['subject_boxes'])
        if 'object_boxes' in outputs and 'object_boxes' in targets:
            losses['object_bbox_loss'] = self.bbox_loss(outputs['object_boxes'], targets['object_boxes'])
        if 'interaction' in targets and outputs.get('interaction_logits') is not None:
            losses['interaction_loss'] = self.interaction_loss(outputs['interaction_logits'], targets['interaction'])

        total_loss = sum(losses.values()) if losses else torch.tensor(0.0, device=outputs['interaction_logits'].device)
        losses['total_loss'] = total_loss
        return losses


if __name__ == '__main__':
    model = build_annotation_model(num_interactions=17, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print('Output keys:', output.keys())
    print('Subject box shape:', output['subject_boxes'].shape)
    print('Object box shape:', output['object_boxes'].shape)
    print('Interaction logits shape:', output['interaction_logits'].shape)
