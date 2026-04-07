"""
Annotation Model for 2D Image Labeling
自动标注模型：识别主体、客体边界框及动作类型

架构设计：
- Backbone: ResNet-50 特征提取
- Detection Head: 检测主体(subject)和客体(object)边界框
- Classification Head: 分类交互动作类型（17种可供性）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.ops import RoIAlign, nms
import numpy as np


class FeaturePyramidNetwork(nn.Module):
    """
    特征金字塔网络 (FPN)
    用于多尺度特征融合
    """
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        
        for in_channels in in_channels_list:
            lateral_conv = nn.Conv2d(in_channels, out_channels, 1)
            output_conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)
    
    def forward(self, features):
        # features: [C3, C4, C5] from backbone
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        
        # 自顶向下融合
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode='nearest'
            )
        
        # 输出特征
        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
        return outputs


class RegionProposalNetwork(nn.Module):
    """
    区域建议网络 (RPN)
    生成候选区域
    """
    def __init__(self, in_channels=256, num_anchors=9):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, 1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, 1)
        
        # 锚点生成：3种尺度 × 3种比例
        self.scales = [32, 64, 128]
        self.ratios = [0.5, 1.0, 2.0]
    
    def forward(self, features):
        # features: P3 from FPN
        x = F.relu(self.conv(features))
        cls_logits = self.cls_logits(x)
        bbox_pred = self.bbox_pred(x)
        
        return cls_logits, bbox_pred
    
    def generate_anchors(self, feature_size, stride=8):
        """生成锚点框"""
        anchors = []
        h, w = feature_size
        
        for i in range(h):
            for j in range(w):
                cx = (j + 0.5) * stride
                cy = (i + 0.5) * stride
                
                for scale in self.scales:
                    for ratio in self.ratios:
                        w_anchor = scale * np.sqrt(ratio)
                        h_anchor = scale / np.sqrt(ratio)
                        anchors.append([
                            cx - w_anchor / 2,
                            cy - h_anchor / 2,
                            cx + w_anchor / 2,
                            cy + h_anchor / 2
                        ])
        
        return torch.tensor(anchors, dtype=torch.float32)


class BBoxHead(nn.Module):
    """
    边界框检测头
    检测主体和客体
    """
    def __init__(self, in_channels=256, num_classes=3):
        # num_classes: background, subject, object
        super().__init__()
        
        # 共享特征层
        self.shared_fc = nn.Sequential(
            nn.Linear(in_channels * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True)
        )
        
        # 分类分支
        self.cls_score = nn.Linear(1024, num_classes)
        
        # 回归分支
        self.bbox_pred = nn.Linear(1024, num_classes * 4)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.shared_fc(x)
        cls_score = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_score, bbox_pred


class InteractionClassifier(nn.Module):
    """
    交互动作分类器
    基于主体-客体对进行动作分类
    """
    def __init__(self, in_channels=256, num_interactions=17):
        super().__init__()
        
        # 主体特征处理
        self.subject_fc = nn.Sequential(
            nn.Linear(in_channels * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # 客体特征处理
        self.object_fc = nn.Sequential(
            nn.Linear(in_channels * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # 空间关系编码
        self.spatial_encoder = nn.Sequential(
            nn.Linear(4, 64),  # 相对位置 [dx, dy, dw, dh]
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True)
        )
        
        # 融合分类
        self.fusion = nn.Sequential(
            nn.Linear(512 + 512 + 128, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_interactions)
        )
    
    def forward(self, subject_feat, object_feat, spatial_info):
        """
        Args:
            subject_feat: 主体ROI特征 [B, C, 7, 7]
            object_feat: 客体ROI特征 [B, C, 7, 7]
            spatial_info: 空间关系 [B, 4] (相对位置)
        """
        s_feat = self.subject_fc(subject_feat.view(subject_feat.size(0), -1))
        o_feat = self.object_fc(object_feat.view(object_feat.size(0), -1))
        spatial = self.spatial_encoder(spatial_info)
        
        combined = torch.cat([s_feat, o_feat, spatial], dim=1)
        logits = self.fusion(combined)
        
        return logits


class AnnotationModel(nn.Module):
    """
    完整的标注模型
    
    输入: RGB图像
    输出: 
        - 主体边界框
        - 客体边界框
        - 动作类型（17种可供性）
    """
    
    def __init__(self, num_interactions=17, pretrained=True):
        super().__init__()
        
        self.num_interactions = num_interactions
        
        # ============ Backbone: ResNet-50 ============
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        
        # 提取各阶段特征
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 1/4, 256ch
        self.layer2 = resnet.layer2  # 1/8, 512ch
        self.layer3 = resnet.layer3  # 1/16, 1024ch
        self.layer4 = resnet.layer4  # 1/32, 2048ch
        
        # ============ FPN ============
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[512, 1024, 2048],  # C3, C4, C5
            out_channels=256
        )
        
        # ============ RPN ============
        self.rpn = RegionProposalNetwork(in_channels=256, num_anchors=9)
        
        # ============ ROI Align ============
        self.roi_align = RoIAlign(
            output_size=(7, 7),
            spatial_scale=1.0 / 8,  # P3尺度
            sampling_ratio=2
        )
        
        # ============ Detection Heads ============
        # 检测主体和客体
        self.bbox_head = BBoxHead(in_channels=256, num_classes=3)
        
        # ============ Interaction Classifier ============
        self.interaction_classifier = InteractionClassifier(
            in_channels=256,
            num_interactions=num_interactions
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化新增层的权重"""
        for m in [self.bbox_head, self.interaction_classifier]:
            for module in m.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, 0, 0.01)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Conv2d):
                    nn.init.normal_(module.weight, 0, 0.01)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    
    def extract_features(self, x):
        """提取backbone特征"""
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # 各阶段
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        
        return c2, c3, c4  # C3, C4, C5 for FPN
    
    def forward(self, images, proposals=None):
        """
        前向传播
        
        Args:
            images: [B, 3, H, W] 输入图像
            proposals: 可选的候选区域 [List of [N, 4] tensors]
                       如果为None，使用RPN生成
        
        Returns:
            训练模式: losses dict
            推理模式: detections dict
        """
        batch_size = images.size(0)
        device = images.device
        
        # ============ Backbone ============
        c3, c4, c5 = self.extract_features(images)
        
        # ============ FPN ============
        features = self.fpn([c3, c4, c5])
        p3 = features[0]  # [B, 256, H/8, W/8]
        
        # ============ RPN ============
        rpn_cls, rpn_bbox = self.rpn(p3)
        
        if proposals is None:
            # 推理模式：生成候选区域
            proposals = self._generate_proposals(rpn_cls, rpn_bbox, p3.shape[-2:])
        
        # ============ ROI Align ============
        # 将proposals转换为ROI格式
        roi_features = []
        for i, props in enumerate(proposals):
            if len(props) > 0:
                # 添加batch index
                batch_idx = torch.full((len(props), 1), i, device=device, dtype=torch.float32)
                rois = torch.cat([batch_idx, props], dim=1)
                roi_feat = self.roi_align(p3, rois)
                roi_features.append(roi_feat)
        
        if len(roi_features) == 0:
            return {
                'boxes': [],
                'scores': [],
                'labels': [],
                'interaction_logits': None
            }
        
        roi_features = torch.cat(roi_features, dim=0)
        
        # ============ Detection Heads ============
        cls_scores, bbox_preds = self.bbox_head(roi_features)
        
        # ============ 推理模式 ============
        if not self.training:
            detections = self._post_process(
                proposals, cls_scores, bbox_preds, 
                images.shape[-2:], device
            )
            
            # 如果检测到主体和客体，进行交互分类
            if len(detections['subject_boxes']) > 0 and len(detections['object_boxes']) > 0:
                interaction_logits = self._classify_interactions(
                    p3, detections, device
                )
                detections['interaction_logits'] = interaction_logits
            else:
                detections['interaction_logits'] = None
            
            return detections
        
        # ============ 训练模式 ============
        return {
            'rpn_cls': rpn_cls,
            'rpn_bbox': rpn_bbox,
            'cls_scores': cls_scores,
            'bbox_preds': bbox_preds,
            'roi_features': roi_features,
            'proposals': proposals
        }
    
    def _generate_proposals(self, rpn_cls, rpn_bbox, feature_size, pre_nms_top_n=1000, post_nms_top_n=300):
        """生成候选区域"""
        batch_size = rpn_cls.size(0)
        device = rpn_cls.device
        
        proposals = []
        anchors = self.rpn.generate_anchors(feature_size).to(device)
        
        for i in range(batch_size):
            # 获取得分
            scores = rpn_cls[i].view(-1)
            scores = torch.sigmoid(scores)
            
            # 获取边界框偏移
            bbox_deltas = rpn_bbox[i].view(-1, 4).permute(1, 0)
            
            # 应用偏移
            proposals_i = self._apply_deltas(anchors, bbox_deltas.T)
            
            # 裁剪到图像范围
            proposals_i = torch.clamp(proposals_i, 0, 224)  # 假设输入224x224
            
            # 选择top-k
            top_n = min(pre_nms_top_n, len(scores))
            top_scores, top_idx = scores.topk(top_n)
            top_proposals = proposals_i[top_idx]
            
            # NMS
            keep = nms(top_proposals, top_scores, 0.7)
            proposals.append(top_proposals[keep][:post_nms_top_n])
        
        return proposals
    
    def _apply_deltas(self, anchors, deltas):
        """应用边界框偏移"""
        # anchors: [N, 4], deltas: [N, 4]
        widths = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        ctr_x = anchors[:, 0] + 0.5 * widths
        ctr_y = anchors[:, 1] + 0.5 * heights
        
        dx = deltas[:, 0]
        dy = deltas[:, 1]
        dw = deltas[:, 2]
        dh = deltas[:, 3]
        
        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights
        
        pred_boxes = torch.stack([
            pred_ctr_x - 0.5 * pred_w,
            pred_ctr_y - 0.5 * pred_h,
            pred_ctr_x + 0.5 * pred_w,
            pred_ctr_y + 0.5 * pred_h
        ], dim=1)
        
        return pred_boxes
    
    def _post_process(self, proposals, cls_scores, bbox_preds, image_size, device, conf_threshold=0.5):
        """后处理：NMS和结果整理"""
        results = {
            'subject_boxes': [],
            'object_boxes': [],
            'subject_scores': [],
            'object_scores': [],
            'all_boxes': [],
            'all_scores': [],
            'all_labels': []
        }
        
        cls_probs = F.softmax(cls_scores, dim=1)
        
        # 分离各批次结果
        offset = 0
        for batch_idx, props in enumerate(proposals):
            n_props = len(props)
            if n_props == 0:
                continue
            
            batch_cls = cls_probs[offset:offset + n_props]
            batch_bbox = bbox_preds[offset:offset + n_props]
            
            # 应用类别特定的边界框回归
            # 只处理前景类别（subject=1, object=2）
            for cls_idx, cls_name in [(1, 'subject'), (2, 'object')]:
                scores = batch_cls[:, cls_idx]
                keep = scores > conf_threshold
                
                if keep.sum() == 0:
                    continue
                
                keep_scores = scores[keep]
                keep_props = props[keep]
                keep_bbox = batch_bbox[keep, cls_idx * 4:(cls_idx + 1) * 4]
                
                # 应用精细边界框偏移
                refined_boxes = self._apply_deltas(keep_props, keep_bbox)
                
                # NMS
                nms_keep = nms(refined_boxes, keep_scores, 0.5)
                
                final_boxes = refined_boxes[nms_keep]
                final_scores = keep_scores[nms_keep]
                
                results[f'{cls_name}_boxes'].append(final_boxes)
                results[f'{cls_name}_scores'].append(final_scores)
                results['all_boxes'].append(final_boxes)
                results['all_scores'].append(final_scores)
                results['all_labels'].append(torch.full((len(final_boxes),), cls_idx, device=device))
            
            offset += n_props
        
        # 合并结果
        for key in ['subject_boxes', 'object_boxes', 'subject_scores', 'object_scores']:
            if results[key]:
                results[key] = torch.cat(results[key], dim=0)
            else:
                results[key] = torch.tensor([], device=device).reshape(0, 4) if 'box' in key else torch.tensor([], device=device)
        
        return results
    
    def _classify_interactions(self, p3, detections, device):
        """对主体-客体对进行交互分类"""
        subject_boxes = detections['subject_boxes']
        object_boxes = detections['object_boxes']
        
        if len(subject_boxes) == 0 or len(object_boxes) == 0:
            return None
        
        # 对每个主体-客体对进行分类
        all_logits = []
        
        for s_box in subject_boxes:
            for o_box in object_boxes:
                # 提取ROI特征
                batch_idx = torch.zeros((2, 1), device=device, dtype=torch.float32)
                s_roi = torch.cat([batch_idx[0:1], s_box.unsqueeze(0)], dim=1)
                o_roi = torch.cat([batch_idx[1:2], o_box.unsqueeze(0)], dim=1)
                
                s_feat = self.roi_align(p3, s_roi)
                o_feat = self.roi_align(p3, o_roi)
                
                # 计算空间关系
                s_center = (s_box[:2] + s_box[2:]) / 2
                o_center = (o_box[:2] + o_box[2:]) / 2
                s_size = s_box[2:] - s_box[:2]
                o_size = o_box[2:] - o_box[:2]
                
                spatial = torch.cat([
                    (o_center - s_center) / (s_size + 1e-6),
                    torch.log(o_size / (s_size + 1e-6) + 1e-6)
                ]).unsqueeze(0)
                
                # 分类
                logits = self.interaction_classifier(s_feat, o_feat, spatial)
                all_logits.append(logits)
        
        if all_logits:
            return torch.cat(all_logits, dim=0)
        return None


def build_annotation_model(num_interactions=17, pretrained=True):
    """构建标注模型"""
    model = AnnotationModel(
        num_interactions=num_interactions,
        pretrained=pretrained
    )
    return model


# ============ 损失函数 ============

class AnnotationLoss(nn.Module):
    """
    标注模型损失函数
    包含：RPN损失 + 检测损失 + 交互分类损失
    """
    def __init__(self):
        super().__init__()
        self.rpn_cls_loss = nn.BCEWithLogitsLoss()
        self.cls_loss = nn.CrossEntropyLoss()
        self.bbox_loss = nn.SmoothL1Loss()
        self.interaction_loss = nn.CrossEntropyLoss()
    
    def forward(self, outputs, targets):
        """
        Args:
            outputs: 模型输出
            targets: 标注目标
                - 'gt_boxes': 真实边界框
                - 'gt_labels': 真实类别 (0=bg, 1=subject, 2=object)
                - 'gt_interactions': 真实交互类别
        """
        losses = {}
        
        # RPN损失
        if 'rpn_cls' in outputs and 'rpn_targets' in targets:
            rpn_cls = outputs['rpn_cls'].view(-1)
            rpn_targets = targets['rpn_targets'].view(-1)
            valid = rpn_targets >= 0
            if valid.sum() > 0:
                losses['rpn_cls_loss'] = self.rpn_cls_loss(
                    rpn_cls[valid], rpn_targets[valid].float()
                )
        
        # 检测损失
        if 'cls_scores' in outputs and 'gt_labels' in targets:
            cls_scores = outputs['cls_scores']
            gt_labels = targets['gt_labels']
            if len(gt_labels) > 0:
                losses['cls_loss'] = self.cls_loss(cls_scores, gt_labels)
        
        if 'bbox_preds' in outputs and 'gt_boxes' in targets and 'gt_labels' in targets:
            bbox_preds = outputs['bbox_preds']
            gt_boxes = targets['gt_boxes']
            gt_labels = targets['gt_labels']
            
            # 只计算前景类别的边界框损失
            fg_mask = gt_labels > 0
            if fg_mask.sum() > 0:
                losses['bbox_loss'] = self.bbox_loss(
                    bbox_preds[fg_mask], gt_boxes[fg_mask]
                )
        
        # 交互分类损失
        if 'interaction_logits' in outputs and outputs['interaction_logits'] is not None:
            if 'gt_interactions' in targets:
                interaction_logits = outputs['interaction_logits']
                gt_interactions = targets['gt_interactions']
                if len(gt_interactions) > 0:
                    losses['interaction_loss'] = self.interaction_loss(
                        interaction_logits, gt_interactions
                    )
        
        # 总损失
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss
        
        return losses


if __name__ == "__main__":
    # 测试模型
    model = build_annotation_model(num_interactions=17)
    model.eval()
    
    # 测试输入
    x = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        output = model(x)
    
    print("Model output keys:", output.keys())
    if output['interaction_logits'] is not None:
        print("Interaction logits shape:", output['interaction_logits'].shape)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
