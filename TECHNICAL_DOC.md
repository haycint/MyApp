# IAGNet 技术文档

## 1. 概述

IAGNet (Interaction-driven 3D Affordance Grounding Network) 是一个用于从2D图像交互中定位3D物体可供性的深度学习模型。该模型基于论文《Grounding 3D Object Affordance from 2D Interactions in Images》(ICCV 2023)实现。

## 2. 模型架构

### 2.1 整体架构

IAGNet由以下主要组件组成：

```
输入: 
  - 图像 (B, 3, 224, 224)
  - 点云 (B, 3, 2048)
  - 主体边界框 (B, 4)
  - 客体边界框 (B, 4)

输出:
  - 3D可供性分数 (B, 2048, 1)
  - 可供性分类logits (B, num_affordance)
```

### 2.2 模块详解

#### 2.2.1 图像编码器 (Img_Encoder)
- 基于ResNet18架构
- 输出特征维度: 512
- 特征图大小: 7x7 (输入224x224)

#### 2.2.2 点云编码器 (Point_Encoder)
- 基于PointNet++的多尺度集合抽象层
- 使用三层的MSG (Multi-Scale Grouping)
- 输出层次化特征: [B, 3, 2048] -> [B, 512, 64]

#### 2.2.3 联合区域对齐模块 (Joint_Region_Alignment, JRA)
- 对齐图像特征和点云特征
- 使用双向注意力机制
- 输出联合特征 F_j: [B, N_p + N_i, 512]

#### 2.2.4 可供性揭示模块 (Affordance_Revealed_Module, ARM)
- 使用交叉注意力机制
- 结合主体特征、客体特征和场景特征
- 输出可供性特征 F_alpha

#### 2.2.5 解码器 (Decoder)
- 特征上采样 (Feature Propagation)
- 输出头: Linear -> BatchNorm -> ReLU -> Linear -> Sigmoid
- 分类头: 用于可供性类型分类

## 3. 数据格式

### 3.1 点云数据格式 (.txt文件)
```
index x y z affordance_1 affordance_2 ... affordance_17
```
- index: 点索引
- x, y, z: 点坐标
- affordance_i: 第i个可供性标签 (0或1)

### 3.2 边界框数据格式 (.json文件)
```json
{
  "shapes": [
    {
      "label": "subject",
      "points": [[x1, y1], [x2, y2]]
    },
    {
      "label": "object", 
      "points": [[x1, y1], [x2, y2]]
    }
  ]
}
```

### 3.3 数据列表文件格式 (.txt文件)
每行一个文件路径:
```
/path/to/data/object_affordance_index.jpg
```

## 4. 损失函数

### 4.1 混合损失 (HM_Loss)
结合Focal Loss和Dice Loss:
- Focal Loss: 处理类别不平衡
- Dice Loss: 处理分割任务

### 4.2 交叉熵损失
用于可供性类型分类

### 4.3 KL散度损失
用于特征对齐

### 4.4 总损失
```
L_total = L_hm + 0.3 * L_ce + 0.5 * L_kl
```

## 5. 评估指标

### 5.1 AUC (Area Under ROC Curve)
- 衡量二分类性能
- 计算每个样本的AUC后取平均

### 5.2 IOU (Intersection over Union)
- 使用20个阈值计算平均IOU
- 公式: IOU = Intersection / Union

### 5.3 SIM (Similarity)
- 计算预测和真值的相似度
- 公式: SIM = sum(min(pred, gt)) / sum(pred + gt)

### 5.4 MAE (Mean Absolute Error)
- 平均绝对误差
- 公式: MAE = mean(|pred - gt|)

## 6. 超参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| batch_size | 8 | 批量大小 |
| lr | 0.0001 | 学习率 |
| epochs | 80 | 训练轮数 |
| N_p | 64 | 点云采样点数 |
| emb_dim | 512 | 嵌入维度 |
| proj_dim | 512 | 投影维度 |
| num_heads | 4 | 注意力头数 |
| num_affordance | 17 | 可供性类别数 |
| pairing_num | 2 | 训练时配对数量 |

## 7. 文件结构

```
iagnet_app/
├── app.py              # Streamlit应用主文件
├── train.py            # 训练模块
├── config/
│   ├── config_seen.yaml    # Seen设置配置
│   └── config_unseen.yaml  # Unseen设置配置
├── data_utils/
│   └── dataset.py      # 数据集加载器
├── model/
│   ├── __init__.py
│   ├── iagnet.py       # IAGNet模型
│   └── pointnet2_utils.py  # PointNet++工具
├── utils/
│   ├── loss.py         # 损失函数
│   ├── eval.py         # 评估指标
│   ├── visualization.py # 可视化工具
│   └── utils.py        # 通用工具
└── ckpt/               # 模型检查点目录
```

## 8. 运行要求

- Python >= 3.9
- PyTorch >= 1.13.1
- torchvision >= 0.14.1
- streamlit >= 1.20.0
- numpy >= 1.24.1
- scikit-learn >= 1.2.0
- matplotlib >= 3.6.3
- PyYAML >= 6.0
