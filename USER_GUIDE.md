# IAGNet 使用文档

## 1. 环境配置

### 1.1 安装依赖

```bash
pip install torch torchvision streamlit numpy scikit-learn matplotlib pyyaml pillow
```
或者
```bash
pip install -r requirements.txt
```

### 1.2 验证安装

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import streamlit; print(f'Streamlit version: {streamlit.__version__}')"
```

## 2. 数据集准备

### 2.1 数据集位置

请将PIAD数据集放置在以下目录结构中：

```
Data/
├── Seen/
│   ├── Img_Train.txt      # 训练图像路径列表
│   ├── Img_Test.txt       # 测试图像路径列表
│   ├── Point_Train.txt    # 训练点云路径列表
│   ├── Point_Test.txt     # 测试点云路径列表
│   ├── Box_Train.txt      # 训练边界框路径列表
│   ├── Box_Test.txt       # 测试边界框路径列表
│   ├── images/            # 图像文件目录
│   │   ├── Bag/
│   │   ├── Chair/
│   │   └── ...
│   ├── points/            # 点云文件目录
│   │   ├── Bag/
│   │   ├── Chair/
│   │   └── ...
│   └── boxes/             # 边界框JSON文件目录
│       ├── Bag/
│       ├── Chair/
│       └── ...
├── Unseen/
│   ├── Img_Train.txt
│   ├── Img_Test.txt
│   ├── Point_Train.txt
│   ├── Point_Test.txt
│   ├── Box_Train.txt
│   ├── Box_Test.txt
│   └── ...
```

### 2.2 数据下载

PIAD数据集可从以下链接下载：
- Google Drive: https://drive.google.com/drive/folders/1F242TsdXjRZkKQotiBsiN2u6rJAGRZ2W

### 2.3 数据格式说明

#### 图像文件
- 格式: JPG
- 命名规则: `{Object}_{Affordance}_{Index}.jpg`
- 示例: `Bag_lift_1.jpg`

#### 点云文件
- 格式: TXT
- 每行格式: `index x y z affordance_1 affordance_2 ... affordance_17`
- 示例: `1 0.123 0.456 0.789 1 0 0 ...`

#### 边界框文件
- 格式: JSON (LabelMe格式)
- 包含"subject"(交互主体)和"object"(交互客体)两个边界框

## 3. 运行应用

### 3.1 启动Streamlit应用

```bash
cd /home/z/my-project/iagnet_app
streamlit run app.py
```

### 3.2 访问界面

应用将在浏览器中自动打开，默认地址: http://localhost:8501

## 4. 模型训练

### 4.1 训练步骤

1. 在侧边栏选择"🏋️ Model Training"
2. 选择数据集设置: "Seen" 或 "Unseen"
3. 设置训练参数:
   - Epochs: 训练轮数 (推荐80)
   - Batch Size: 批量大小 (推荐8-16)
   - Learning Rate: 学习率 (推荐0.0001)
4. 点击"🚀 Start Training"开始训练
5. 观察训练进度和指标变化
6. 训练完成后模型自动保存

### 4.2 模型保存

模型保存在 `/ckpt` 目录下，命名格式:
```
{Year}-{Month}-{Day}-{Hour}-{Minute}-{Setting}-model.pt
```
示例: `2026-3-11-11-12-Seen-model.pt`

### 4.3 训练日志和曲线

训练完成后会自动生成:
- 训练日志: `{model_name}-log.txt`
- 损失曲线: `{model_name}-loss.png`

## 5. 效果展示

### 5.1 加载模型

1. 在侧边栏选择"🎨 Effect Demonstration"
2. 从下拉列表选择已训练的模型
3. 模型会自动加载，同时显示对应的训练曲线和日志

### 5.2 推理操作

1. **下一个**: 展示下一个测试样本的推理结果
2. **继续**: 自动连续展示推理结果
3. **暂停**: 停止自动展示

### 5.3 结果展示

每个样本展示:
- 输入图像
- 3D点云可视化 (预测 vs 真值)
- 样本评估指标 (MAE, SIM)

## 6. 可供性类别

IAGNet支持以下17种可供性类别:

| 编号 | 可供性 | 说明 |
|------|--------|------|
| 1 | grasp | 抓握 |
| 2 | contain | 容纳 |
| 3 | lift | 提起 |
| 4 | open | 打开 |
| 5 | lay | 躺卧 |
| 6 | sit | 坐 |
| 7 | support | 支撑 |
| 8 | wrapgrasp | 包裹抓握 |
| 9 | pour | 倾倒 |
| 10 | move | 移动 |
| 11 | display | 展示 |
| 12 | push | 推 |
| 13 | listen | 聆听 |
| 14 | wear | 穿戴 |
| 15 | press | 按压 |
| 16 | cut | 切割 |
| 17 | stab | 刺 |

## 7. 物体类别

### 7.1 Seen设置 (23类)
Earphone, Bag, Chair, Refrigerator, Knife, Dishwasher, Keyboard, Scissors, Table, StorageFurniture, Bottle, Bowl, Microwave, Display, TrashCan, Hat, Clock, Door, Mug, Faucet, Vase, Laptop, Bed

### 7.2 Unseen设置 (17类，其中6类未见)
Knife, Refrigerator, Earphone, Bag, Keyboard, Chair, Hat, Door, TrashCan, Table, Faucet, StorageFurniture, Bottle, Bowl, Display, Mug, Clock

## 8. 常见问题

### Q1: 训练时出现CUDA内存不足
- 减小batch_size
- 使用CPU训练 (较慢)

### Q2: 数据集路径错误
- 检查Data目录下的文件结构
- 确保txt文件中的路径正确

### Q3: 模型加载失败
- 确保模型文件完整
- 检查PyTorch版本兼容性

### Q4: 推理结果不理想
- 增加训练轮数
- 调整学习率
- 检查数据质量

## 9. 联系方式

如有问题，请联系: yyuhang@mail.ustc.edu.cn

## 10. 引用

如果您使用此代码，请引用：

```bibtex
@InProceedings{Yang_2023_ICCV,
    author    = {Yang, Yuhang and Zhai, Wei and Luo, Hongchen and Cao, Yang and Luo, Jiebo and Zha, Zheng-Jun},
    title     = {Grounding 3D Object Affordance from 2D Interactions in Images},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {10905-10915}
}
```
