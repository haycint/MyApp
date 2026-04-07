<<<<<<< HEAD
# MyApp
=======
# IAGNet: 3D Object Affordance Grounding from 2D Interactions

[![Paper](https://img.shields.io/badge/arXiv-2303.10437-b31b1b.svg)](https://arxiv.org/abs/2303.10437)
[![ICCV 2023](https://img.shields.io/badge/ICCV-2023-blue)](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Grounding_3D_Object_Affordance_from_2D_Interactions_in_Images_ICCV_2023_paper.html)

## Overview

This application implements **IAGNet** (Interaction-driven 3D Affordance Grounding Network), a deep learning model for grounding 3D object affordance from 2D interactions in images.

### What is Affordance Grounding?

**Affordance** refers to the "action possibilities" that objects offer. For example:
- A chair can be "sat on"
- A cup can be "grasped" 
- A bed can be "lay on"

The goal of 3D affordance grounding is to identify which parts of a 3D object are suitable for specific interactions.

## Features

- 🏋️ **Model Training**: Train IAGNet on PIAD dataset (Seen/Unseen settings)
- 🎨 **Effect Demonstration**: Interactive visualization of inference results
- 📊 **Training Monitoring**: Real-time loss and metrics curves
- 💾 **Model Management**: Save/load trained models

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Dataset

### Download PIAD Dataset

Download the PIAD dataset from [Google Drive](https://drive.google.com/drive/folders/1F242TsdXjRZkKQotiBsiN2u6rJAGRZ2W).

### Dataset Structure

Place the dataset in the `Data` directory:

```
Data/
├── Seen/
│   ├── Img_Train.txt
│   ├── Img_Test.txt
│   ├── Point_Train.txt
│   ├── Point_Test.txt
│   ├── Box_Train.txt
│   ├── Box_Test.txt
│   └── (image, point, box files)
├── Unseen/
│   └── (same structure as Seen)
```

## Usage

### Start Application

```bash
cd iagnet_app
streamlit run app.py
```

### Training

1. Select "🏋️ Model Training" from sidebar
2. Choose dataset setting (Seen/Unseen)
3. Set training parameters
4. Click "🚀 Start Training"

### Inference

1. Select "🎨 Effect Demonstration" from sidebar
2. Load a trained model
3. Use "Next", "Continue", "Pause" buttons to control visualization

## Model Architecture

```
IAGNet
├── Image Encoder (ResNet18)
├── Point Encoder (PointNet++)
├── Joint Region Alignment Module
├── Affordance Revealed Module
└── Decoder
```

## Supported Affordances

17 affordance types: grasp, contain, lift, open, lay, sit, support, wrapgrasp, pour, move, display, push, listen, wear, press, cut, stab

## Citation

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

## License

This project is for research purposes only. Please contact the authors for commercial use license.

## References

- Original Paper: https://arxiv.org/abs/2303.10437
- Original Code: https://github.com/yyvhang/IAGNet
>>>>>>> d453872 (my commit)
