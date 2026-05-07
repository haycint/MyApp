"""
Streamlit-based annotation inference tool
"""

import os
import sys
import yaml
import torch
from PIL import Image
import numpy as np
import streamlit as st
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from annotation_model import build_annotation_model


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_image(image, img_size):
    image = image.convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image), image


def format_box(box, original_size, resized_size):
    ow, oh = original_size
    rw, rh = resized_size
    scale_x = ow / rw
    scale_y = oh / rh
    x1, y1, x2, y2 = box
    return [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]


def draw_boxes(image, boxes, scores, labels=None):
    import cv2
    image_np = np.array(image).copy()
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f'{scores[idx]:.2f}'
        if labels is not None and idx < len(labels):
            text = f'{labels[idx]} {text}'
        cv2.putText(image_np, text, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return Image.fromarray(image_np)


def main():
    st.title('2D Annotation Tool')
    config_path = st.sidebar.text_input('Config path', 'annotation/config_annotation.yaml')
    model_path = st.sidebar.text_input('Model checkpoint', 'ckpt/2026-3-17-2-39-Seen-model.pt')
    img_size = tuple(st.sidebar.number_input('Image size', min_value=64, max_value=1024, value=224, step=32))

    if not os.path.exists(config_path):
        st.warning(f'Config not found: {config_path}')
        return
    config = load_config(config_path)

    model = build_annotation_model(num_interactions=config.get('num_interactions', 17), pretrained=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        st.sidebar.success('Model loaded')
    else:
        st.sidebar.warning('Checkpoint not found, using untrained model')

    image_file = st.file_uploader('Upload image', type=['jpg', 'jpeg', 'png'])
    if image_file is None:
        st.info('Please upload an image to annotate.')
        return

    image = Image.open(image_file).convert('RGB')
    input_tensor, resized_image = load_image(image, img_size)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)

    subject_boxes = outputs['subject_boxes'].cpu().numpy()
    object_boxes = outputs['object_boxes'].cpu().numpy()
    subject_scores = outputs['subject_scores'].cpu().numpy()
    object_scores = outputs['object_scores'].cpu().numpy()
    interaction_logits = outputs['interaction_logits'].cpu().numpy()
    interaction_label = np.argmax(interaction_logits, axis=1)[0]
    interaction_score = np.max(torch.softmax(torch.from_numpy(interaction_logits), dim=1).numpy(), axis=1)[0]

    scaled_subject = format_box(subject_boxes[0], image.size, img_size)
    scaled_object = format_box(object_boxes[0], image.size, img_size)

    annotated = draw_boxes(image, [scaled_subject, scaled_object], [subject_scores[0], object_scores[0]], labels=['subject', 'object'])

    st.image(annotated, caption=f'Predicted interaction {interaction_label} ({interaction_score:.2f})', use_column_width=True)
    st.write('Subject box:', scaled_subject)
    st.write('Object box:', scaled_object)
    st.write('Interaction label:', int(interaction_label))
    st.write('Interaction score:', float(interaction_score))


if __name__ == '__main__':
    main()
