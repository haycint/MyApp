"""
3D Object Affordance Grounding from 2D Interactions

Frontend interface with polling update and proper thread management
"""

import streamlit as st
import os
import sys
import time
import threading
import numpy as np
from datetime import datetime
from PIL import Image
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# 在现有的导入下面添加
from backend import run_few_shot_training
# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend import (
    TrainingState, TrainerBackend, InferenceBackend,
    run_training, get_available_models, get_available_breakpoints,
    get_available_logs, find_associated_files, AFFORDANCE_LABELS,
    CKPT_DIR, BREAK_POINT_DIR, LOG_DIR, DATA_DIR
)
from data_utils.dataset import PIADInference
import pandas as pd

file_len=0
# Page config
st.set_page_config(
    page_title="3D Affordance Grounding System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Polling interval in seconds
POLL_INTERVAL = 2


def init_session_state():
    """Initialize session state variables"""
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "overview"
    if 'training_state' not in st.session_state:
        st.session_state.training_state = TrainingState()
    if 'training_thread' not in st.session_state:
        st.session_state.training_thread = None
    if 'trainer_backend' not in st.session_state:
        st.session_state.trainer_backend = None
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'loaded_model_path' not in st.session_state:
        st.session_state.loaded_model_path = None
    if 'inference_backend' not in st.session_state:
        st.session_state.inference_backend = None
    if 'inference_dataset' not in st.session_state:
        st.session_state.inference_dataset = None
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    if 'auto_play' not in st.session_state:
        st.session_state.auto_play = False
    if 'current_log_path' not in st.session_state:
        st.session_state.current_log_path = None
    if 'selected_log_name' not in st.session_state:
        st.session_state.selected_log_name = None
    if 'poll_counter' not in st.session_state:
        st.session_state.poll_counter = 0
    if 'stop_requested' not in st.session_state:
        st.session_state.stop_requested = False

def stop_training():
    """The trigger function of the button [Stop Training]"""
    """Stop training thread properly"""
    # get the Train state
    state = st.session_state.training_state
    # if is training
    if state.is_training:
        # Signal stop through the state (this is shared with the training thread)
        state.request_stop()

        # Wait for thread to finish (with timeout 5s)
        if st.session_state.training_thread and st.session_state.training_thread.is_alive():
            st.session_state.training_thread.join(timeout=5)

        # Update state
        st.session_state.training_state.is_training = False
        st.session_state.training_thread = None
        st.session_state.trainer_backend = None
        st.session_state.stop_requested = True
        global file_len
        file_len = 0  # Reset log length tracking


def get_colors(scores):
    """Get colors for point cloud visualization"""
    # scores展平为1D
    scores = np.array(scores).flatten()
    # 高分为Red
    reference_color = np.array([255, 0, 0])
    # 低分为Grey
    back_color = np.array([190, 190, 190])
    colors = np.zeros((len(scores), 3))
    for i, score in enumerate(scores):
        colors[i] = (reference_color - back_color) * score + back_color
    return colors / 255.0


def visualize_point_cloud(points, pred_scores, gt_scores=None, title="Point Cloud"):
    """Create point cloud visualization figure"""
    # 推理点云显示
    fig = Figure(figsize=(12, 5))

    if gt_scores is not None:
        ax1 = fig.add_subplot(121, projection='3d')
        colors_pred = get_colors(pred_scores)
        ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors_pred, s=8)# s 表示点的大小
        ax1.set_title(f'{title} - Prediction')

        ax2 = fig.add_subplot(122, projection='3d')
        colors_gt = get_colors(gt_scores)
        ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors_gt, s=8)
        ax2.set_title(f'{title} - Ground Truth')
    else:
        ax = fig.add_subplot(111, projection='3d')
        colors_pred = get_colors(pred_scores)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors_pred, s=8)
        ax.set_title(title)

    return fig


def read_log_file(log_path, max_lines=None):
    """Read log file content"""
    if log_path and os.path.exists(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                if max_lines:
                    lines = f.readlines()
                    return ''.join(lines[-max_lines:])
                return f.read()
        except Exception as e:
            return f"Error reading log: {e}"
    return "Not Found"


def render_overview():
    """Render overview section"""
    st.markdown("""
    <h2 id="overview">📖 Overview</h2>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### **What is 3D Affordance Grounding?**
    
    #### **Affordance** refers to the "action possibilities" that objects offer. For example:
    - A **chair** can be "sat on"
    - A **cup** can be "grasped"
    - A **bed** can be "lay on"
    
    ### **The goal of 3D affordance grounding is to identify which parts of a 3D object are suitable for specific interactions.**
    
    #### My design
    
    #### what the model takes as input:
    - **A 3D point cloud** of an object
    - **A 2D image** showing human-object interaction
    
    #### And outputs:
    - **Affordance scores** for each point in the 3D point cloud

    """)

    st.markdown("### Supported Affordances")
    cols = st.columns(6)
    for i, aff in enumerate(AFFORDANCE_LABELS):
        cols[i % 6].metric(aff, "")


def render_training():
    """Render training section with polling"""
    # file_len=0
    st.markdown("""
    <h2 id="training">🏋️ Model Training</h2>
    """, unsafe_allow_html=True)
    state = st.session_state.training_state

    # Sync thread state 同步线程状态
    if st.session_state.training_thread and not st.session_state.training_thread.is_alive():
        if state.is_training:
            state.is_training = False
        st.session_state.training_thread = None

    # Check GPU
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        st.success(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
    else:
        st.warning("⚠️ No GPU detected, training will be slower on CPU")

    # Settings
    col1, col2 = st.columns([2, 1])
    with col1:
        setting = st.radio(
            "Select Dataset Setting",
            ["Seen", "Unseen"],
            index=0,
            help="Seen: train and test on known objects. Unseen: test on novel objects.",
            key="setting_radio"
        )
    with col2:
        epochs = st.number_input("Epochs", min_value=1, max_value=200, value=80, key="epochs_input")
        batch_size = st.selectbox("Batch Size", [4, 8, 16, 32], index=1, key="batch_select")
        lr = st.text_input("Learning Rate", value="0.0001", key="lr_input")

    data_dir = st.text_input("Data Directory", value=DATA_DIR, key="data_dir_input")

    # Check data exists
    data_path = os.path.join(data_dir, setting)
    if os.path.exists(data_path):
        st.success(f"✅ Data directory found: {data_path}")
    else:
        st.warning(f"⚠️ Data directory not found: {data_path}")

    # Breakpoint selection
    st.markdown("### Start Mode")
    breakpoints = get_available_breakpoints(setting)

    col1, col2 = st.columns([1, 1])
    with col1:
        breakpoint_options = ["Pure (Start from scratch)"] + [f"{bp['name']} (Epoch {bp['epoch']})" for bp in breakpoints]
        selected_breakpoint = st.selectbox(
            "Select Training Start Point",
            options=breakpoint_options,
            key="breakpoint_select",
            help="Choose 'Pure' to start fresh, or select a checkpoint to resume training"
        )

    with col2:
        model_name = st.text_input("Model Name (auto-generated if empty)", value="", key="model_name_input")

    # Training controls
    st.markdown("### Training Controls")

    col1, col2 = st.columns([1, 3])
    with col1:
        if state.is_training:
            start_btn = st.button("🚀 Start Training", type="primary", disabled=True)
            stop_btn = st.button("⏹️ Stop Training", type="secondary", key="stop_training_btn")
        else:
            start_btn = st.button("🚀 Start Training", type="primary", key="start_training_btn")
            stop_btn = st.button("⏹️ Stop Training", disabled=True)

    # Handle start training
    if start_btn and not state.is_training:
        if not os.path.exists(data_path):
            st.error("Please ensure the dataset is in the correct location!")
        else:
            # Get breakpoint path
            bp_path = None
            if selected_breakpoint != "Pure (Start from scratch)":
                bp_name = selected_breakpoint.split(" (Epoch")[0]
                for bp in breakpoints:
                    if bp['name'] == bp_name:
                        bp_path = bp['path']
                        break
                
                if not bp_path:
                    st.error(f"Breakpoint file not found: {bp_name}")
                    return
            # Generate model name
            if not model_name:
                now = datetime.now()
                model_name = f"{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}-{setting}-model"

            # Clear stop flag before starting
            state.clear_stop()

            # Start training thread
            def training_thread():
                run_training(
                    state=state,
                    setting=setting,
                    data_dir=data_dir,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=float(lr),
                    use_gpu=use_gpu,
                    start_from_breakpoint=bp_path,
                    model_name=model_name
                )

            thread = threading.Thread(target=training_thread, daemon=True)
            thread.start()
            st.session_state.training_thread = thread
            st.session_state.stop_requested = False

            # Set current log path
            time.sleep(0.3)
            st.session_state.current_log_path = state.log_file_path

            st.rerun()

    # Handle stop training
    if stop_btn and state.is_training:
        stop_training()
        st.success("Training stopped!")
        time.sleep(0.5)
        st.rerun()

    # Log file selection
    st.markdown("### Log Viewer")
    available_logs = get_available_logs()

    # Add current training log if training
    if state.log_file_path:
        current_log_option = f"📊 Current Training: {os.path.basename(state.log_file_path).replace('.txt', '')}"
    else:
        current_log_option = "📊 No Active Training"

    log_options = [current_log_option] + [f"📁 {lg['name']} ({lg['setting']})" for lg in available_logs]

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_log = st.selectbox(
            "Select Log File",
            options=log_options,
            key="log_select",
            help="Select a log file to view"
        )

    with col2:
        refresh_btn = st.button("🔄 Refresh Logs", key="refresh_logs_btn")

    # Update current log path based on selection
    if selected_log.startswith("📊 Current Training") and state.log_file_path:
        st.session_state.current_log_path = state.log_file_path
    elif selected_log.startswith("📁"):
        log_name = selected_log.split(" ")[1]
        matching_log = next((lg for lg in available_logs if lg['name'] == log_name), None)
        if matching_log:
            st.session_state.current_log_path = matching_log['path']

    # Training progress display
    if state.is_training or len(state.history['train_loss']) > 0:
        st.markdown("---")
        st.markdown("### Training Progress")

        # Progress bar
        progress = min(state.current_epoch / state.total_epochs, 1.0) if state.total_epochs > 0 else 0
        st.progress(progress, text=f"Epoch {state.current_epoch}/{state.total_epochs}")

        # Current metrics
        cols = st.columns(6)
        cols[0].metric("Train Loss", f"{state.train_loss:.4f}")
        cols[1].metric("Val Loss", f"{state.val_loss:.4f}")
        cols[2].metric("AUC", f"{state.auc:.4f}")
        cols[3].metric("IOU", f"{state.iou:.4f}")
        cols[4].metric("SIM", f"{state.sim:.4f}")
        cols[5].metric("MAE", f"{state.mae:.4f}")

        # Training curves
        if len(state.history['train_loss']) > 1:
            st.markdown("### Training Curves")

            col1, col2 = st.columns(2)

            with col1:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(state.history['train_loss'], label='Train Loss', color='blue')
                if state.history['val_loss']:
                    ax.plot(state.history['val_loss'], label='Val Loss', color='red')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title('Loss Curve')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
                plt.close()

            with col2:
                fig, ax = plt.subplots(figsize=(8, 4))
                if state.history['val_auc']:
                    ax.plot(state.history['val_auc'], label='AUC', color='green')
                if state.history['val_iou']:
                    ax.plot(state.history['val_iou'], label='IOU', color='orange')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Value')
                ax.set_title('AUC & IOU Curves')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
                plt.close()

    # Training logs from file (scrollable)
    st.markdown("### Training Logs")

    log_content = read_log_file(st.session_state.current_log_path)
    flag=False
    global file_len
    if len(log_content) > file_len:

        file_len = len(log_content)
        flag = True  # 强制刷新状态，确保日志路径更新

    # st.markdown("### Training Logs (Debug Info)")
    # print(f"当前日志路径: `{st.session_state.current_log_path}`")
    st.text_area(
        "Log Output",
        value=log_content[-500:],
        height=300,
        # key="training_logs_area",
        disabled=True
    )
    # print("当前训练状态:", "正在训练" if state.is_training else "未在训练")
    if state.is_training and flag:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="width: 10px; height: 10px; background-color: green; border-radius: 50%; animation: blink 1s infinite;"></div>
            <span>Training in progress... Auto-refreshing every {POLL_INTERVAL}s</span>
        </div>
        <style>
            @keyframes blink {{
                0% {{ opacity: 1; }}
                50% {{ opacity: 0.3; }}
                100% {{ opacity: 1; }}
            }}
        </style>
        """, unsafe_allow_html=True)

        # Polling: wait and rerun
        time.sleep(POLL_INTERVAL)
        st.session_state.poll_counter += 1
        st.rerun()


def render_training_1():
    """Render training section with polling"""
    # file_len=0
    st.markdown("""
    <h2 id="training">🏋️ Model Training</h2>
    """, unsafe_allow_html=True)
    state = st.session_state.training_state

    # Sync thread state 同步线程状态
    if st.session_state.training_thread and not st.session_state.training_thread.is_alive():
        if state.is_training:
            state.is_training = False
        st.session_state.training_thread = None

    # Check GPU
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        st.success(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
    else:
        st.warning("⚠️ No GPU detected, training will be slower on CPU")

    # Settings
    col1, col2 = st.columns([2, 1])
    with col1:
        setting = st.radio(
            "Select Dataset Setting",
            ["Seen", "Unseen"],
            index=0,
            help="Seen: train and test on known objects. Unseen: test on novel objects.",
            key="setting_radio"
        )
    with col2:
        epochs = st.number_input("Epochs", min_value=1, max_value=200, value=80, key="epochs_input")
        batch_size = st.selectbox("Batch Size", [4, 8, 16, 32], index=1, key="batch_select")
        lr = st.text_input("Learning Rate", value="0.0001", key="lr_input")
    
    data_dir = st.text_input("Data Directory", value=DATA_DIR, key="data_dir_input")
    
    
    # 添加Few-shot选项
    st.markdown("### Few-shot Learning")
    if setting == "Unseen":
        few_shot = st.slider(
            "Few-shot samples per class (0 for zero-shot)",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
            help="Number of test samples per affordance to use for training (Unseen only)",
            key="few_shot_slider"
        )
        
        if few_shot > 0:
            st.info(f"🎯 Few-shot learning enabled: {few_shot} samples per affordance will be used for training")
            st.warning("⚠️ In few-shot mode, the test set will be split into training and testing subsets")
    else:
        few_shot = 0
        st.info("🔧 Few-shot learning is only available for Unseen setting")

    # Check data exists
    data_path = os.path.join(data_dir, setting)
    if os.path.exists(data_path):
        st.success(f"✅ Data directory found: {data_path}")
    else:
        st.warning(f"⚠️ Data directory not found: {data_path}")

    # Breakpoint selection
    st.markdown("### Start Mode")
    breakpoints = get_available_breakpoints(setting)

    col1, col2 = st.columns([1, 1])
    with col1:
        breakpoint_options = ["Pure (Start from scratch)"] + [f"{bp['name']} (Epoch {bp['epoch']})" for bp in breakpoints]
        selected_breakpoint = st.selectbox(
            "Select Training Start Point",
            options=breakpoint_options,
            key="breakpoint_select",
            help="Choose 'Pure' to start fresh, or select a checkpoint to resume training"
        )

    with col2:
        model_name = st.text_input("Model Name (auto-generated if empty)", value="", key="model_name_input")

    # Training controls
    st.markdown("### Training Controls")

    col1, col2 = st.columns([1, 3])
    with col1:
        if state.is_training:
            start_btn = st.button("🚀 Start Training", type="primary", disabled=True)
            stop_btn = st.button("⏹️ Stop Training", type="secondary", key="stop_training_btn")
        else:
            start_btn = st.button("🚀 Start Training", type="primary", key="start_training_btn")
            stop_btn = st.button("⏹️ Stop Training", disabled=True)

    # Handle start training
    if start_btn and not state.is_training:
        if not os.path.exists(data_path):
            st.error("Please ensure the dataset is in the correct location!")
        else:
            # Get breakpoint path
            bp_path = None
            if selected_breakpoint != "Pure (Start from scratch)":
                bp_name = selected_breakpoint.split(" (Epoch")[0]
                for bp in breakpoints:
                    if bp['name'] == bp_name:
                        bp_path = bp['path']
                        break
                
                if not bp_path:
                    st.error(f"Breakpoint file not found: {bp_name}")
                    return
            # Generate model name
            if not model_name:
                now = datetime.now()
                model_name = f"{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}-{setting}-model"

            # Clear stop flag before starting
            state.clear_stop()

            # Start training thread
            def training_thread():
                run_few_shot_training(
                    state=state,
                    setting=setting,
                    data_dir=data_dir,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=float(lr),
                    use_gpu=use_gpu,
                    start_from_breakpoint=bp_path,
                    model_name=model_name,
                    few_shot=few_shot
                )

            thread = threading.Thread(target=training_thread, daemon=True)
            thread.start()
            st.session_state.training_thread = thread
            st.session_state.stop_requested = False

            # Set current log path
            time.sleep(0.3)
            st.session_state.current_log_path = state.log_file_path

            st.rerun()

    # Handle stop training
    if stop_btn and state.is_training:
        stop_training()
        st.success("Training stopped!")
        time.sleep(0.5)
        st.rerun()

    # Log file selection
    st.markdown("### Log Viewer")
    available_logs = get_available_logs()

    # Add current training log if training
    if state.log_file_path:
        current_log_option = f"📊 Current Training: {os.path.basename(state.log_file_path).replace('.txt', '')}"
    else:
        current_log_option = "📊 No Active Training"

    log_options = [current_log_option] + [f"📁 {lg['name']} ({lg['setting']})" for lg in available_logs]

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_log = st.selectbox(
            "Select Log File",
            options=log_options,
            key="log_select",
            help="Select a log file to view"
        )

    with col2:
        refresh_btn = st.button("🔄 Refresh Logs", key="refresh_logs_btn")

    # Update current log path based on selection
    if selected_log.startswith("📊 Current Training") and state.log_file_path:
        st.session_state.current_log_path = state.log_file_path
    elif selected_log.startswith("📁"):
        log_name = selected_log.split(" ")[1]
        matching_log = next((lg for lg in available_logs if lg['name'] == log_name), None)
        if matching_log:
            st.session_state.current_log_path = matching_log['path']

    # Training progress display
    if state.is_training or len(state.history['train_loss']) > 0:
        st.markdown("---")
        st.markdown("### Training Progress")

        # Progress bar
        progress = min(state.current_epoch / state.total_epochs, 1.0) if state.total_epochs > 0 else 0
        st.progress(progress, text=f"Epoch {state.current_epoch}/{state.total_epochs}")

        # Current metrics
        cols = st.columns(6)
        cols[0].metric("Train Loss", f"{state.train_loss:.4f}")
        cols[1].metric("Val Loss", f"{state.val_loss:.4f}")
        cols[2].metric("AUC", f"{state.auc:.4f}")
        cols[3].metric("IOU", f"{state.iou:.4f}")
        cols[4].metric("SIM", f"{state.sim:.4f}")
        cols[5].metric("MAE", f"{state.mae:.4f}")

        # Training curves
        if len(state.history['train_loss']) > 1:
            st.markdown("### Training Curves")

            col1, col2 = st.columns(2)

            with col1:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(state.history['train_loss'], label='Train Loss', color='blue')
                if state.history['val_loss']:
                    ax.plot(state.history['val_loss'], label='Val Loss', color='red')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title('Loss Curve')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
                plt.close()

            with col2:
                fig, ax = plt.subplots(figsize=(8, 4))
                if state.history['val_auc']:
                    ax.plot(state.history['val_auc'], label='AUC', color='green')
                if state.history['val_iou']:
                    ax.plot(state.history['val_iou'], label='IOU', color='orange')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Value')
                ax.set_title('AUC & IOU Curves')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
                plt.close()

    # Training logs from file (scrollable)
    st.markdown("### Training Logs")

    log_content = read_log_file(st.session_state.current_log_path)
    flag=False
    global file_len
    if len(log_content) > file_len:

        file_len = len(log_content)
        flag = True  # 强制刷新状态，确保日志路径更新

    # st.markdown("### Training Logs (Debug Info)")
    # print(f"当前日志路径: `{st.session_state.current_log_path}`")
    st.text_area(
        "Log Output",
        value=log_content[-500:],
        height=300,
        # key="training_logs_area",
        disabled=True
    )
    # print("当前训练状态:", "正在训练" if state.is_training else "未在训练")
    if state.is_training and flag:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="width: 10px; height: 10px; background-color: green; border-radius: 50%; animation: blink 1s infinite;"></div>
            <span>Training in progress... Auto-refreshing every {POLL_INTERVAL}s</span>
        </div>
        <style>
            @keyframes blink {{
                0% {{ opacity: 1; }}
                50% {{ opacity: 0.3; }}
                100% {{ opacity: 1; }}
            }}
        </style>
        """, unsafe_allow_html=True)

        # Polling: wait and rerun
        time.sleep(POLL_INTERVAL)
        st.session_state.poll_counter += 1
        st.rerun()



def render_inference():
    """Render inference section"""
    st.markdown("""
    <h2 id="inference">🎨 Effect Demonstration</h2>
    """, unsafe_allow_html=True)

    # Model selection
    st.markdown("### Model Selection")
    models = get_available_models()

    if not models:
        st.warning("⚠️ No trained models found. Please train a model first.")
        st.info("Models should be saved in the `ckpt` directory.")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_model = st.selectbox(
            "Select Model",
            options=[m['name'] for m in models],
            format_func=lambda x: f"{x} ({next(m['size'] for m in models if m['name'] == x):.1f} MB)",
            key="model_select"
        )
    with col2:
        st.markdown(f"**Available Models:** {len(models)}")

    model_path = next(m['path'] for m in models if m['name'] == selected_model)

    # Load model
    if st.session_state.loaded_model_path != model_path:
        st.session_state.model_loaded = False
        st.session_state.loaded_model_path = model_path

    if not st.session_state.model_loaded:
        with st.spinner("Loading model..."):
            try:
                backend = InferenceBackend()
                setting = backend.load_model(model_path)
                st.session_state.inference_backend = backend
                st.session_state.model_setting = setting
                st.session_state.model_loaded = True
                st.session_state.inference_dataset = None  # Reset dataset

                # Find associated files
                log_file, loss_file = find_associated_files(selected_model)
                st.session_state.log_file = log_file
                st.session_state.loss_file = loss_file

                st.success(f"✅ Model loaded successfully! (Setting: {setting})")
            except Exception as e:
                st.error(f"Failed to load model: {str(e)}")
                return

    # Show training curves
    if st.session_state.get('loss_file') and os.path.exists(st.session_state.loss_file):
        with st.expander("View Training Curves", expanded=False):
            st.image(st.session_state.loss_file, width=600)

    # Show log
    if st.session_state.get('log_file') and os.path.exists(st.session_state.log_file):
        with st.expander("View Training Log", expanded=False):
            log_content = read_log_file(st.session_state.log_file)
            st.text_area("Training Log", value=log_content, height=300, disabled=True)

    # Data loading
    st.markdown("### Inference Data")
    data_dir = st.text_input("Data Directory", value=DATA_DIR, key="inference_data_dir")

    setting = st.session_state.get('model_setting', 'Seen')

    # Load dataset
    if st.session_state.inference_dataset is None:
        test_data_path = os.path.join(data_dir, setting)

        if os.path.exists(test_data_path):
            try:
                st.session_state.inference_dataset = PIADInference(
                    point_path=os.path.join(test_data_path, 'Point_Test.txt'),
                    img_path=os.path.join(test_data_path, 'Img_Test.txt'),
                    box_path=os.path.join(test_data_path, 'Box_Test.txt')
                )
                st.success(f"✅ Loaded {len(st.session_state.inference_dataset)} test samples")
            except Exception as e:
                st.error(f"Failed to load dataset: {str(e)}")
        else:
            st.warning(f"⚠️ Data directory not found: {test_data_path}")

    # Inference display
    dataset = st.session_state.inference_dataset
    backend = st.session_state.inference_backend

    if dataset and backend:
        st.markdown("### Inference Display")

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⏭️ Next", type="primary", key="next_btn"):
                st.session_state.current_index = (st.session_state.current_index + 1) % len(dataset)
                st.session_state.auto_play = False
        with col2:
            if st.button("▶️ Continue", key="continue_btn"):
                st.session_state.auto_play = True
        with col3:
            if st.button("⏸️ Pause", key="pause_btn"):
                st.session_state.auto_play = False

        # Get current sample
        idx = st.session_state.current_index
        sample = dataset[idx]

        img, point, label, img_path, point_path, sub_box, obj_box, aff_idx = sample

        # Affordance name
        aff_name = AFFORDANCE_LABELS[aff_idx] if aff_idx < len(AFFORDANCE_LABELS) else "Unknown"

        st.markdown(f"**Sample {idx + 1}/{len(dataset)}** - Affordance: **{aff_name}**")

        # Run inference
        with torch.no_grad():
            pred = backend.predict(img, point, sub_box, obj_box)

        # Display
        col1, col2 = st.columns(2)

        with col1:
            # Convert tensor to image
            img_np = img.permute(1, 2, 0).numpy()
            img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            st.image(img_np, caption=f"Input Image - {aff_name}")

        with col2:
            # Visualize point cloud
            points = point.T
            fig = visualize_point_cloud(points, pred, label, f"Affordance: {aff_name}")
            st.pyplot(fig)

        # Metrics
        pred_flat = pred.flatten()
        label_flat = label.flatten()
        mae = np.mean(np.abs(pred_flat - label_flat))
        sim = np.sum(np.minimum(pred_flat, label_flat)) / (np.sum(pred_flat + label_flat + 1e-8) / 2)

        cols = st.columns(2)
        cols[0].metric("Sample MAE", f"{mae:.4f}")
        cols[1].metric("Sample SIM", f"{sim:.4f}")

        # Auto-play
        if st.session_state.auto_play:
            time.sleep(1)
            st.session_state.current_index = (st.session_state.current_index + 1) % len(dataset)
            st.rerun()


def render_inference_1():
    """Render inference section"""
    st.markdown("""
    <h2 id="inference">🎨 Effect Demonstration</h2>
    """, unsafe_allow_html=True)
    
    # Model selection
    st.markdown("### Model Selection")
    models = get_available_models()
    
    if not models:
        st.warning("⚠️ No trained models found. Please train a model first.")
        st.info("Models should be saved in the `ckpt` directory.")
        return
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_model = st.selectbox(
            "Select Model",
            options=[m['name'] for m in models],
            format_func=lambda x: f"{x} ({next(m['size'] for m in models if m['name'] == x):.1f} MB)",
            key="model_select"
        )
    with col2:
        st.markdown(f"**Available Models:** {len(models)}")
    
    model_path = next(m['path'] for m in models if m['name'] == selected_model)
    
    # Load model
    if st.session_state.loaded_model_path != model_path:
        st.session_state.model_loaded = False
        st.session_state.loaded_model_path = model_path
    
    if not st.session_state.model_loaded:
        with st.spinner("Loading model..."):
            try:
                backend = InferenceBackend()
                setting = backend.load_model(model_path)
                st.session_state.inference_backend = backend
                st.session_state.model_setting = setting
                st.session_state.model_loaded = True
                st.session_state.inference_dataset = None  # Reset dataset
                
                # Find associated files
                log_file, loss_file = find_associated_files(selected_model)
                st.session_state.log_file = log_file
                st.session_state.loss_file = loss_file
                
                st.success(f"✅ Model loaded successfully! (Setting: {setting})")
            except Exception as e:
                st.error(f"Failed to load model: {str(e)}")
                return
    
    # Show training curves
    if st.session_state.get('loss_file') and os.path.exists(st.session_state.loss_file):
        with st.expander("View Training Curves", expanded=False):
            st.image(st.session_state.loss_file, width=600)
    
    # Show log
    if st.session_state.get('log_file') and os.path.exists(st.session_state.log_file):
        with st.expander("View Training Log", expanded=False):
            log_content = read_log_file(st.session_state.log_file)
            st.text_area("Training Log", value=log_content, height=300, disabled=True)
    
    # Data loading
    st.markdown("### Inference Data")
    data_dir = st.text_input("Data Directory", value=DATA_DIR, key="inference_data_dir")
    
    setting = st.session_state.get('model_setting', 'Seen')
    
    # Load dataset
    if st.session_state.inference_dataset is None:
        test_data_path = os.path.join(data_dir, setting)
        if os.path.exists(test_data_path):
            try:
                st.session_state.inference_dataset = PIADInference(
                    point_path=os.path.join(test_data_path, 'Point_Test.txt'),
                    img_path=os.path.join(test_data_path, 'Img_Test.txt'),
                    box_path=os.path.join(test_data_path, 'Box_Test.txt')
                )
                st.success(f"✅ Loaded {len(st.session_state.inference_dataset)} test samples")
            except Exception as e:
                st.error(f"Failed to load dataset: {str(e)}")
        else:
            st.warning(f"⚠️ Data directory not found: {test_data_path}")
    
    # Inference display
    dataset = st.session_state.inference_dataset
    backend = st.session_state.inference_backend
    
    if dataset and backend:
        st.markdown("### Inference Display")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⏭️ Next", type="primary", key="next_btn"):
                st.session_state.current_index = (st.session_state.current_index + 1) % len(dataset)
                st.session_state.auto_play = False
        with col2:
            if st.button("▶️ Continue", key="continue_btn"):
                st.session_state.auto_play = True
        with col3:
            if st.button("⏸️ Pause", key="pause_btn"):
                st.session_state.auto_play = False
        
        # Get current sample
        idx = st.session_state.current_index
        sample = dataset[idx]
        
        img, point, label, img_path, point_path, sub_box, obj_box, aff_idx = sample
        
        # Affordance name
        aff_name = AFFORDANCE_LABELS[aff_idx] if aff_idx < len(AFFORDANCE_LABELS) else "Unknown"
        
        st.markdown(f"**Sample {idx + 1}/{len(dataset)}** - Ground Truth Affordance: **{aff_name}**")
        
        # Run inference
        with torch.no_grad():
            result = backend.predict_1(img, point, sub_box, obj_box)
        
        # 提取结果
        pred = result['point_cloud_pred']
        predicted_class_name = result['predicted_class_name']
        confidence = result['confidence']
        class_probabilities = result['class_probabilities']
        
        # 显示分类预测结果
        st.markdown("### Affordance Classification Prediction")
        
        # 创建两列布局
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Ground Truth", 
                aff_name,
                delta=""
            )
        
        with col2:
            st.metric(
                "Predicted", 
                predicted_class_name,
                delta="✓ Correct" if predicted_class_name == aff_name else "✗ Incorrect",
                delta_color="normal" if predicted_class_name == aff_name else "inverse"
            )
        
        with col3:
            st.metric(
                "Confidence", 
                f"{confidence:.2%}"
            )
        
        # 显示所有类别的概率
        st.markdown("#### All Affordance Probabilities")
        
        # 创建DataFrame显示所有概率
        prob_df = pd.DataFrame({
            'Affordance': AFFORDANCE_LABELS,
            'Probability': class_probabilities
        })
        
        # 按概率排序
        prob_df = prob_df.sort_values('Probability', ascending=False).reset_index(drop=True)
        
        # 高亮显示预测的类别
        def highlight_row(row):
            if row['Affordance'] == predicted_class_name:
                return ['background-color: #e6f7ff'] * len(row)
            elif row['Affordance'] == aff_name:
                return ['background-color: #f0ffe6'] * len(row)
            return [''] * len(row)
        
        # 显示为表格
        st.dataframe(
            prob_df.style.apply(highlight_row, axis=1).format({'Probability': '{:.2%}'}),
            height=400
        )
        
        # 可选：显示概率条形图
        with st.expander("📊 Probability Distribution", expanded=False):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 创建水平条形图
            bars = ax.barh(prob_df['Affordance'], prob_df['Probability'])
            
            # 为预测类别和真实类别设置不同颜色
            for i, bar in enumerate(bars):
                affordance = prob_df.iloc[i]['Affordance']
                if affordance == predicted_class_name:
                    bar.set_color('#1890ff')  # 蓝色表示预测类别
                elif affordance == aff_name:
                    bar.set_color('#52c41a')  # 绿色表示真实类别
            
            ax.set_xlabel('Probability')
            ax.set_xlim(0, 1)
            ax.set_title('Affordance Probability Distribution')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        st.markdown("---")
        
        # 3D点云可视化
        st.markdown("### 3D Affordance Segmentation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Convert tensor to image
            img_np = img.permute(1, 2, 0).numpy()
            img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            st.image(img_np, caption=f"Input Image - {aff_name}")
        
        with col2:
            # Visualize point cloud
            points = point.T
            fig = visualize_point_cloud(points, pred, label, f"Affordance: {aff_name}")
            st.pyplot(fig)
        
        # Segmentation metrics
        pred_flat = pred.flatten()
        label_flat = label.flatten()
        mae = np.mean(np.abs(pred_flat - label_flat))
        sim = np.sum(np.minimum(pred_flat, label_flat)) / (np.sum(pred_flat + label_flat + 1e-8) / 2)
        
        cols = st.columns(2)
        cols[0].metric("Sample MAE", f"{mae:.4f}")
        cols[1].metric("Sample SIM", f"{sim:.4f}")
        
        # Auto-play
        if st.session_state.auto_play:
            time.sleep(1)
            st.session_state.current_index = (st.session_state.current_index + 1) % len(dataset)
            st.rerun()


def render_inference_2():
    """Render inference section"""
    st.markdown("""
    <h2 id="inference">🎨 Effect Demonstration</h2>
    """, unsafe_allow_html=True)
    
    # Model selection
    st.markdown("### Model Selection")
    models = get_available_models()
    
    if not models:
        st.warning("⚠️ No trained models found. Please train a model first.")
        st.info("Models should be saved in the `ckpt` directory.")
        return
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_model = st.selectbox(
            "Select Model",
            options=[m['name'] for m in models],
            format_func=lambda x: f"{x} ({next(m['size'] for m in models if m['name'] == x):.1f} MB)",
            key="model_select"
        )
    with col2:
        st.markdown(f"**Available Models:** {len(models)}")
    
    model_path = next(m['path'] for m in models if m['name'] == selected_model)
    
    # Load model
    if st.session_state.loaded_model_path != model_path:
        st.session_state.model_loaded = False
        st.session_state.loaded_model_path = model_path
    
    if not st.session_state.model_loaded:
        with st.spinner("Loading model..."):
            try:
                backend = InferenceBackend()
                setting = backend.load_model(model_path)
                st.session_state.inference_backend = backend
                st.session_state.model_setting = setting
                st.session_state.model_loaded = True
                st.session_state.inference_dataset = None  # Reset dataset
                
                # Find associated files
                log_file, loss_file = find_associated_files(selected_model)
                st.session_state.log_file = log_file
                st.session_state.loss_file = loss_file
                
                st.success(f"✅ Model loaded successfully! (Setting: {setting})")
            except Exception as e:
                st.error(f"Failed to load model: {str(e)}")
                return
    
    # Show training curves
    if st.session_state.get('loss_file') and os.path.exists(st.session_state.loss_file):
        with st.expander("View Training Curves", expanded=False):
            st.image(st.session_state.loss_file, width=600)
    
    # Show log
    if st.session_state.get('log_file') and os.path.exists(st.session_state.log_file):
        with st.expander("View Training Log", expanded=False):
            log_content = read_log_file(st.session_state.log_file)
            st.text_area("Training Log", value=log_content, height=300, disabled=True)
    
    # Data loading
    st.markdown("### Inference Data")
    data_dir = st.text_input("Data Directory", value=DATA_DIR, key="inference_data_dir")
    
    setting = st.session_state.get('model_setting', 'Seen')
    
    # Load dataset
    if st.session_state.inference_dataset is None:
        test_data_path = os.path.join(data_dir, setting)
        if os.path.exists(test_data_path):
            try:
                st.session_state.inference_dataset = PIADInference(
                    point_path=os.path.join(test_data_path, 'Point_Test.txt'),
                    img_path=os.path.join(test_data_path, 'Img_Test.txt'),
                    box_path=os.path.join(test_data_path, 'Box_Test.txt')
                )
                st.success(f"✅ Loaded {len(st.session_state.inference_dataset)} test samples")
            except Exception as e:
                st.error(f"Failed to load dataset: {str(e)}")
        else:
            st.warning(f"⚠️ Data directory not found: {test_data_path}")
    
    # Inference display
    dataset = st.session_state.inference_dataset
    backend = st.session_state.inference_backend
    
    if dataset and backend:
        st.markdown("### Inference Display")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⏭️ Next", type="primary", key="next_btn"):
                st.session_state.current_index = (st.session_state.current_index + 1) % len(dataset)
                st.session_state.auto_play = False
        with col2:
            if st.button("▶️ Continue", key="continue_btn"):
                st.session_state.auto_play = True
        with col3:
            if st.button("⏸️ Pause", key="pause_btn"):
                st.session_state.auto_play = False
        
        # Get current sample
        idx = st.session_state.current_index
        sample = dataset[idx]
        
        img, point, label, img_path, point_path, sub_box, obj_box, aff_idx = sample
        
        # Affordance name
        aff_name = AFFORDANCE_LABELS[aff_idx] if aff_idx < len(AFFORDANCE_LABELS) else "Unknown"
        
        st.markdown(f"**Sample {idx + 1}/{len(dataset)}** - Ground Truth Affordance: **{aff_name}**")
        
        # Run inference
        with torch.no_grad():
            result = backend.predict_1(img, point, sub_box, obj_box)
        
        # 提取结果
        pred = result['point_cloud_pred']
        predicted_class_name = result['predicted_class_name']
        confidence = result['confidence']
        class_probabilities = result['class_probabilities']
        all_predicted_class_name = result['predicted_class_name']
        
        # 显示分类预测结果
        st.markdown("### Affordance Classification Prediction")
        
        # 创建两列布局
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Ground Truth", 
                aff_name,
                delta=""
            )
        
        with col2:
            st.metric(
                "Predicted", 
                predicted_class_name,
                delta="✓ Correct" if predicted_class_name == aff_name else "✗ Incorrect",
                delta_color="normal" if predicted_class_name == aff_name else "inverse"
            )
        
        with col3:
            st.metric(
                "Confidence", 
                f"{confidence:.2%}"
            )
        
        # 显示所有类别的概率
        st.markdown("#### All Affordance Probabilities")
        
        # 创建DataFrame显示所有概率
        prob_df = pd.DataFrame({
            'Affordance': AFFORDANCE_LABELS,
            'Probability': class_probabilities
        })
        
        # 按概率排序
        prob_df = prob_df.sort_values('Probability', ascending=False).reset_index(drop=True)
        
        # 高亮显示预测的类别
        def highlight_row(row):
            if row['Affordance'] == predicted_class_name:
                return ['background-color: #e6f7ff'] * len(row)
            elif row['Affordance'] == aff_name:
                return ['background-color: #f0ffe6'] * len(row)
            return [''] * len(row)
        
        # 显示为表格
        st.dataframe(
            prob_df.style.apply(highlight_row, axis=1).format({'Probability': '{:.2%}'}),
            height=400
        )
        
        # 可选：显示概率条形图
        with st.expander("📊 Probability Distribution", expanded=False):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 创建水平条形图
            bars = ax.barh(prob_df['Affordance'], prob_df['Probability'])
            
            # 为预测类别和真实类别设置不同颜色
            for i, bar in enumerate(bars):
                affordance = prob_df.iloc[i]['Affordance']
                if affordance == predicted_class_name:
                    bar.set_color('#1890ff')  # 蓝色表示预测类别
                elif affordance == aff_name:
                    bar.set_color('#52c41a')  # 绿色表示真实类别
            
            ax.set_xlabel('Probability')
            ax.set_xlim(0, 1)
            ax.set_title('Affordance Probability Distribution')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        st.markdown("---")
        
        # 3D点云可视化
        st.markdown("### 3D Affordance Segmentation")
        
        # 添加多类别可视化选项
        st.markdown("#### Multi-Affordance Visualization")
        col1, col2 = st.columns(2)
        
        with col1:
            # 选择要显示的前几个类别
            top_k = st.slider(
                "Number of top affordances to visualize",
                min_value=1,
                max_value=5,
                value=3,
                help="Select how many top affordance categories to visualize with different colors"
            )
        
        with col2:
            # 选择可视化模式
            viz_mode = st.radio(
                "Visualization Mode",
                ["Single (Ground Truth)", "Multi-Affordance"],
                index=1,
                help="Single: show only ground truth affordance. Multi: show top-K affordances with different colors."
            )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Convert tensor to image
            img_np = img.permute(1, 2, 0).numpy()
            img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            st.image(img_np, caption=f"Input Image - {aff_name}")
        
        with col2:
            if viz_mode == "Single (Ground Truth)":
                # 原始的单类别可视化
                points = point.T
                fig = visualize_point_cloud(points, pred, label, f"Affordance: {aff_name}")
                st.pyplot(fig)
            else:
                # 新的多类别可视化
                points = point.T
                
                # 获取前K个最可能类别的概率
                top_k_indices = np.argsort(class_probabilities)[-top_k:][::-1]
                top_k_probs = class_probabilities[top_k_indices]
                top_k_names = [AFFORDANCE_LABELS[i] for i in top_k_indices]
                
                st.markdown(f"**Top-{top_k} Affordances:**")
                for i, (name, prob) in enumerate(zip(top_k_names, top_k_probs)):
                    st.write(f"{i+1}. {name}: {prob:.2%}")
                
                # 创建多类别可视化
                fig = visualize_multi_affordance_point_cloud(
                    points, 
                    pred, 
                    top_k_indices, 
                    top_k_probs,
                    top_k_names,
                    title=f"Top-{top_k} Affordances"
                )
                st.pyplot(fig)
                
                # 添加图例说明
                st.markdown("**Color Legend:**")
                colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF']
                legend_cols = st.columns(top_k)
                for i in range(top_k):
                    with legend_cols[i]:
                        st.markdown(
                            f"<div style='background-color:{colors[i%len(colors)]}; "
                            f"color:white; padding:5px; text-align:center; border-radius:5px;'>"
                            f"{top_k_names[i]}</div>", 
                            unsafe_allow_html=True
                        )
        
        # Segmentation metrics
        pred_flat = pred.flatten()
        label_flat = label.flatten()
        mae = np.mean(np.abs(pred_flat - label_flat))
        sim = np.sum(np.minimum(pred_flat, label_flat)) / (np.sum(pred_flat + label_flat + 1e-8) / 2)
        
        cols = st.columns(2)
        cols[0].metric("Sample MAE", f"{mae:.4f}")
        cols[1].metric("Sample SIM", f"{sim:.4f}")
        
        # Auto-play
        if st.session_state.auto_play:
            time.sleep(1)
            st.session_state.current_index = (st.session_state.current_index + 1) % len(dataset)
            st.rerun()


def visualize_multi_affordance_point_cloud(points, pred_scores, affordance_indices, 
                                          affordance_probs, affordance_names, title="Multi-Affordance Point Cloud"):
    """Create point cloud visualization with multiple affordance categories
    
    Args:
        points: 点云坐标 (N, 3)
        pred_scores: 预测得分 (N, 1) 或 (N,)
        affordance_indices: 要显示的可承受性类别索引列表
        affordance_probs: 对应类别的概率
        affordance_names: 对应类别的名称
        title: 图表标题
    
    Returns:
        matplotlib Figure对象
    """
    # 确保pred_scores是正确形状
    if len(pred_scores.shape) == 1:
        pred_scores = pred_scores.reshape(-1, 1)
    
    fig = Figure(figsize=(12, 5))
    
    # 左侧：多类别可视化
    ax1 = fig.add_subplot(121, projection='3d')
    
    # 定义每个类别的颜色
    base_colors = np.array([
        [1.0, 0.0, 0.0],  # 红色 - 第一个类别
        [0.0, 1.0, 0.0],  # 绿色 - 第二个类别
        [0.0, 0.0, 1.0],  # 蓝色 - 第三个类别
        [1.0, 1.0, 0.0],  # 黄色 - 第四个类别
        [1.0, 0.0, 1.0],  # 洋红色 - 第五个类别
    ])
    
    # 为每个点计算混合颜色
    colors = np.zeros((points.shape[0], 3))
    
    # 使用预测得分作为权重，但这里我们简化：使用均匀颜色表示不同区域
    # 实际上，我们需要模型的完整多类别预测，这里用单类别预测近似
    
    # 简单方法：根据预测得分阈值分配颜色
    thresholds = np.linspace(0, 1, len(affordance_indices) + 1)
    
    for i, idx in enumerate(affordance_indices):
        # 选择这个类别得分较高的点
        mask = pred_scores.flatten() >= thresholds[i]
        if i < len(base_colors):
            colors[mask] = base_colors[i]
    
    # 对于那些得分很低的点，使用灰色
    low_score_mask = pred_scores.flatten() < thresholds[0]
    colors[low_score_mask] = [0.7, 0.7, 0.7]  # 灰色
    
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=8)
    ax1.set_title(f'{title} - Prediction')
    
    # 右侧：类别概率分布条形图
    ax2 = fig.add_subplot(122)
    bars = ax2.barh(range(len(affordance_names)), affordance_probs)
    
    # 设置条形颜色
    for i, bar in enumerate(bars):
        if i < len(base_colors):
            bar.set_color(base_colors[i])
    
    ax2.set_yticks(range(len(affordance_names)))
    ax2.set_yticklabels(affordance_names)
    ax2.set_xlabel('Probability')
    ax2.set_xlim(0, 1)
    ax2.set_title('Affordance Probabilities')
    
    # 添加概率值标签
    for i, (bar, prob) in enumerate(zip(bars, affordance_probs)):
        ax2.text(prob + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{prob:.2%}', va='center')
    
    plt.tight_layout()
    return fig

def visualize_enhanced_multi_affordance(points, pred_scores, all_class_probs, 
                                       top_k_indices, top_k_names, title="Multi-Affordance Visualization"):
    """增强版多类别可视化，使用类别概率混合颜色"""
    
    fig = Figure(figsize=(15, 5))
    
    # 左侧：3D点云可视化
    ax1 = fig.add_subplot(131, projection='3d')
    
    # 定义基础颜色
    base_colors = np.array([
        [1.0, 0.0, 0.0],  # 红色
        [0.0, 1.0, 0.0],  # 绿色
        [0.0, 0.0, 1.0],  # 蓝色
        [1.0, 0.5, 0.0],  # 橙色
        [0.5, 0.0, 1.0],  # 紫色
    ])
    
    # 为每个点计算混合颜色
    # 这里我们使用前K个类别的概率作为混合权重
    colors = np.zeros((points.shape[0], 3))
    
    # 简化：使用预测得分和类别概率的乘积作为权重
    for i, class_idx in enumerate(top_k_indices[:5]):  # 最多5个类别
        if i < len(base_colors):
            # 使用该类别概率作为权重
            weight = all_class_probs[class_idx]
            # 将颜色按权重混合
            colors += weight * base_colors[i]
    
    # 归一化颜色
    colors = np.clip(colors, 0, 1)
    
    # 根据预测得分调整亮度
    brightness = pred_scores.flatten().reshape(-1, 1)
    colors = colors * brightness + 0.3 * (1 - brightness)  # 增加对比度
    
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=8)
    ax1.set_title('3D Point Cloud with Multi-Affordance')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # 中间：每个类别的热力图
    ax2 = fig.add_subplot(132)
    
    # 创建类别概率的热力图
    class_probs_matrix = all_class_probs.reshape(1, -1)
    im = ax2.imshow(class_probs_matrix, aspect='auto', cmap='viridis')
    
    ax2.set_xticks(range(len(AFFORDANCE_LABELS)))
    ax2.set_xticklabels(AFFORDANCE_LABELS, rotation=90, fontsize=8)
    ax2.set_yticks([])
    ax2.set_title('All Affordance Probabilities')
    
    # 添加颜色条
    plt.colorbar(im, ax=ax2, shrink=0.8)
    
    # 右侧：Top-K类别雷达图
    ax3 = fig.add_subplot(133, polar=True)
    
    # 准备雷达图数据
    angles = np.linspace(0, 2 * np.pi, len(top_k_indices), endpoint=False).tolist()
    values = all_class_probs[top_k_indices].tolist()
    
    # 闭合图形
    values += values[:1]
    angles += angles[:1]
    
    ax3.plot(angles, values, 'o-', linewidth=2)
    ax3.fill(angles, values, alpha=0.25)
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(top_k_names, fontsize=9)
    ax3.set_ylim(0, 1)
    ax3.set_title('Top Affordances Radar Chart')
    ax3.grid(True)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    return fig

def main():
    """Main application"""
    init_session_state()

    # Header
    st.title("🎯 3D Object Affordance Grounding System")
    st.markdown("""
    *Grounding 3D object affordance from 2D interactions in images*
    """)

    # Navigation - internal links
    st.markdown("""
    <div style="display: flex; gap: 20px; margin-bottom: 20px;">
        <a href="#overview" style="text-decoration: none; padding: 8px 16px; background-color: #f0f2f6; border-radius: 5px;">📖 Overview</a>
        <a href="#inference" style="text-decoration: none; padding: 8px 16px; background-color: #f0f2f6; border-radius: 5px;">🎨 Inference</a>
        <a href="#training" style="text-decoration: none; padding: 8px 16px; background-color: #f0f2f6; border-radius: 5px;">🏋️ Training</a>        
    </div>
    <hr/>
    """, unsafe_allow_html=True)

    # Render sections
    render_overview()
    st.markdown("<hr/>", unsafe_allow_html=True)
    render_inference_2()
    st.markdown("<hr/>", unsafe_allow_html=True)
    render_training_1()
    # Footer

    st.markdown("""
    <hr/>
    <div style="text-align: center; color: #666;">
        <p>Contact me | <a href="2993239432@qq.com">Mail</a></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
