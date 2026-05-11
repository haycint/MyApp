"""
3D Object Affordance Grounding System - Streamlit Frontend (v4.0)

Three-tab interface:
  1. 工作界面 (Work)  — real-time point-cloud / image display, annotation, inference
  2. 训练界面 (Train) — model training with dataset selection / import
  3. 配置界面 (Config)— org / UID / scene-key, memory library, model parallelism

Communication:
  - WebSocket (bidirectional, real-time) via background daemon thread
  - REST API fallback for non-real-time operations
"""

import streamlit as st
import os, sys, time, json, uuid, queue, threading, io, base64, math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
from typing import Optional, List, Dict, Any, Tuple

import requests

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False

try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

from scipy.spatial import KDTree

# =============================================================================
# Configuration
# =============================================================================
API_BASE = os.environ.get("API_BASE", "http://localhost:8000")
POLL_INTERVAL = 2
HEATMAP_STYLES = ["Red-Grey", "Jet", "Hot", "Viridis", "Plasma", "Coolwarm"]
AFFORDANCE_LABELS = [
    "grasp", "contain", "lift", "open", "lay", "sit", "support",
    "wrapgrasp", "pour", "move", "display", "push", "listen",
    "wear", "press", "cut", "stab",
]

st.set_page_config(
    page_title="3D Affordance Grounding",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# WebSocket Client (same as v3, kept for real-time updates)
# =============================================================================

class WSClient:
    """WebSocket client running in a background daemon thread."""

    def __init__(self, base_url: str, org: str, uid: str):
        self.base_url = base_url
        self.org = org
        self.uid = uid
        self.ws = None
        self.connected = False
        self.message_queue: queue.Queue = queue.Queue()
        self._thread = None
        self._connect_time = None
        self._last_error = None

    # ---- lifecycle ----
    def connect(self):
        if self._thread and self._thread.is_alive():
            self.disconnect()
        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = ws_url.rstrip("/") + f"/ws?org={self.org}&uid={self.uid}"
        self._last_error = None
        try:
            self.ws = websocket.WebSocketApp(
                ws_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_close=self._on_close,
                on_error=self._on_error,
            )
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            time.sleep(0.3)
        except Exception as e:
            self._last_error = str(e)
            self.connected = False

    def _run(self):
        try:
            self.ws.run_forever(ping_interval=30, ping_timeout=10)
        except Exception as e:
            self._last_error = str(e)
            self.connected = False

    def disconnect(self):
        self.connected = False
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._thread = None

    def send(self, data: dict) -> bool:
        if self.ws and self.connected:
            try:
                self.ws.send(json.dumps(data))
                return True
            except Exception as e:
                self._last_error = str(e)
        return False

    def drain_messages(self) -> list:
        msgs = []
        while not self.message_queue.empty():
            try:
                msgs.append(self.message_queue.get_nowait())
            except queue.Empty:
                break
        return msgs

    # ---- callbacks ----
    def _on_open(self, ws):
        self.connected = True
        self._connect_time = datetime.now().isoformat()
        self._last_error = None
        self.message_queue.put({"type": "ws_connected", "org": self.org, "uid": self.uid})

    def _on_message(self, ws, message):
        try:
            self.message_queue.put(json.loads(message))
        except json.JSONDecodeError:
            self.message_queue.put({"type": "raw", "data": message})

    def _on_close(self, ws, status, msg):
        self.connected = False
        self.message_queue.put({"type": "ws_disconnected", "close_status": status})

    def _on_error(self, ws, error):
        self._last_error = str(error)
        self.connected = False
        self.message_queue.put({"type": "ws_error", "error": str(error)})


# =============================================================================
# REST API helpers
# =============================================================================

def api_get(endpoint: str, params=None):
    try:
        r = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return None


def api_post(endpoint: str, data=None, files=None):
    try:
        if files:
            r = requests.post(f"{API_BASE}{endpoint}", data=data, files=files, timeout=60)
        else:
            r = requests.post(f"{API_BASE}{endpoint}", json=data, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return None


def api_delete(endpoint: str, data=None):
    """Send a DELETE request to the backend API."""
    try:
        r = requests.delete(f"{API_BASE}{endpoint}", json=data, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return None


def check_backend() -> bool:
    try:
        return requests.get(f"{API_BASE}/api/health", timeout=3).status_code == 200
    except Exception:
        return False


# =============================================================================
# Session state initialisation
# =============================================================================

def init_session_state():
    # ---- identity ----
    if "org_name" not in st.session_state:
        st.session_state.org_name = ""
    if "uid" not in st.session_state:
        st.session_state.uid = ""
    if "scene_keys" not in st.session_state:
        st.session_state.scene_keys: Dict[str, str] = {}  # name -> key
    if "active_scene_key" not in st.session_state:
        st.session_state.active_scene_key = ""

    # ---- ws ----
    if "ws_client" not in st.session_state:
        st.session_state.ws_client = None
    if "ws_connected" not in st.session_state:
        st.session_state.ws_connected = False

    # ---- work interface state ----
    if "point_size" not in st.session_state:
        st.session_state.point_size = 4
    if "heatmap_style" not in st.session_state:
        st.session_state.heatmap_style = "Red-Grey"
    if "sync_rotation" not in st.session_state:
        st.session_state.sync_rotation = True
    if "camera_azimuth" not in st.session_state:
        st.session_state.camera_azimuth = 45.0
    if "camera_elevation" not in st.session_state:
        st.session_state.camera_elevation = 30.0
    if "camera_zoom" not in st.session_state:
        st.session_state.camera_zoom = 1.0
    if "current_pc_original" not in st.session_state:
        st.session_state.current_pc_original = None
    if "current_pc_pred" not in st.session_state:
        st.session_state.current_pc_pred = None
    if "current_images" not in st.session_state:
        st.session_state.current_images = []
    if "memory_pc" not in st.session_state:
        st.session_state.memory_pc = None
    if "work_raw_pc" not in st.session_state:
        st.session_state.work_raw_pc = None
    if "uploaded_images" not in st.session_state:
        st.session_state.uploaded_images = []
    if "annotation_results" not in st.session_state:
        st.session_state.annotation_results = []
    if "annotation_approved" not in st.session_state:
        st.session_state.annotation_approved = False
    if "edit_mode" not in st.session_state:
        st.session_state.edit_mode = False
    if "annotation_type" not in st.session_state:
        st.session_state.annotation_type = "subject"
    if "pc_annotation_range" not in st.session_state:
        st.session_state.pc_annotation_range = 0
    if "pc_annotations" not in st.session_state:
        st.session_state.pc_annotations = np.array([], dtype=int)
    if "pc_additive_annotations" not in st.session_state:
        st.session_state.pc_additive_annotations = np.array([], dtype=int)

    # ---- training state ----
    if "training_status" not in st.session_state:
        st.session_state.training_status = None
    if "ws_log_lines" not in st.session_state:
        st.session_state.ws_log_lines = []
    if "ws_log_lines_max" not in st.session_state:
        st.session_state.ws_log_lines_max = 500

    # ---- config state ----
    if "memory_libraries" not in st.session_state:
        st.session_state.memory_libraries = []
    if "model_parallel" not in st.session_state:
        st.session_state.model_parallel = 1
    if "config_saved" not in st.session_state:
        st.session_state.config_saved = False

    # ---- inference ----
    if "inference_result" not in st.session_state:
        st.session_state.inference_result = None

    # ---- memory viewer state ----
    if "memory_entries_page" not in st.session_state:
        st.session_state.memory_entries_page = 1
    if "memory_selected_entry" not in st.session_state:
        st.session_state.memory_selected_entry = None
    if "memory_filter_affordance" not in st.session_state:
        st.session_state.memory_filter_affordance = ""
    if "memory_pc_points" not in st.session_state:
        st.session_state.memory_pc_points = None
    if "memory_pc_pref" not in st.session_state:
        st.session_state.memory_pc_pref = None
    if "memory_show_preference" not in st.session_state:
        st.session_state.memory_show_preference = True

    # ---- image memory state ----
    if "use_image_memory" not in st.session_state:
        st.session_state.use_image_memory = False
    if "image_memory_object_category" not in st.session_state:
        st.session_state.image_memory_object_category = ""
    if "image_memory_retrieved_images" not in st.session_state:
        st.session_state.image_memory_retrieved_images = []

    # ---- auto-generate UID ----
    if not st.session_state.uid:
        st.session_state.uid = str(uuid.uuid4())[:8]


def get_ws_client() -> Optional[WSClient]:
    if not WEBSOCKET_AVAILABLE:
        return None
    return st.session_state.ws_client


def ensure_ws_connected() -> Optional[WSClient]:
    if not WEBSOCKET_AVAILABLE:
        return None
    ws = st.session_state.ws_client
    org = st.session_state.org_name
    uid = st.session_state.uid
    if ws is None:
        ws = WSClient(API_BASE, org, uid)
        st.session_state.ws_client = ws
        ws.connect()
        return ws
    if ws.org != org or ws.uid != uid:
        ws.disconnect()
        ws = WSClient(API_BASE, org, uid)
        st.session_state.ws_client = ws
        st.session_state.ws_log_lines = []
        st.session_state.training_status = None
        ws.connect()
        return ws
    if not ws.connected:
        ws.connect()
    return ws


def process_ws_messages():
    ws = st.session_state.ws_client
    if ws is None:
        return
    st.session_state.ws_connected = ws.connected
    for msg in ws.drain_messages():
        t = msg.get("type", "")
        if t == "ws_connected":
            st.session_state.ws_connected = True
            ws.send({"type": "get_state"})
        elif t in ("ws_disconnected", "ws_error"):
            st.session_state.ws_connected = False
        elif t in ("training_progress", "training_started", "training_stopped", "training_completed", "scene_update"):
            st.session_state.training_status = msg.get("data", msg)
        elif t == "log_update":
            line = msg.get("data", msg.get("message", ""))
            if line:
                st.session_state.ws_log_lines.append(str(line))
                if len(st.session_state.ws_log_lines) > st.session_state.ws_log_lines_max:
                    st.session_state.ws_log_lines = st.session_state.ws_log_lines[-st.session_state.ws_log_lines_max:]
        elif t == "inference_result":
            st.session_state.inference_result = msg.get("data", msg)
        elif t == "scenes_list":
            st.session_state.available_scenes = msg.get("scenes", [])
        elif t in ("scene_subscribed", "scene_joined"):
            sid = msg.get("scene_id", "")
            if sid:
                st.session_state.subscribed_scenes[sid] = msg.get("scene_info", {})


# =============================================================================
# Heatmap colour helper
# =============================================================================

def scores_to_colors(scores: np.ndarray, style: str = "Red-Grey") -> np.ndarray:
    """Map 1-D scores in [0,1] to (N,3) RGB array."""
    s = np.asarray(scores).flatten()
    if style == "Red-Grey":
        ref = np.array([255, 0, 0], dtype=float)
        bg = np.array([190, 190, 190], dtype=float)
        c = np.outer(s, ref - bg) + bg
        return c / 255.0
    # Use matplotlib colormap for other styles
    cmap_map = {
        "Jet": "jet", "Hot": "hot", "Viridis": "viridis",
        "Plasma": "plasma", "Coolwarm": "coolwarm",
    }
    cmap_name = cmap_map.get(style, "jet")
    cmap = plt.get_cmap(cmap_name)
    return cmap(s)[:, :3]


# =============================================================================
# Plotly 3D point-cloud builder
# =============================================================================

def make_pc_figure(
    points: np.ndarray,
    scores: Optional[np.ndarray] = None,
    title: str = "Point Cloud",
    point_size: int = 4,
    heatmap_style: str = "Red-Grey",
    azimuth: float = 45,
    elevation: float = 30,
    zoom: float = 1.0,
    highlight_indices: Optional[np.ndarray] = None,
):
    """Return a plotly Figure for a 3D scatter point cloud."""
    if points is None or len(points) == 0:
        fig = go.Figure()
        fig.update_layout(title=title, height=350)
        return fig

    pts = np.asarray(points)
    if pts.shape[1] != 3:
        pts = pts.T  # (3,N) -> (N,3)

    # Colours
    if scores is not None:
        colors = scores_to_colors(np.asarray(scores).flatten(), heatmap_style)
        color_vals = np.asarray(scores).flatten()
    else:
        colors = np.full((len(pts), 3), 0.7)
        color_vals = np.zeros(len(pts))

    # Highlight annotated points in green
    if highlight_indices is not None and len(highlight_indices) > 0:
        colors[highlight_indices] = [0.0, 1.0, 0.0]

    fig = go.Figure(
        data=go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(
                size=point_size,
                color=color_vals if scores is not None else "lightgrey",
                colorscale="Jet" if heatmap_style != "Red-Grey" else [[0, "rgb(190,190,190)"], [1, "rgb(255,0,0)"]],
                showscale=scores is not None,
                opacity=0.9,
            ),
            hovertemplate="x:%{x:.3f} y:%{y:.3f} z:%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            camera=dict(
                eye=dict(x=zoom * math.cos(math.radians(elevation)) * math.cos(math.radians(azimuth)),
                         y=zoom * math.cos(math.radians(elevation)) * math.sin(math.radians(azimuth)),
                         z=zoom * math.sin(math.radians(elevation))),
            ),
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True),
            zaxis=dict(showgrid=True),
        ),
        height=350,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


# =============================================================================
# TAB 1 — 工作界面 (Work Interface)
# =============================================================================

def render_work_interface():
    """Render the Work tab with 4 rows."""

    # ---- Row 1: connection status + scene key selector ----
    st.markdown("### 🖥️ 工作界面")
    row1_col1, row1_col2, row1_col3 = st.columns([1, 2, 1])

    with row1_col1:
        backend_ok = check_backend()
        if backend_ok:
            st.markdown("🟢 **已连接到代理**")
        else:
            st.markdown("🔴 **未连接到代理**")

    with row1_col2:
        # Scene key dropdown — fetch from API / config
        scene_key_names = list(st.session_state.scene_keys.keys())
        if not scene_key_names:
            # Try fetching from backend
            resp = api_get("/api/scenes/keys")
            if resp and "keys" in resp:
                for item in resp["keys"]:
                    st.session_state.scene_keys[item["name"]] = item["key"]
                scene_key_names = list(st.session_state.scene_keys.keys())

        options = ["— 选择监控密钥 —"] + scene_key_names + ["➕ 新增密钥…"]
        selected = st.selectbox("监控密钥", options, index=0, key="work_scene_key_select")

        if selected == "➕ 新增密钥…":
            new_name = st.text_input("密钥名称", key="new_key_name")
            new_key = st.text_input("密钥值 (留空自动生成)", key="new_key_value")
            if st.button("添加密钥", key="add_key_btn"):
                if new_name:
                    key_val = new_key.strip() if new_key.strip() else str(uuid.uuid4())[:12]
                    st.session_state.scene_keys[new_name] = key_val
                    # Register with backend
                    api_post("/api/scenes/keys", data={"name": new_name, "key": key_val})
                    st.rerun()
        elif selected != "— 选择监控密钥 —":
            st.session_state.active_scene_key = st.session_state.scene_keys.get(selected, "")

    with row1_col3:
        if st.session_state.active_scene_key:
            st.caption(f"当前密钥: `{st.session_state.active_scene_key}`")

    st.markdown("---")

    # ---- Shared point cloud display settings ----
    with st.expander("⚙️ 点云显示设置", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.session_state.point_size = st.slider("点大小", 1, 15, st.session_state.point_size, key="pc_size")
        with c2:
            st.session_state.heatmap_style = st.selectbox("热力图风格", HEATMAP_STYLES,
                                                          index=HEATMAP_STYLES.index(st.session_state.heatmap_style),
                                                          key="pc_heatmap")
        with c3:
            st.session_state.sync_rotation = st.checkbox("同步旋转", value=st.session_state.sync_rotation, key="pc_sync")
        with c4:
            st.session_state.camera_azimuth = st.slider("方位角", 0, 360,
                                                        int(st.session_state.camera_azimuth), key="cam_az")
            st.session_state.camera_elevation = st.slider("仰角", -90, 90,
                                                          int(st.session_state.camera_elevation), key="cam_el")

    # ---- Row 2: Original point cloud (left) + Prediction (right) ----
    st.markdown("#### 点云对比 — 原始 / 预测")
    row2_l, row2_r = st.columns(2)

    # Fetch current inference data from backend
    inf_data = api_get("/api/inference/current")

    pc_orig = None
    pc_pred_scores = None
    if inf_data:
        if "point_cloud" in inf_data:
            pc_raw = inf_data["point_cloud"]
            if isinstance(pc_raw, list):
                pc_orig = np.array(pc_raw)
            elif isinstance(pc_raw, str):
                # base64 encoded numpy
                try:
                    pc_orig = np.frombuffer(base64.b64decode(pc_raw), dtype=np.float32).reshape(-1, 3)
                except Exception:
                    pc_orig = None
            st.session_state.current_pc_original = pc_orig
        if "pred_scores" in inf_data:
            ps = inf_data["pred_scores"]
            st.session_state.current_pc_pred = np.array(ps) if isinstance(ps, list) else None
            pc_pred_scores = st.session_state.current_pc_pred

    pc_orig = st.session_state.current_pc_original
    pc_pred_scores = st.session_state.current_pc_pred

    az = st.session_state.camera_azimuth
    el = st.session_state.camera_elevation
    ps = st.session_state.point_size
    hs = st.session_state.heatmap_style

    with row2_l:
        fig_orig = make_pc_figure(pc_orig, scores=None, title="原始点云",
                                  point_size=ps, azimuth=az, elevation=el, heatmap_style=hs)
        st.plotly_chart(fig_orig, use_container_width=True, key="pc_orig_plot")

    with row2_r:
        fig_pred = make_pc_figure(pc_orig, scores=pc_pred_scores, title="预测结果",
                                  point_size=ps, azimuth=az, elevation=el, heatmap_style=hs)
        st.plotly_chart(fig_pred, use_container_width=True, key="pc_pred_plot")

    # Navigation controls for inference
    nav_c1, nav_c2, nav_c3, nav_c4 = st.columns(4)
    with nav_c1:
        if st.button("⏮ 上一条", key="inf_prev"):
            api_post("/api/inference/navigate", data={"direction": "prev"})
            st.rerun()
    with nav_c2:
        if st.button("⏭ 下一条", key="inf_next"):
            api_post("/api/inference/navigate", data={"direction": "next"})
            st.rerun()
    with nav_c3:
        if st.button("▶ 自动播放", key="inf_auto"):
            api_post("/api/inference/navigate", data={"direction": "auto"})
    with nav_c4:
        if st.button("⏸ 暂停", key="inf_pause"):
            api_post("/api/inference/navigate", data={"direction": "stop"})

    # ---- Image Memory Enhancement for Inference ----
    img_mem_c1, img_mem_c2 = st.columns([1, 2])
    with img_mem_c1:
        st.session_state.use_image_memory = st.checkbox(
            "使用图片记忆增强", value=st.session_state.use_image_memory, key="use_img_mem_checkbox",
            help="启用后，推理时将使用图片记忆系统进行特征平均增强"
        )
    with img_mem_c2:
        if st.session_state.use_image_memory:
            st.session_state.image_memory_object_category = st.text_input(
                "物体类别", value=st.session_state.image_memory_object_category,
                key="img_mem_obj_cat_input",
                help="输入物体类别以匹配图片记忆 (如: mug, bowl, chair)"
            )
            # Pass image memory config to backend
            try:
                api_post("/api/inference/image-memory-config", data={
                    "use_image_memory": True,
                    "object_category": st.session_state.image_memory_object_category,
                })
            except Exception:
                pass

    st.markdown("---")

    # ---- Row 3: Model reading images (left) + Memory system point cloud (right) ----
    st.markdown("#### 图片读取 / 记忆系统")
    row3_l, row3_r = st.columns(2)

    with row3_l:
        st.markdown("**模型正在读取的图片**")
        # Fetch images from backend
        images_data = api_get("/api/inference/images")
        if images_data and "images" in images_data:
            st.session_state.current_images = images_data["images"]
        # Display images in a horizontally scrollable container
        images = st.session_state.current_images
        if images:
            # Build horizontal scrollable image strip using HTML
            img_html_parts = []
            for idx, img_info in enumerate(images):
                if isinstance(img_info, dict) and "base64" in img_info:
                    img_html_parts.append(
                        f'<img src="data:image/png;base64,{img_info["base64"]}" '
                        f'style="height:200px; margin-right:8px; border-radius:6px; border:1px solid #ccc;" />'
                    )
                elif isinstance(img_info, str):
                    # Assume URL
                    img_html_parts.append(
                        f'<img src="{img_info}" '
                        f'style="height:200px; margin-right:8px; border-radius:6px; border:1px solid #ccc;" />'
                    )
            scrollable_html = f"""
            <div style="overflow-x: auto; white-space: nowrap; padding: 8px 0; border: 1px solid #eee; border-radius: 8px;">
                {''.join(img_html_parts)}
            </div>
            """
            st.markdown(scrollable_html, unsafe_allow_html=True)
        else:
            st.info("暂无图片数据")

    with row3_r:
        st.markdown("**记忆系统点云**")

        # Fetch memory stats
        mem_stats = api_get("/api/config/memory/stats")
        total_mem = mem_stats.get("total_memories", 0) if mem_stats else 0

        if total_mem > 0:
            # Show memory count and controls
            mem_ctrl1, mem_ctrl2, mem_ctrl3 = st.columns([2, 1, 1])
            with mem_ctrl1:
                st.caption(f"记忆库: {total_mem} 条记录")
            with mem_ctrl2:
                st.session_state.memory_show_preference = st.checkbox(
                    "显示偏好", value=st.session_state.memory_show_preference, key="mem_show_pref"
                )
            with mem_ctrl3:
                if st.button("刷新记忆", key="mem_refresh"):
                    st.rerun()

            # Fetch latest memory entries
            mem_entries = api_get("/api/memory/entries", params={
                "page": 1, "per_page": 10, "sort_by": "timestamp", "sort_order": "desc"
            })

            if mem_entries and mem_entries.get("entries"):
                # Dropdown to select a memory entry
                entry_options = []
                for e in mem_entries["entries"]:
                    aff_label = e.get("affordance_label", "?") or "?"
                    outcome = e.get("outcome", "?") or "?"
                    reward = e.get("reward", 0)
                    eid_short = e.get("id", "")[:8]
                    entry_options.append(f"[{eid_short}] {aff_label} | {outcome} | r={reward:.2f}")

                selected_idx = st.selectbox(
                    "选择记忆条目", range(len(entry_options)),
                    format_func=lambda i: entry_options[i],
                    key="mem_entry_select"
                )

                selected_entry_id = mem_entries["entries"][selected_idx]["id"]

                # Fetch point cloud data for selected entry
                pc_data = api_get(f"/api/memory/entry/{selected_entry_id}/pointcloud")

                if pc_data:
                    try:
                        # Decode point cloud
                        pc_b64 = pc_data.get("point_cloud", "")
                        if pc_b64:
                            pc_bytes = base64.b64decode(pc_b64)
                            mem_pc = np.frombuffer(pc_bytes, dtype=np.float32).reshape(-1, 3)
                            st.session_state.memory_pc_points = mem_pc
                        else:
                            mem_pc = st.session_state.memory_pc_points

                        # Decode preference matrix
                        pref_b64 = pc_data.get("preference_matrix", "")
                        pref_scores = None
                        if pref_b64 and st.session_state.memory_show_preference:
                            pref_bytes = base64.b64decode(pref_b64)
                            pref_np = np.frombuffer(pref_bytes, dtype=np.float32)
                            if mem_pc is not None and len(pref_np) == len(mem_pc):
                                pref_scores = pref_np
                                st.session_state.memory_pc_pref = pref_scores
                            elif mem_pc is not None and pref_np.size > 0:
                                # Try reshaping
                                try:
                                    pref_np = pref_np.reshape(-1)
                                    if len(pref_np) == len(mem_pc):
                                        pref_scores = pref_np
                                        st.session_state.memory_pc_pref = pref_scores
                                except Exception:
                                    pass
                    except Exception as e:
                        st.warning(f"解析记忆点云失败: {e}")
                        mem_pc = st.session_state.memory_pc_points
                        pref_scores = st.session_state.memory_pc_pref
                else:
                    mem_pc = st.session_state.memory_pc_points
                    pref_scores = st.session_state.memory_pc_pref

                # Render the memory point cloud
                if mem_pc is not None:
                    fig_mem = make_pc_figure(
                        mem_pc, scores=pref_scores,
                        title=f"记忆点云 — {mem_entries['entries'][selected_idx].get('affordance_label', '')}",
                        point_size=ps, azimuth=az, elevation=el, heatmap_style=hs
                    )
                    st.plotly_chart(fig_mem, use_container_width=True, key="pc_mem_plot")
                else:
                    st.info("选择记忆条目以查看点云")
            else:
                st.info("无法加载记忆条目")
        else:
            st.caption("记忆库为空 — 请在配置界面预填充或启用记忆增强推理")

    st.markdown("---")

    # ---- Row 4: Image upload + annotation (left) / Point cloud annotation (right) ----
    st.markdown("#### 标注工作区")
    row4_l, row4_r = st.columns(2)

    # =============== LEFT: Image upload + annotation ===============
    with row4_l:
        st.markdown("**图片上传与标注**")
        uploaded_files = st.file_uploader(
            "上传图片", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="img_upload"
        )

        if uploaded_files:
            # Display uploaded images
            for uf in uploaded_files:
                img = Image.open(uf)
                st.image(img, caption=uf.name, use_container_width=True)

                # Convert to bytes for backend annotation
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                img_bytes = buf.getvalue()

                # Send to backend for annotation
                if st.button(f"标注: {uf.name}", key=f"annotate_{uf.name}"):
                    with st.spinner("正在标注…"):
                        result = api_post("/api/annotation/image", files={"file": (uf.name, img_bytes, "image/png")})
                        if result:
                            st.session_state.annotation_results.append({
                                "filename": uf.name,
                                "result": result,
                            })
                            st.success("标注完成")
                            st.rerun()

            # Show annotation results
            for ann in st.session_state.annotation_results:
                st.markdown(f"**{ann['filename']} 标注结果:**")
                res = ann["result"]
                if "boxes" in res:
                    for box in res["boxes"]:
                        st.write(f"  - 框: {box.get('label','?')} ({box.get('confidence',0):.2f})")
                if "objects" in res:
                    st.write(f"  物品: {', '.join(res['objects'])}")
                if "actions" in res:
                    st.write(f"  动作: {', '.join(res['actions'])}")

            # Approve button
            approve_col1, approve_col2 = st.columns(2)
            with approve_col1:
                if st.button("✅ 同意标注", key="approve_ann"):
                    st.session_state.annotation_approved = True
                    api_post("/api/annotation/approve", data={"approved": True})
                    st.success("标注已确认")
            with approve_col2:
                if st.button("✏️ 编辑标注", key="edit_ann"):
                    st.session_state.edit_mode = True

            # Edit mode
            if st.session_state.edit_mode and st.session_state.annotation_results:
                st.markdown("**编辑标注**")
                annotation_type = st.selectbox(
                    "标注类型",
                    ["主体 (Subject)", "客体 (Object)", "物体类别 (Object Category)", "动作类别 (Action Category)"],
                    key="ann_type_select",
                )
                st.session_state.annotation_type = annotation_type.split(" ")[0].lower()

                if st.session_state.annotation_type in ("主体", "客体", "subject", "object"):
                    # Bounding box annotation via drawable canvas
                    if CANVAS_AVAILABLE and st.session_state.annotation_results:
                        last_ann = st.session_state.annotation_results[-1]
                        # Try to get the image
                        for uf in uploaded_files:
                            if uf.name == last_ann["filename"]:
                                pil_img = Image.open(uf)
                                canvas_result = st_canvas(
                                    fill_color="rgba(255, 165, 0, 0.3)",
                                    stroke_width=2,
                                    stroke_color="#ff0000",
                                    background_image=pil_img,
                                    drawing_mode="rect",
                                    key=f"canvas_{uf.name}",
                                    height=400,
                                )
                                if canvas_result.json_data is not None:
                                    objects = canvas_result.json_data.get("objects", [])
                                    for obj in objects:
                                        if obj.get("type") == "rect":
                                            left, top = obj.get("left", 0), obj.get("top", 0)
                                            w, h = obj.get("width", 0), obj.get("height", 0)
                                            st.caption(f"框: ({left:.0f},{top:.0f}) {w:.0f}x{h:.0f}")
                                break
                else:
                    # Category input via dropdown/text
                    if "物体" in annotation_type or "object" in st.session_state.annotation_type:
                        obj_cat = st.text_input("物体类别", key="obj_cat_input")
                    if "动作" in annotation_type or "action" in st.session_state.annotation_type:
                        act_cat = st.text_input("动作类别", key="act_cat_input")

                if st.button("保存编辑", key="save_edit_btn"):
                    st.session_state.edit_mode = False
                    st.success("标注已更新")

    # =============== RIGHT: Point cloud annotation ===============
    with row4_r:
        st.markdown("**点云标注**")
        st.caption("左键选取 (重置), 右键追加 | 与上方点云同步旋转")

        # Annotation range control
        st.session_state.pc_annotation_range = st.number_input(
            "标注范围 (0=仅点击点, n=最近n个点)", min_value=0, max_value=100,
            value=st.session_state.pc_annotation_range, key="pc_ann_range",
        )

        # The raw point cloud for annotation (same as Row 2 original)
        ann_pc = st.session_state.current_pc_original or st.session_state.work_raw_pc
        if ann_pc is not None:
            pts = np.asarray(ann_pc)
            if pts.shape[1] != 3:
                pts = pts.T

            # Build KDTree for nearest-neighbor lookup
            kdtree = KDTree(pts)

            # Build annotation figure with current annotations highlighted
            all_highlight = np.concatenate([
                st.session_state.pc_annotations,
                st.session_state.pc_additive_annotations,
            ]).astype(int) if len(st.session_state.pc_additive_annotations) > 0 else st.session_state.pc_annotations

            fig_ann = make_pc_figure(
                pts, scores=None, title="点云标注 (点击选取)",
                point_size=ps, azimuth=az, elevation=el,
                highlight_indices=all_highlight if len(all_highlight) > 0 else None,
            )
            # Enable click events
            fig_ann.update_layout(clickmode="event+select")
            event = st.plotly_chart(fig_ann, use_container_width=True, key="pc_ann_plot",
                                    on_select="rerun")

            # Process click events from plotly
            if event and event.selection:
                for point in event.selection.points:
                    click_idx = point.get("point_index", point.get("pointNumber", None))
                    if click_idx is not None:
                        n_range = st.session_state.pc_annotation_range
                        if n_range == 0:
                            selected_indices = np.array([click_idx])
                        else:
                            _, idxs = kdtree.query(pts[click_idx], k=min(n_range + 1, len(pts)))
                            selected_indices = np.array(idxs).flatten()

                        # Determine left vs right click via session logic
                        # (Streamlit/Plotly doesn't distinguish; use a toggle)
                        if st.session_state.get("pc_click_mode", "reset") == "reset":
                            st.session_state.pc_annotations = selected_indices
                        else:
                            st.session_state.pc_additive_annotations = np.concatenate([
                                st.session_state.pc_additive_annotations, selected_indices
                            ]).astype(int)

            # Click mode toggle
            mode_col1, mode_col2 = st.columns(2)
            with mode_col1:
                if st.button("🖱️ 左键模式 (重置)", key="pc_left_mode"):
                    st.session_state.pc_click_mode = "reset"
            with mode_col2:
                if st.button("🖱️ 右键模式 (追加)", key="pc_right_mode"):
                    st.session_state.pc_click_mode = "additive"

            current_mode = st.session_state.get("pc_click_mode", "reset")
            st.caption(f"当前模式: {'重置选取' if current_mode == 'reset' else '追加选取'}")

            # Show annotation count
            total_ann = len(st.session_state.pc_annotations) + len(st.session_state.pc_additive_annotations)
            st.metric("已标注点数", total_ann)

            # Submit annotation
            if st.button("📤 提交点云标注", key="submit_pc_ann"):
                all_indices = np.concatenate([
                    st.session_state.pc_annotations,
                    st.session_state.pc_additive_annotations,
                ]).astype(int)
                result = api_post("/api/annotation/pointcloud", data={
                    "indices": all_indices.tolist(),
                    "point_cloud_shape": list(pts.shape),
                })
                if result:
                    st.success("点云标注已提交")
                    st.session_state.pc_annotations = np.array([], dtype=int)
                    st.session_state.pc_additive_annotations = np.array([], dtype=int)

            # Clear annotation
            if st.button("🗑️ 清除标注", key="clear_pc_ann"):
                st.session_state.pc_annotations = np.array([], dtype=int)
                st.session_state.pc_additive_annotations = np.array([], dtype=int)
                st.rerun()
        else:
            st.info("暂无点云数据，请先连接后端并选择场景密钥")


# =============================================================================
# TAB 2 — 训练界面 (Training Interface)
# =============================================================================

def render_training_interface():
    """Render the Training tab — reference original training UI."""
    st.markdown("### 🏋️ 训练界面")

    ws_client = get_ws_client()
    ws_connected = st.session_state.ws_connected

    # Prefer WS status, fall back to REST
    if ws_connected and st.session_state.training_status:
        status = st.session_state.training_status
    else:
        status = api_get("/api/training/status")
    if status is None:
        status = {}

    is_training = status.get("is_training", False)

    # Real-time indicator
    if ws_connected:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
            <div style="width:10px;height:10px;background:#22c55e;border-radius:50%;animation:blink 1s infinite;"></div>
            <span style="color:#22c55e;font-weight:600;">实时模式 (WebSocket)</span>
        </div>
        <style>@keyframes blink{0%{opacity:1}50%{opacity:.3}100%{opacity:1}}</style>
        """, unsafe_allow_html=True)
    else:
        st.info("📡 WebSocket 未连接 — 使用 REST 轮询")

    # GPU
    gpu_info = api_get("/api/gpu/status")
    use_gpu = False
    if gpu_info:
        if gpu_info.get("available"):
            st.success(f"✅ GPU: {gpu_info.get('device_name','')}")
            use_gpu = True
        else:
            st.warning("⚠️ 无 GPU，训练将使用 CPU")

    # Dataset selection
    st.markdown("#### 数据集")
    ds_col1, ds_col2 = st.columns([2, 1])
    with ds_col1:
        setting = st.radio("数据集设置", ["Seen", "Unseen"], index=0, key="train_setting",
                           help="Seen: 已知物体  |  Unseen: 未知物体")
    with ds_col2:
        # Dataset selection dropdown
        datasets_resp = api_get("/api/datasets")
        dataset_list = datasets_resp.get("datasets", []) if datasets_resp else []
        dataset_names = [d["name"] for d in dataset_list] if dataset_list else ["默认数据集"]
        selected_dataset = st.selectbox("选择数据集", dataset_names, key="train_dataset_select")

        # Import dataset
        with st.expander("📥 导入数据集"):
            ds_file = st.file_uploader("上传数据集 (zip)", type=["zip"], key="ds_import")
            ds_name = st.text_input("数据集名称", key="ds_import_name")
            if st.button("导入", key="ds_import_btn"):
                if ds_file:
                    result = api_post("/api/datasets/import",
                                      files={"file": (ds_name or "dataset.zip", ds_file, "application/zip")},
                                      data={"name": ds_name})
                    if result:
                        st.success("数据集导入成功")
                        st.rerun()

    data_dir = st.text_input("数据目录", value=api_get("/api/dirs", {}).get("data_dir", "") if api_get("/api/dirs") else "",
                             key="train_data_dir")

    # Few-shot
    few_shot = 0
    if setting == "Unseen":
        few_shot = st.slider("Few-shot (0=zero-shot)", 0, 20, 0, key="train_few_shot")

    # Training parameters
    st.markdown("#### 训练参数")
    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        epochs = st.number_input("Epochs", 1, 200, 80, key="train_epochs")
    with p_col2:
        batch_size = st.selectbox("Batch Size", [4, 8, 16, 32], index=1, key="train_bs")
    with p_col3:
        lr = st.text_input("Learning Rate", "0.0001", key="train_lr")

    # Breakpoint
    st.markdown("#### 训练起点")
    bp_resp = api_get("/api/breakpoints", params={"setting": setting})
    breakpoints = bp_resp.get("breakpoints", []) if bp_resp else []
    bp_options = ["从头开始"] + [f"{b['name']} (Epoch {b['epoch']})" for b in breakpoints]
    selected_bp = st.selectbox("断点选择", bp_options, key="train_bp")
    model_name = st.text_input("模型名称 (留空自动生成)", key="train_model_name")

    # Controls
    st.markdown("#### 训练控制")
    ctrl_c1, ctrl_c2 = st.columns(2)
    with ctrl_c1:
        if is_training:
            if st.button("🚀 开始训练", disabled=True, key="train_start"):
                pass
        else:
            if st.button("🚀 开始训练", type="primary", key="train_start_2"):
                bp_path = None
                if selected_bp != "从头开始":
                    bp_name = selected_bp.split(" (Epoch")[0]
                    for b in breakpoints:
                        if b["name"] == bp_name:
                            bp_path = b["path"]
                            break
                payload = {
                    "setting": setting,
                    "data_dir": data_dir,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": float(lr),
                    "use_gpu": use_gpu,
                    "start_from_breakpoint": bp_path,
                    "model_name": model_name or None,
                    "few_shot": few_shot,
                }
                ws = get_ws_client()
                if ws and ws.connected:
                    ws.send({"type": "start_training", "data": payload})
                    st.success("训练请求已发送 (WebSocket)")
                else:
                    result = api_post("/api/training/start", data=payload)
                    if result:
                        st.success("训练已开始")
                time.sleep(0.5)
                st.rerun()

    with ctrl_c2:
        if is_training:
            if st.button("⏹ 停止训练", key="train_stop"):
                ws = get_ws_client()
                if ws and ws.connected:
                    ws.send({"type": "stop_training"})
                else:
                    api_post("/api/training/stop")
                time.sleep(0.5)
                st.rerun()
        else:
            st.button("⏹ 停止训练", disabled=True, key="train_stop_dis")

    # Progress
    if is_training or len(status.get("history", {}).get("train_loss", [])) > 0:
        st.markdown("---")
        st.markdown("#### 训练进度")
        cur_ep = status.get("current_epoch", 0)
        tot_ep = status.get("total_epochs", 1)
        st.progress(min(cur_ep / tot_ep, 1.0) if tot_ep > 0 else 0, text=f"Epoch {cur_ep}/{tot_ep}")

        m_cols = st.columns(6)
        m_cols[0].metric("Train Loss", f"{status.get('train_loss',0):.4f}")
        m_cols[1].metric("Val Loss", f"{status.get('val_loss',0):.4f}")
        m_cols[2].metric("AUC", f"{status.get('auc',0):.4f}")
        m_cols[3].metric("IOU", f"{status.get('iou',0):.4f}")
        m_cols[4].metric("SIM", f"{status.get('sim',0):.4f}")
        m_cols[5].metric("MAE", f"{status.get('mae',0):.4f}")

        history = status.get("history", {})
        if len(history.get("train_loss", [])) > 1:
            c1, c2 = st.columns(2)
            with c1:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(history["train_loss"], label="Train Loss", color="blue")
                if history.get("val_loss"):
                    ax.plot(history["val_loss"], label="Val Loss", color="red")
                ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Loss Curve")
                ax.legend(loc="best"); ax.grid(True)
                st.pyplot(fig); plt.close()
            with c2:
                fig, ax = plt.subplots(figsize=(8, 4))
                if history.get("val_auc"):
                    ax.plot(history["val_auc"], label="AUC", color="green")
                if history.get("val_iou"):
                    ax.plot(history["val_iou"], label="IOU", color="orange")
                ax.set_xlabel("Epoch"); ax.set_ylabel("Value"); ax.set_title("AUC & IOU")
                ax.legend(loc="best"); ax.grid(True)
                st.pyplot(fig); plt.close()

    # Logs
    st.markdown("#### 训练日志")
    log_lines = st.session_state.ws_log_lines
    if ws_connected and log_lines:
        st.text_area("日志 (实时)", value="\n".join(log_lines[-200:]), height=300, disabled=True, key="ws_log_area")
    else:
        log_path = status.get("log_file_path")
        if log_path:
            resp = api_get("/api/logs/content", params={"path": log_path})
            content = resp.get("content", "") if resp else ""
            st.text_area("日志", value=content[-500:], height=300, disabled=True, key="rest_log_area")

    # Auto-refresh during training
    if is_training:
        time.sleep(POLL_INTERVAL)
        st.rerun()


# =============================================================================
# TAB 3 — 配置界面 (Configuration Interface)
# =============================================================================

def render_config_interface():
    """Render the Config tab — user info, memory, model parallel."""
    st.markdown("### ⚙️ 配置界面")

    # ---- Section 1: User Identity ----
    st.markdown("#### 👤 用户信息")
    id_col1, id_col2 = st.columns(2)
    with id_col1:
        new_org = st.text_input("组织名称", value=st.session_state.org_name, key="cfg_org")
        if new_org != st.session_state.org_name:
            st.session_state.org_name = new_org
    with id_col2:
        new_uid = st.text_input("唯一识别码 (UID)", value=st.session_state.uid, key="cfg_uid")
        if new_uid != st.session_state.uid:
            if not new_uid.strip():
                new_uid = str(uuid.uuid4())[:8]
            st.session_state.uid = new_uid

    # Scene key management
    st.markdown("#### 🔑 监控密钥管理")
    if st.session_state.scene_keys:
        for name, key in st.session_state.scene_keys.items():
            kc1, kc2, kc3 = st.columns([2, 3, 1])
            kc1.markdown(f"**{name}**")
            kc2.code(key)
            if kc3.button("🗑️", key=f"del_key_{name}"):
                del st.session_state.scene_keys[name]
                api_post("/api/scenes/keys/delete", data={"name": name})
                st.rerun()
    else:
        st.caption("暂无密钥，可在工作界面添加")

    add_col1, add_col2, add_col3 = st.columns([2, 3, 1])
    with add_col1:
        new_kn = st.text_input("新密钥名称", key="cfg_new_key_name")
    with add_col2:
        new_kv = st.text_input("新密钥值 (留空自动生成)", key="cfg_new_key_val")
    with add_col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("添加", key="cfg_add_key"):
            if new_kn:
                kv = new_kv.strip() if new_kv.strip() else str(uuid.uuid4())[:12]
                st.session_state.scene_keys[new_kn] = kv
                api_post("/api/scenes/keys", data={"name": new_kn, "key": kv})
                st.rerun()

    # WebSocket connect/disconnect
    st.markdown("#### 📡 连接管理")
    ws = get_ws_client()
    ws_ok = ws is not None and ws.connected

    conn_c1, conn_c2 = st.columns(2)
    with conn_c1:
        if st.button("🔌 连接", key="cfg_connect", use_container_width=True):
            ensure_ws_connected()
            time.sleep(0.5)
            st.rerun()
    with conn_c2:
        if st.button("🔌 断开", key="cfg_disconnect", use_container_width=True):
            if ws:
                ws.disconnect()
            st.session_state.ws_client = None
            st.rerun()

    if ws_ok:
        st.success(f"🟢 WebSocket 已连接 (Org: {st.session_state.org_name}, UID: {st.session_state.uid})")
    else:
        backend_ok = check_backend()
        if backend_ok:
            st.info("🟡 后端已连接，WebSocket 未连接")
        else:
            st.error("🔴 后端未连接")

    st.markdown("---")

    # ---- Section 2: Memory Library ----
    st.markdown("#### 🧠 记忆库管理")
    mem_resp = api_get("/api/config/memory")
    if mem_resp and "libraries" in mem_resp:
        st.session_state.memory_libraries = mem_resp["libraries"]

    # Memory stats
    mem_stats = api_get("/api/config/memory/stats")
    if mem_stats:
        stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
        stat_c1.metric("总记忆数", mem_stats.get("total_memories", 0))
        stat_c2.metric("最大容量", mem_stats.get("max_memories", 0))
        stat_c3.metric("使用率", f"{mem_stats.get('usage_pct', 0):.1f}%")
        stat_c4.metric("索引维度", mem_stats.get("index_dim", 128))
        use_faiss = mem_stats.get("use_faiss", False)
        st.caption(f"向量索引: {'FAISS' if use_faiss else 'NumPy (暴力搜索)'}")

    # Available libraries
    st.markdown("**可用记忆库**")
    if st.session_state.memory_libraries:
        for lib in st.session_state.memory_libraries:
            mc1, mc2, mc3 = st.columns([3, 1, 1])
            mc1.markdown(f"- **{lib.get('name','?')}** ({lib.get('size','?')}) — {lib.get('num_memories',0)} 条记忆")
            if mc2.button("选择", key=f"sel_mem_{lib.get('name','')}"):
                api_post("/api/config/memory/select", data={"name": lib.get("name", "")})
                st.success(f"已选择: {lib.get('name','')}")
                st.rerun()
    else:
        st.caption("暂无记忆库")

    # Memory entries browser
    st.markdown("**记忆条目浏览**")
    mem_filter_col1, mem_filter_col2 = st.columns(2)
    with mem_filter_col1:
        aff_filter = st.selectbox(
            "按可供性过滤", ["全部"] + AFFORDANCE_LABELS,
            key="cfg_mem_aff_filter"
        )
    with mem_filter_col2:
        outcome_filter = st.selectbox(
            "按结果过滤", ["全部", "success", "partial", "failure"],
            key="cfg_mem_outcome_filter"
        )

    filter_params = {
        "page": st.session_state.memory_entries_page,
        "per_page": 20,
        "sort_by": "timestamp",
        "sort_order": "desc",
    }
    if aff_filter != "全部":
        filter_params["affordance_label"] = aff_filter
    if outcome_filter != "全部":
        filter_params["outcome"] = outcome_filter

    mem_entries = api_get("/api/memory/entries", params=filter_params)

    if mem_entries and mem_entries.get("entries"):
        # Display entries as a table
        entries_data = []
        for e in mem_entries["entries"]:
            entries_data.append({
                "ID": e.get("id", "")[:8],
                "可供性": e.get("affordance_label", ""),
                "结果": e.get("outcome", ""),
                "奖励": f"{e.get('reward', 0):.2f}",
                "置信度": f"{e.get('confidence', 0):.2f}",
                "访问次数": e.get("access_count", 0),
            })
        st.dataframe(entries_data, use_container_width=True, hide_index=True)

        # Pagination
        total_pages = mem_entries.get("total_pages", 1)
        current_page = mem_entries.get("page", 1)
        total_entries = mem_entries.get("total", 0)
        pag_c1, pag_c2, pag_c3 = st.columns([1, 2, 1])
        with pag_c1:
            if st.button("上一页", key="mem_prev_page", disabled=current_page <= 1):
                st.session_state.memory_entries_page = max(1, current_page - 1)
                st.rerun()
        with pag_c2:
            st.caption(f"第 {current_page}/{total_pages} 页 (共 {total_entries} 条)")
        with pag_c3:
            if st.button("下一页", key="mem_next_page", disabled=current_page >= total_pages):
                st.session_state.memory_entries_page = current_page + 1
                st.rerun()

        # View selected entry detail
        st.markdown("**查看记忆详情**")
        detail_entry_id = st.text_input("输入记忆ID (完整或前8位)", key="cfg_mem_detail_id")
        if detail_entry_id and st.button("查看详情", key="cfg_mem_view_detail"):
            detail = api_get(f"/api/memory/entry/{detail_entry_id}")
            if detail:
                det_c1, det_c2 = st.columns(2)
                with det_c1:
                    st.json({
                        "id": detail.get("id", ""),
                        "affordance_label": detail.get("affordance_label", ""),
                        "outcome": detail.get("outcome", ""),
                        "reward": detail.get("reward", 0),
                        "confidence": detail.get("confidence", 0),
                        "object_category": detail.get("object_category", ""),
                        "access_count": detail.get("access_count", 0),
                        "timestamp": detail.get("timestamp", 0),
                    })
                with det_c2:
                    # Render point cloud if available
                    pc_b64 = detail.get("point_cloud", "")
                    pref_b64 = detail.get("preference_matrix", "")
                    if pc_b64:
                        try:
                            pc_bytes = base64.b64decode(pc_b64)
                            pc_np = np.frombuffer(pc_bytes, dtype=np.float32).reshape(-1, 3)
                            pref_np = None
                            if pref_b64:
                                pref_bytes = base64.b64decode(pref_b64)
                                pref_raw = np.frombuffer(pref_bytes, dtype=np.float32)
                                if len(pref_raw) == len(pc_np):
                                    pref_np = pref_raw
                            fig_detail = make_pc_figure(
                                pc_np, scores=pref_np,
                                title=f"记忆点云 — {detail.get('affordance_label', '')}",
                                point_size=3, heatmap_style="Jet"
                            )
                            st.plotly_chart(fig_detail, use_container_width=True, key="cfg_mem_detail_plot")
                        except Exception as e:
                            st.warning(f"渲染点云失败: {e}")
                    else:
                        st.info("此记忆无点云数据")
            else:
                st.warning("未找到该记忆条目")

        # Delete entry
        del_c1, del_c2 = st.columns(2)
        with del_c1:
            del_id = st.text_input("删除记忆ID", key="cfg_mem_del_id")
        with del_c2:
            if st.button("删除记忆", key="cfg_mem_del_btn"):
                if del_id:
                    result = api_delete(f"/api/memory/entry/{del_id}")
                    if result:
                        st.success("记忆已删除")
                        st.rerun()
    else:
        st.caption("暂无记忆条目")

    # Pre-populate memory from dataset
    with st.expander("🔄 从训练数据预填充记忆库"):
        pop_c1, pop_c2 = st.columns(2)
        with pop_c1:
            pop_setting = st.selectbox("数据集设置", ["Seen", "Unseen"], key="cfg_pop_setting")
        with pop_c2:
            pop_max = st.number_input("最大样本数", 10, 1000, 100, key="cfg_pop_max")

        if st.button("开始预填充", key="cfg_pop_btn"):
            with st.spinner("正在预填充记忆库 (可能需要几分钟)..."):
                result = api_post("/api/config/memory/populate", data={
                    "setting": pop_setting, "max_samples": pop_max
                })
                if result:
                    st.success(result.get("message", "预填充完成"))
                    st.rerun()
                else:
                    st.error("预填充失败")

    # Import memory library
    with st.expander("📥 导入记忆库"):
        mem_file = st.file_uploader("上传记忆库文件 (zip)", type=["zip"], key="mem_import")
        mem_name = st.text_input("记忆库名称", key="mem_import_name")
        if st.button("导入记忆库", key="mem_import_btn"):
            if mem_file:
                result = api_post("/api/config/memory/import",
                                  files={"file": (mem_name or "memory.zip", mem_file, "application/zip")},
                                  data={"name": mem_name})
                if result:
                    st.success("记忆库导入成功")
                    st.rerun()
            else:
                st.warning("请选择文件")

    # Clear all memories
    with st.expander("⚠️ 危险操作"):
        st.warning("清除所有记忆将不可恢复！")
        if st.button("清除所有记忆", key="cfg_mem_clear_btn"):
            result = api_delete("/api/memory/clear")
            if result:
                st.success("所有记忆已清除")
                st.rerun()

    st.markdown("---")

    # ---- Section 2.5: Image Memory System ----
    st.markdown("#### 🖼️ 图片记忆系统")

    # Initialize image memory system
    im_init_c1, im_init_c2 = st.columns([1, 2])
    with im_init_c1:
        if st.button("初始化图片记忆系统", key="cfg_img_mem_init", use_container_width=True):
            result = api_post("/api/image-memory/init")
            if result:
                st.success(result.get("message", "图片记忆系统初始化成功"))
            else:
                st.error("初始化失败，请检查后端服务")
    with im_init_c2:
        st.caption("初始化图片记忆系统以存储和检索物体-可供性关联的图片特征")

    # Display image memory stats
    im_stats = api_get("/api/image-memory/stats")
    if im_stats:
        im_stat_c1, im_stat_c2, im_stat_c3 = st.columns(3)
        im_stat_c1.metric("存储键数", im_stats.get("num_keys", 0))
        im_stat_c2.metric("总图片数", im_stats.get("total_images", 0))
        im_stat_c3.metric("特征维度", im_stats.get("feature_dim", "—"))
        storage_mb = im_stats.get("storage_size_mb", 0)
        st.caption(f"存储大小: {storage_mb:.2f} MB")
    else:
        st.info("图片记忆系统尚未初始化")

    # Populate image memory from training data
    with st.expander("🔄 从训练数据填充图片记忆"):
        pop_im_c1, pop_im_c2 = st.columns(2)
        with pop_im_c1:
            pop_im_setting = st.selectbox("数据集设置", ["Seen", "Unseen"], key="cfg_im_pop_setting")
        with pop_im_c2:
            pop_im_max = st.number_input("最大样本数", 10, 2000, 200, key="cfg_im_pop_max")

        if st.button("开始填充图片记忆", key="cfg_im_pop_btn"):
            with st.spinner("正在从训练数据填充图片记忆 (可能需要几分钟)..."):
                result = api_post("/api/image-memory/populate", data={
                    "setting": pop_im_setting, "max_samples": pop_im_max
                })
                if result:
                    st.success(result.get("message", "图片记忆填充完成"))
                    st.rerun()
                else:
                    st.error("图片记忆填充失败")

    # Browse and retrieve stored images
    st.markdown("**浏览图片记忆**")
    im_categories = api_get("/api/image-memory/categories")
    if im_categories and im_categories.get("categories"):
        category_pairs = im_categories["categories"]
        pair_labels = [
            f"{p.get('object_category', '?')} — {p.get('affordance_label', '?')}  ({p.get('count', 0)} 张)"
            for p in category_pairs
        ]
        selected_pair_idx = st.selectbox(
            "选择 (物体类别, 可供性标签) 对",
            range(len(pair_labels)),
            format_func=lambda i: pair_labels[i],
            key="cfg_im_cat_select"
        )
        selected_pair = category_pairs[selected_pair_idx]

        im_retrieve_c1, im_retrieve_c2 = st.columns([1, 3])
        with im_retrieve_c1:
            if st.button("查看图片", key="cfg_im_view_btn"):
                obj_cat = selected_pair.get("object_category", "")
                aff_label = selected_pair.get("affordance_label", "")
                result = api_get("/api/image-memory/retrieve", params={
                    "object_category": obj_cat,
                    "affordance_label": aff_label,
                })
                if result and result.get("images"):
                    st.session_state.image_memory_retrieved_images = result["images"]
                else:
                    st.session_state.image_memory_retrieved_images = []
                    st.warning("未找到匹配的图片记忆")

        # Display retrieved images
        if st.session_state.image_memory_retrieved_images:
            with im_retrieve_c2:
                st.caption(f"共 {len(st.session_state.image_memory_retrieved_images)} 张图片")
                for idx, img_data in enumerate(st.session_state.image_memory_retrieved_images):
                    try:
                        if isinstance(img_data, str):
                            # base64 encoded numpy array
                            buf = io.BytesIO(base64.b64decode(img_data))
                            img_array = np.load(buf)
                            st.image(img_array, caption=f"图片 {idx + 1}", use_container_width=True)
                        elif isinstance(img_data, dict) and "base64" in img_data:
                            buf = io.BytesIO(base64.b64decode(img_data["base64"]))
                            img_array = np.load(buf)
                            st.image(img_array, caption=f"图片 {idx + 1}", use_container_width=True)
                        elif isinstance(img_data, dict) and "data" in img_data:
                            buf = io.BytesIO(base64.b64decode(img_data["data"]))
                            img_array = np.load(buf)
                            st.image(img_array, caption=f"图片 {idx + 1}", use_container_width=True)
                    except Exception as e:
                        st.warning(f"解码图片 {idx + 1} 失败: {e}")
    else:
        st.caption("暂无图片记忆类别 — 请先初始化并填充图片记忆")

    # Clear image memories
    with st.expander("⚠️ 清除图片记忆"):
        st.warning("清除所有图片记忆将不可恢复！")
        if st.button("清除所有图片记忆", key="cfg_im_clear_btn"):
            result = api_delete("/api/image-memory/clear")
            if result:
                st.success("所有图片记忆已清除")
                st.session_state.image_memory_retrieved_images = []
                st.rerun()
            else:
                st.error("清除图片记忆失败")

    st.markdown("---")

    # ---- Section 3: Model Parallelism (interface reserved) ----
    st.markdown("#### 🔀 模型并行 (接口预留)")
    par_resp = api_get("/api/config/parallel")
    if par_resp and "parallel" in par_resp:
        st.session_state.model_parallel = par_resp["parallel"]

    st.session_state.model_parallel = st.slider(
        "模型切片数 (并行任务数)", min_value=1, max_value=8,
        value=st.session_state.model_parallel, key="cfg_parallel",
        help="模型切为多少片，意味着可同时处理多少任务 (功能尚未实现)",
    )
    st.caption("🚧 模型并行功能尚未实现，接口已预留")

    if st.button("保存配置", key="cfg_save"):
        result = api_post("/api/config/user", data={
            "org_name": st.session_state.org_name,
            "uid": st.session_state.uid,
            "scene_keys": st.session_state.scene_keys,
            "model_parallel": st.session_state.model_parallel,
        })
        if result:
            st.success("配置已保存")


# =============================================================================
# Main App
# =============================================================================

def main():
    init_session_state()
    process_ws_messages()

    tab_work, tab_train, tab_config = st.tabs(["🖥️ 工作界面", "🏋️ 训练界面", "⚙️ 配置界面"])

    with tab_work:
        render_work_interface()

    with tab_train:
        render_training_interface()

    with tab_config:
        render_config_interface()


if __name__ == "__main__":
    main()
