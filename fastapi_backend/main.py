"""
3D Object Affordance Grounding System - FastAPI Backend (v4.0)

Standalone backend server exposing:
  - WebSocket endpoint for bidirectional real-time communication (org/UID routing)
  - REST API endpoints for work, training, inference, annotation, configuration

All API logic is fully implemented, backed by:
  - IAG model (MyNet / IAG_TextEmb) for 3D affordance grounding
  - AnnotationModel for 2D image annotation
  - MemoryManager for external memory-enhanced inference
  - Full training / validation / inference pipeline
"""

import os
import sys
import json
import io
import time
import base64
import uuid
import zipfile
import asyncio
import sqlite3
import threading
import numpy as np
from datetime import datetime
from collections import deque
from typing import Optional, List, Dict, Set, Tuple, Any
from contextlib import asynccontextmanager

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from PIL import Image
from torchvision import transforms

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model.MyNet import get_MyNet, get_IAG_TextEmb
from data_utils.dataset import PIAD, PIADInference, PIADUnseenFewShot
from utils.loss import HM_Loss, kl_div
from utils.eval import SIM
from annotation.annotation_model import build_annotation_model
from memory_system.memory_manager import MemoryManager
from memory_system.integration import MemoryEnhancedInference, prepopulate_from_dataset
from memory_system.image_memory_manager import ImageMemoryManager

from model.pipeline_api_server import router as pipeline_router

# =============================================================================
# Constants
# =============================================================================

CKPT_DIR = os.path.join(PROJECT_ROOT, "ckpt")
BREAK_POINT_DIR = os.path.join(PROJECT_ROOT, "break_point")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
MEMORY_DIR = os.path.join(PROJECT_ROOT, "memory_store")
IMAGE_MEMORY_DIR = os.path.join(PROJECT_ROOT, "image_memory_store")
ANNOTATION_CKPT_DIR = os.path.join(PROJECT_ROOT, "annotation", "ckpt")

AFFORDANCE_LABELS = [
    "grasp", "contain", "lift", "open", "lay", "sit", "support",
    "wrapgrasp", "pour", "move", "display", "push", "listen",
    "wear", "press", "cut", "stab",
]


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# =============================================================================
# Pydantic Request / Response Models
# =============================================================================

# ---- Generic ----
class MessageResponse(BaseModel):
    success: bool = True
    message: str = ""


# ---- Health / System ----
class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "4.0"
    uptime: float = 0.0


class GPUStatusResponse(BaseModel):
    available: bool = False
    device_name: str = ""
    device_count: int = 0


class DirsResponse(BaseModel):
    data_dir: str = ""
    ckpt_dir: str = ""
    log_dir: str = ""
    break_point_dir: str = ""


# ---- Scenes / Keys ----
class SceneKeyItem(BaseModel):
    name: str
    key: str


class SceneKeyListResponse(BaseModel):
    keys: List[SceneKeyItem] = []


class SceneKeyCreateRequest(BaseModel):
    name: str
    key: str = ""


class SceneKeyDeleteRequest(BaseModel):
    name: str


class SceneInfoResponse(BaseModel):
    scene_id: str = ""
    org: str = ""
    scene_key: str = ""
    type: str = ""
    status: str = ""
    created_at: str = ""


# ---- Inference ----
class InferenceCurrentResponse(BaseModel):
    """Current inference data: point cloud, prediction, images."""
    point_cloud: Any = None
    pred_scores: Any = None
    images: List[dict] = []
    affordance: str = ""
    sample_index: int = 0
    total_samples: int = 0
    memory_applied: bool = False
    class_probabilities: Any = None
    predicted_class: int = -1
    predicted_class_name: str = ""
    confidence: float = 0.0


class InferenceNavigateRequest(BaseModel):
    direction: str = "next"
    scene_id: str = ""


class InferenceLoadRequest(BaseModel):
    model_path: str = ""
    setting: str = "Seen"
    use_memory: bool = False
    memory_dir: str = ""


# ---- Annotation ----
class ImageAnnotationResponse(BaseModel):
    """Result of image annotation by the model."""
    boxes: List[dict] = []
    objects: List[str] = []
    actions: List[str] = []


class AnnotationApproveRequest(BaseModel):
    approved: bool = True
    annotation_data: str = ""  # JSON string of annotation data to save


class PointCloudAnnotationRequest(BaseModel):
    indices: List[int] = []
    point_cloud_shape: List[int] = []
    affordance_label: str = ""
    scene_id: str = ""


# ---- Training ----
class TrainingStartRequest(BaseModel):
    setting: str = "Seen"
    data_dir: str = ""
    epochs: int = 80
    batch_size: int = 8
    lr: float = 0.0001
    use_gpu: bool = True
    start_from_breakpoint: Optional[str] = None
    model_name: Optional[str] = None
    few_shot: int = 0
    use_text_emb: bool = False


class TrainingStatusResponse(BaseModel):
    is_training: bool = False
    current_epoch: int = 0
    total_epochs: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    auc: float = 0.0
    iou: float = 0.0
    sim: float = 0.0
    mae: float = 0.0
    lr: float = 0.0
    history: dict = {}
    model_name: str = ""
    setting: str = ""
    error_message: Optional[str] = None
    log_file_path: Optional[str] = None
    start_time: Optional[str] = None


class BreakpointItem(BaseModel):
    name: str
    path: str
    epoch: int = 0
    setting: str = ""
    size_mb: float = 0.0


class BreakpointListResponse(BaseModel):
    breakpoints: List[BreakpointItem] = []


class ModelItem(BaseModel):
    name: str
    path: str
    size_mb: float = 0.0
    setting: str = ""
    mtime: str = ""


class ModelListResponse(BaseModel):
    models: List[ModelItem] = []


class LogItem(BaseModel):
    name: str
    path: str
    setting: str = ""
    size_kb: float = 0.0
    mtime: str = ""


class LogListResponse(BaseModel):
    logs: List[LogItem] = []


class LogContentResponse(BaseModel):
    content: str = ""


# ---- Datasets ----
class DatasetItem(BaseModel):
    name: str
    path: str = ""
    setting: str = ""
    sample_count: int = 0


class DatasetListResponse(BaseModel):
    datasets: List[DatasetItem] = []


class DatasetImportResponse(BaseModel):
    success: bool = True
    message: str = ""
    name: str = ""


# ---- Config ----
class UserConfigRequest(BaseModel):
    org_name: str = ""
    uid: str = ""
    scene_keys: Dict[str, str] = {}
    model_parallel: int = 1


class UserConfigResponse(BaseModel):
    org_name: str = ""
    uid: str = ""
    scene_keys: Dict[str, str] = {}
    model_parallel: int = 1


class MemoryLibraryItem(BaseModel):
    name: str
    size: str = ""
    path: str = ""
    num_memories: int = 0


class MemoryLibraryResponse(BaseModel):
    libraries: List[MemoryLibraryItem] = []


class MemorySelectRequest(BaseModel):
    name: str


class MemoryImportResponse(BaseModel):
    success: bool = True
    message: str = ""


class MemoryStatsResponse(BaseModel):
    total_memories: int = 0
    max_memories: int = 0
    usage_pct: float = 0.0
    index_dim: int = 128
    use_faiss: bool = False


class MemoryEntrySummary(BaseModel):
    """Summary of a memory entry for listing."""
    id: str = ""
    affordance_label: str = ""
    outcome: str = ""
    reward: float = 0.0
    confidence: float = 0.0
    timestamp: float = 0.0
    object_category: str = ""
    access_count: int = 0


class MemoryEntryListResponse(BaseModel):
    entries: List[MemoryEntrySummary] = []
    total: int = 0
    page: int = 1
    per_page: int = 20
    total_pages: int = 0


class MemoryEntryDetailResponse(BaseModel):
    """Full memory entry data including base64-encoded arrays."""
    id: str = ""
    affordance_label: str = ""
    outcome: str = ""
    reward: float = 0.0
    confidence: float = 0.0
    timestamp: float = 0.0
    object_category: str = ""
    access_count: int = 0
    point_cloud: Optional[str] = None
    preference_matrix: Optional[str] = None
    index_vector: Optional[str] = None
    point_features: Optional[str] = None
    action_parameters: Dict[str, Any] = {}


class MemoryPointCloudResponse(BaseModel):
    """Lightweight response for 3D visualization."""
    point_cloud: Optional[str] = None
    preference_matrix: Optional[str] = None


class MemorySearchRequest(BaseModel):
    query_vector: List[float] = []
    top_k: int = 5
    affordance_label: str = ""
    min_reward: float = 0.0


class MemorySearchResultItem(BaseModel):
    id: str = ""
    affordance_label: str = ""
    outcome: str = ""
    reward: float = 0.0
    confidence: float = 0.0
    timestamp: float = 0.0
    object_category: str = ""
    access_count: int = 0
    distance: float = 0.0


class MemorySearchResponse(BaseModel):
    results: List[MemorySearchResultItem] = []


class ParallelConfigResponse(BaseModel):
    parallel: int = 1


class ParallelConfigRequest(BaseModel):
    parallel: int = 1


# =============================================================================
# Connection Manager — WebSocket connections indexed by (org, uid)
# =============================================================================

class ConnectionManager:
    """Manages WebSocket connections indexed by (org, uid)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._connections: Dict[Tuple[str, str], WebSocket] = {}
        self._subscriptions: Dict[Tuple[str, str], Set[str]] = {}

    def register(self, org: str, uid: str, ws: WebSocket):
        with self._lock:
            self._connections[(org, uid)] = ws
            if (org, uid) not in self._subscriptions:
                self._subscriptions[(org, uid)] = set()

    def unregister(self, org: str, uid: str):
        with self._lock:
            self._connections.pop((org, uid), None)
            self._subscriptions.pop((org, uid), None)

    def subscribe(self, org: str, uid: str, scene_id: str):
        with self._lock:
            if (org, uid) in self._subscriptions:
                self._subscriptions[(org, uid)].add(scene_id)

    def unsubscribe(self, org: str, uid: str, scene_id: str):
        with self._lock:
            if (org, uid) in self._subscriptions:
                self._subscriptions[(org, uid)].discard(scene_id)

    def get_subscribers(self, scene_id: str) -> List[Tuple[str, str]]:
        with self._lock:
            return [k for k, s in self._subscriptions.items() if scene_id in s]

    def get_connection(self, org: str, uid: str) -> Optional[WebSocket]:
        with self._lock:
            return self._connections.get((org, uid))

    async def send_to_subscribers(self, scene_id: str, message: dict):
        for org, uid in self.get_subscribers(scene_id):
            ws = self.get_connection(org, uid)
            if ws:
                try:
                    await ws.send_json(message)
                except Exception:
                    pass

    async def send_to_one(self, org: str, uid: str, message: dict):
        ws = self.get_connection(org, uid)
        if ws:
            try:
                await ws.send_json(message)
            except Exception:
                pass


# =============================================================================
# Scene Manager — scenes with org-based routing and unique keys
# =============================================================================

class SceneInfo:
    def __init__(self, scene_id: str, org: str, scene_type: str):
        self.scene_id = scene_id
        self.org = org
        self.scene_key: str = str(uuid.uuid4())
        self.type = scene_type
        self.subscribers: Set[Tuple[str, str]] = set()
        self.status: str = "active"
        self.created_at: str = datetime.now().isoformat()

    def to_dict(self):
        return {
            "scene_id": self.scene_id,
            "org": self.org,
            "scene_key": self.scene_key,
            "type": self.type,
            "subscribers": list(self.subscribers),
            "status": self.status,
            "created_at": self.created_at,
        }


class SceneManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._scenes: Dict[str, SceneInfo] = {}
        self._key_to_scene: Dict[str, str] = {}

    def create_scene(self, scene_id: str, org: str, scene_type: str) -> SceneInfo:
        with self._lock:
            if scene_id in self._scenes:
                return self._scenes[scene_id]
            s = SceneInfo(scene_id, org, scene_type)
            self._scenes[scene_id] = s
            self._key_to_scene[s.scene_key] = scene_id
            return s

    def get_scene(self, scene_id: str) -> Optional[SceneInfo]:
        with self._lock:
            return self._scenes.get(scene_id)

    def get_scene_by_key(self, scene_key: str) -> Optional[SceneInfo]:
        with self._lock:
            sid = self._key_to_scene.get(scene_key)
            return self._scenes.get(sid) if sid else None

    def add_subscriber(self, scene_id: str, org: str, uid: str):
        with self._lock:
            s = self._scenes.get(scene_id)
            if s:
                s.subscribers.add((org, uid))

    def remove_subscriber(self, scene_id: str, org: str, uid: str):
        with self._lock:
            s = self._scenes.get(scene_id)
            if s:
                s.subscribers.discard((org, uid))

    def set_status(self, scene_id: str, status: str):
        with self._lock:
            s = self._scenes.get(scene_id)
            if s:
                s.status = status

    def list_scenes(self) -> List[dict]:
        with self._lock:
            return [s.to_dict() for s in self._scenes.values()]

    def list_org_scenes(self, org: str) -> List[dict]:
        with self._lock:
            return [s.to_dict() for s in self._scenes.values() if s.org == org]

    def remove_subscriber_from_all(self, org: str, uid: str):
        with self._lock:
            for s in self._scenes.values():
                s.subscribers.discard((org, uid))


# =============================================================================
# Training State (with WebSocket broadcast capability)
# =============================================================================

class TrainingState:
    def __init__(self, max_logs=1000):
        self.max_logs = max_logs
        self._lock = threading.Lock()
        self.log_file = None
        self.log_file_path = None
        self._stop_flag = threading.Event()
        self.reset()

    def reset(self):
        with self._lock:
            self.is_training = False
            self.stop_requested = False
            self.current_epoch = 0
            self.total_epochs = 0
            self.train_loss = 0.0
            self.val_loss = 0.0
            self.auc = 0.0
            self.iou = 0.0
            self.sim = 0.0
            self.mae = 0.0
            self.lr = 0.0
            self.history = {
                "train_loss": [], "val_loss": [],
                "val_auc": [], "val_iou": [],
                "val_sim": [], "val_mae": [],
            }
            self.logs = deque(maxlen=self.max_logs)
            self.model_name = ""
            self.setting = "Seen"
            self.start_time = None
            self.error_message = None
            self.log_file_path = None
            self.log_file = None
            self._stop_flag.clear()

    def request_stop(self):
        self._stop_flag.set()
        self.stop_requested = True

    def is_stop_requested(self):
        return self._stop_flag.is_set()

    def clear_stop(self):
        self._stop_flag.clear()
        self.stop_requested = False

    def init_log_file(self, log_name):
        ensure_dir(LOG_DIR)
        self.log_file_path = os.path.join(LOG_DIR, f"{log_name}.txt")
        self.log_file = open(self.log_file_path, "a", encoding="utf-8")

    def close_log_file(self):
        if self.log_file:
            self.log_file.close()
            self.log_file = None

    def add_log(self, message):
        ts = datetime.now().strftime("%H:%M:%S")
        entry = f"[{ts}] {message}"
        with self._lock:
            self.logs.append(entry)
            if self.log_file:
                try:
                    self.log_file.write(entry + "\n")
                    self.log_file.flush()
                except Exception:
                    pass

    def to_dict(self):
        with self._lock:
            return {
                "is_training": self.is_training,
                "current_epoch": self.current_epoch,
                "total_epochs": self.total_epochs,
                "train_loss": self.train_loss,
                "val_loss": self.val_loss,
                "auc": self.auc,
                "iou": self.iou,
                "sim": self.sim,
                "mae": self.mae,
                "lr": self.lr,
                "history": dict(self.history),
                "model_name": self.model_name,
                "setting": self.setting,
                "error_message": self.error_message,
                "log_file_path": self.log_file_path,
                "start_time": self.start_time,
            }


class BroadcastTrainingState(TrainingState):
    """TrainingState that broadcasts updates to WebSocket subscribers."""

    def __init__(self, scene_id: str, max_logs=1000):
        super().__init__(max_logs)
        self.scene_id = scene_id

    def add_log(self, message):
        super().add_log(message)
        self._broadcast("log_update", {"message": message})

    def broadcast_state(self):
        self._broadcast("scene_update", self.to_dict())

    def broadcast_training_complete(self, model_name: str, metrics: dict):
        self._broadcast("training_complete", {"model_name": model_name, "final_metrics": metrics})

    def _broadcast(self, msg_type: str, data: dict):
        msg = {"type": msg_type, "scene_id": self.scene_id, "data": data}
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(connection_manager.send_to_subscribers(self.scene_id, msg))
            else:
                loop.call_soon_threadsafe(
                    lambda: asyncio.ensure_future(
                        connection_manager.send_to_subscribers(self.scene_id, msg), loop=loop
                    )
                )
        except RuntimeError:
            threading.Thread(target=self._send_sync, args=(msg,), daemon=True).start()

    def _send_sync(self, msg):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(connection_manager.send_to_subscribers(self.scene_id, msg))
            loop.close()
        except Exception:
            pass


# =============================================================================
# Global singletons
# =============================================================================

connection_manager = ConnectionManager()
scene_manager = SceneManager()

_training_states: Dict[str, BroadcastTrainingState] = {}
_training_threads: Dict[str, threading.Thread] = {}
_training_lock = threading.Lock()

_inference_states: Dict[str, dict] = {}

# Persistent scene key store (in-memory; could be persisted to DB/file)
_scene_key_store: Dict[str, str] = {}

# In-memory config store
_user_config: Dict[str, Any] = {
    "org_name": "",
    "uid": "",
    "scene_keys": {},
    "model_parallel": 1,
}

# Global memory manager (lazily initialized)
_memory_manager: Optional[MemoryManager] = None
_memory_manager_lock = threading.Lock()

# Global image memory manager (lazily initialized)
_image_memory_manager: Optional[ImageMemoryManager] = None
_image_memory_manager_lock = threading.Lock()

# Global annotation model (lazily initialized)
_annotation_model = None
_annotation_model_lock = threading.Lock()

_start_time = datetime.now()


# =============================================================================
# Trainer Backend — Full implementation ported from backend.py
# =============================================================================

class TrainerBackend:
    """Backend trainer with full training and validation logic."""

    def __init__(self, state: TrainingState):
        self.state = state
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.criterion_hm = None
        self.criterion_ce = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.config = None
        self._stop_flag = threading.Event()
        self.breakpoint_files = []

    def setup(self, setting, data_dir, epochs, batch_size, lr,
              use_gpu=True, loss_cls=0.3, loss_kl=0.5, few_shot=0,
              use_text_emb=False):
        """Full setup: load data, init model, optimizer, scheduler."""
        self.device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
        self.state.add_log(f"Using device: {self.device}")

        self.config = {
            "Setting": setting, "batch_size": batch_size, "lr": lr, "Epoch": epochs,
            "loss_cls": loss_cls, "loss_kl": loss_kl,
            "N_p": 64, "emb_dim": 512, "proj_dim": 512,
            "num_heads": 4, "N_raw": 2048, "num_affordance": 17,
            "pairing_num": 2, "few_shot": few_shot,
            "use_text_emb": use_text_emb,
        }

        data_path = os.path.join(data_dir, setting)

        # ---- Load datasets ----
        if setting == "Unseen" and few_shot > 0:
            self.state.add_log(f"Using Few-shot learning for Unseen dataset ({few_shot} shots per class)")
            test_files = ["Point_Test.txt", "Img_Test.txt", "Box_Test.txt"]
            for f in test_files:
                if not os.path.exists(os.path.join(data_path, f)):
                    raise FileNotFoundError(f"Test file not found: {os.path.join(data_path, f)}")

            self.state.add_log("Creating Few-shot training dataset...")
            train_dataset = PIADUnseenFewShot(
                run_type="train", setting_type=setting,
                point_path=os.path.join(data_path, "Point_Test.txt"),
                img_path=os.path.join(data_path, "Img_Test.txt"),
                box_path=os.path.join(data_path, "Box_Test.txt"),
                shot_num=few_shot,
            )
            self.state.add_log("Creating Few-shot validation dataset...")
            val_dataset = PIADUnseenFewShot(
                run_type="test", setting_type=setting,
                point_path=os.path.join(data_path, "Point_Test.txt"),
                img_path=os.path.join(data_path, "Img_Test.txt"),
                box_path=os.path.join(data_path, "Box_Test.txt"),
                shot_num=few_shot,
            )
        else:
            required_files = ["Point_Train.txt", "Img_Train.txt", "Box_Train.txt",
                              "Point_Test.txt", "Img_Test.txt", "Box_Test.txt"]
            for f in required_files:
                if not os.path.exists(os.path.join(data_path, f)):
                    raise FileNotFoundError(f"Data file not found: {os.path.join(data_path, f)}")

            self.state.add_log("Loading training data...")
            train_dataset = PIAD("train", setting,
                                 os.path.join(data_path, "Point_Train.txt"),
                                 os.path.join(data_path, "Img_Train.txt"),
                                 os.path.join(data_path, "Box_Train.txt"),
                                 self.config["pairing_num"])
            self.state.add_log("Loading validation data...")
            val_dataset = PIAD("val", setting,
                               os.path.join(data_path, "Point_Test.txt"),
                               os.path.join(data_path, "Img_Test.txt"),
                               os.path.join(data_path, "Box_Test.txt"))

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                       num_workers=4, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                     num_workers=4, shuffle=True)
        self.state.add_log(f"Training samples: {len(train_dataset)}")
        self.state.add_log(f"Validation samples: {len(val_dataset)}")

        # ---- Init model ----
        self.state.add_log("Initializing model...")
        if use_text_emb:
            self.model = get_IAG_TextEmb(
                pre_train=False,
                N_p=self.config["N_p"],
                emb_dim=self.config["emb_dim"],
                proj_dim=self.config["proj_dim"],
                num_heads=self.config["num_heads"],
                N_raw=self.config["N_raw"],
                num_affordance=self.config["num_affordance"],
            )
        else:
            self.model = get_MyNet(
                pre_train=False,
                N_p=self.config["N_p"],
                emb_dim=self.config["emb_dim"],
                proj_dim=self.config["proj_dim"],
                num_heads=self.config["num_heads"],
                N_raw=self.config["N_raw"],
                num_affordance=self.config["num_affordance"],
            )
        self.model = self.model.to(self.device)
        self.state.add_log(f"Model initialized on {self.device}")

        # ---- Loss, optimizer, scheduler ----
        self.criterion_hm = HM_Loss().to(self.device)
        self.criterion_ce = nn.CrossEntropyLoss().to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr,
            betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )

        return True

    def load_breakpoint(self, breakpoint_path):
        """Load from checkpoint."""
        self.state.add_log(f"Loading breakpoint: {breakpoint_path}")
        checkpoint = torch.load(breakpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.state.current_epoch = checkpoint["epoch"] + 1
        self.state.history = checkpoint.get("history", self.state.history)
        self.state.add_log(f"Resumed from epoch {self.state.current_epoch}")
        return True

    def save_breakpoint(self, epoch):
        """Save checkpoint every 5 epochs."""
        ensure_dir(BREAK_POINT_DIR)
        bp_name = f"{self.state.model_name}-{epoch}"
        bp_path = os.path.join(BREAK_POINT_DIR, f"{bp_name}.pt")

        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": epoch,
            "history": self.state.history,
            "config": self.config,
            "model_name": self.state.model_name,
            "setting": self.state.setting,
            "log_file_path": self.state.log_file_path,
        }
        torch.save(checkpoint, bp_path)
        self.breakpoint_files.append(bp_path)
        self.state.add_log(f"Breakpoint saved: {bp_name}")

    def save_final_model(self):
        """Save final model."""
        ensure_dir(CKPT_DIR)
        model_path = os.path.join(CKPT_DIR, f"{self.state.model_name}.pt")
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": self.state.current_epoch,
            "history": self.state.history,
            "config": self.config,
            "model_name": self.state.model_name,
            "setting": self.state.setting,
            "log_file_path": self.state.log_file_path,
        }
        torch.save(checkpoint, model_path)
        self.state.add_log(f"Model saved: {self.state.model_name}.pt")
        self.delete_breakpoints()

    def delete_breakpoints(self):
        """Delete all checkpoints for this model."""
        for bp_path in self.breakpoint_files:
            if os.path.exists(bp_path):
                os.remove(bp_path)
                self.state.add_log(f"Deleted breakpoint: {os.path.basename(bp_path)}")
        self.breakpoint_files = []

    def should_stop(self):
        return self._stop_flag.is_set() or self.state.is_stop_requested()

    def stop(self):
        self._stop_flag.set()
        self.state.request_stop()
        self.state.add_log("Stop signal sent - training will stop after current batch")

    def train_epoch(self):
        """Train one full epoch with real model forward/backward."""
        self.model.train()
        num_batches = len(self.train_loader)
        loss_sum = 0
        use_text_emb = self.config.get("use_text_emb", False)

        for i, batch_data in enumerate(self.train_loader):
            if self.should_stop():
                return None

            self.optimizer.zero_grad()
            temp_loss = 0

            # Training dataset returns:
            #   img, points_list, labels_list, logits_labels_list, sub_box, obj_box
            #   [, text_emb_list] if IAG_TextEmb
            if use_text_emb:
                img, points, labels, logits_labels, sub_box, obj_box, text_embs = batch_data[:7]
            else:
                img, points, labels, logits_labels, sub_box, obj_box = batch_data[:6]
                text_embs = None

            for j, (point, label, logits_label) in enumerate(zip(points, labels, logits_labels)):
                point, label = point.float(), label.float()
                label = label.unsqueeze(dim=-1)

                img_dev = img.to(self.device)
                point_dev = point.to(self.device)
                label_dev = label.to(self.device)
                logits_label_dev = logits_label.to(self.device)
                sub_box_dev = sub_box.to(self.device)
                obj_box_dev = obj_box.to(self.device)

                if use_text_emb and text_embs is not None:
                    text_emb_dev = text_embs[j].to(self.device) if isinstance(text_embs, (list, tuple)) else text_embs.to(self.device)
                    _3d, logits, to_KL = self.model(img_dev, point_dev, sub_box_dev, obj_box_dev, text_emb_dev)
                else:
                    _3d, logits, to_KL = self.model(img_dev, point_dev, sub_box_dev, obj_box_dev)

                loss_hm = self.criterion_hm(_3d, label_dev)
                loss_ce = self.criterion_ce(logits, logits_label_dev)
                loss_kl = kl_div(to_KL[0], to_KL[1])

                temp_loss += loss_hm + self.config["loss_cls"] * loss_ce + self.config["loss_kl"] * loss_kl

            temp_loss.backward()
            self.optimizer.step()
            loss_sum += temp_loss.item()

            if i % 10 == 0:
                self.state.add_log(f"Epoch {self.state.current_epoch} | Batch {i}/{num_batches} | Loss: {temp_loss.item():.4f}")

        mean_loss = loss_sum / (num_batches * self.config["pairing_num"])
        return mean_loss

    def validate(self):
        """Full validation with AUC, IOU, SIM, MAE metrics."""
        self.model.eval()
        val_dataset = self.val_loader.dataset
        results = torch.zeros((len(val_dataset), 2048, 1))
        targets = torch.zeros((len(val_dataset), 2048, 1))

        val_loss_sum = 0
        total_mae = 0
        total_points = 0
        num = 0
        use_text_emb = self.config.get("use_text_emb", False)

        with torch.no_grad():
            for i, batch_data in enumerate(self.val_loader):
                if self.should_stop():
                    break

                if use_text_emb:
                    img, point, label, _, _, _, sub_box, obj_box, text_emb = batch_data[:9]
                else:
                    img, point, label, _, _, _, sub_box, obj_box = batch_data[:8]
                    text_emb = None

                point, label = point.float(), label.float()
                label = label.unsqueeze(dim=-1)

                img_dev = img.to(self.device)
                point_dev = point.to(self.device)
                label_dev = label.to(self.device)
                sub_box_dev = sub_box.to(self.device)
                obj_box_dev = obj_box.to(self.device)

                if use_text_emb and text_emb is not None:
                    text_emb_dev = text_emb.to(self.device)
                    _3d, logits, to_KL = self.model(img_dev, point_dev, sub_box_dev, obj_box_dev, text_emb_dev)
                else:
                    _3d, logits, to_KL = self.model(img_dev, point_dev, sub_box_dev, obj_box_dev)

                val_loss_hm = self.criterion_hm(_3d, label_dev)
                val_loss_kl = kl_div(to_KL[0], to_KL[1])
                val_loss = val_loss_hm + self.config["loss_kl"] * val_loss_kl

                mae = torch.sum(torch.abs(_3d - label_dev), dim=(0, 1))
                point_nums = _3d.shape[0] * _3d.shape[1]
                total_mae += mae.item()
                total_points += point_nums
                val_loss_sum += val_loss.item()

                pred_num = _3d.shape[0]
                results[num:num + pred_num, :, :] = _3d.cpu()
                targets[num:num + pred_num, :, :] = label_dev.cpu()
                num += pred_num

        if total_points == 0:
            return 0, 0, 0, 0, 0

        val_mean_loss = val_loss_sum / len(self.val_loader)
        mean_mae = total_mae / total_points

        results_np = results.numpy()
        targets_np = targets.numpy()

        sim_values = np.zeros(targets_np.shape[0])
        for idx in range(targets_np.shape[0]):
            sim_values[idx] = SIM(results_np[idx], targets_np[idx])
        sim = np.mean(sim_values)

        auc_values = np.zeros(targets_np.shape[0])
        iou_values = np.zeros(targets_np.shape[0])
        iou_thres = np.linspace(0, 1, 20)
        targets_binary = (targets_np >= 0.5).astype(int)

        for idx in range(targets_np.shape[0]):
            t_true = targets_binary[idx]
            p_score = results_np[idx]

            if np.sum(t_true) == 0:
                auc_values[idx] = np.nan
                iou_values[idx] = np.nan
            else:
                auc_values[idx] = roc_auc_score(t_true.flatten(), p_score.flatten())
                temp_iou = []
                for thre in iou_thres:
                    p_mask = (p_score >= thre).astype(int)
                    intersect = np.sum(p_mask & t_true)
                    union = np.sum(p_mask | t_true)
                    temp_iou.append(1. * intersect / union if union > 0 else 0)
                iou_values[idx] = np.mean(temp_iou)

        auc = np.nanmean(auc_values)
        iou = np.nanmean(iou_values)

        return val_mean_loss, auc, iou, sim, mean_mae

    def save_curves(self):
        """Save training curves as PNG."""
        ensure_dir(CKPT_DIR)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        if len(self.state.history["train_loss"]) > 0:
            axes[0, 0].plot(self.state.history["train_loss"], label="Train Loss", color="blue")
            if len(self.state.history["val_loss"]) > 0:
                axes[0, 0].plot(self.state.history["val_loss"], label="Val Loss", color="red")
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].set_title("Loss Curve")
            axes[0, 0].legend(loc="best")
            axes[0, 0].grid(True)

        if len(self.state.history["val_auc"]) > 0:
            axes[0, 1].plot(self.state.history["val_auc"], label="Val AUC", color="green")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("AUC")
            axes[0, 1].set_title("AUC Curve")
            axes[0, 1].legend(loc="best")
            axes[0, 1].grid(True)

        if len(self.state.history["val_iou"]) > 0:
            axes[1, 0].plot(self.state.history["val_iou"], label="Val IOU", color="orange")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("IOU")
            axes[1, 0].set_title("IOU Curve")
            axes[1, 0].legend(loc="best")
            axes[1, 0].grid(True)

        if len(self.state.history["val_sim"]) > 0:
            axes[1, 1].plot(self.state.history["val_sim"], label="Val SIM", color="purple")
            if len(self.state.history["val_mae"]) > 0:
                ax2 = axes[1, 1].twinx()
                ax2.plot(self.state.history["val_mae"], label="Val MAE", color="brown", linestyle="--")
                ax2.set_ylabel("MAE", color="brown")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("SIM")
            axes[1, 1].set_title("SIM and MAE Curves")
            axes[1, 1].legend(loc="best")
            axes[1, 1].grid(True)

        plt.tight_layout()
        curve_path = os.path.join(CKPT_DIR, f"{self.state.model_name.replace('-model', '')}-loss.png")
        plt.savefig(curve_path, dpi=150, bbox_inches="tight")
        plt.close()
        self.state.add_log(f"Training curves saved: {os.path.basename(curve_path)}")
        return curve_path


# =============================================================================
# Inference Backend — Full model loading + prediction + memory enhancement
# =============================================================================

class InferenceBackend:
    """Backend for inference with memory enhancement support."""

    def __init__(self):
        self.model = None
        self.device = None
        self.setting = None
        self.use_text_emb = False
        self.affordance_labels = AFFORDANCE_LABELS
        self.memory_enhancer = None
        self.memory_manager = None

    def load_model(self, model_path, use_memory=False, memory_dir=""):
        """Load model for inference, optionally with memory enhancement."""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        config = checkpoint.get("config", {})
        self.setting = config.get("Setting", "Seen")
        self.use_text_emb = config.get("use_text_emb", False)

        if self.use_text_emb:
            self.model = get_IAG_TextEmb(
                pre_train=False, N_p=64, emb_dim=512, proj_dim=512,
                num_heads=4, N_raw=2048, num_affordance=17,
            )
        else:
            self.model = get_MyNet(
                pre_train=False, N_p=64, emb_dim=512, proj_dim=512,
                num_heads=4, N_raw=2048, num_affordance=17,
            )
        self.model.load_state_dict(checkpoint["model"])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Optionally load memory system
        if use_memory:
            self._init_memory(memory_dir)

        return self.setting

    def _init_memory(self, memory_dir=""):
        """Initialize memory manager and enhancer."""
        global _memory_manager
        try:
            if memory_dir and os.path.exists(memory_dir):
                _memory_manager = MemoryManager(emb_dim=512, index_dim=128, feat_dim=512,
                                                store_dir=memory_dir)
                _memory_manager.load()
            elif _memory_manager is None:
                _memory_manager = MemoryManager(emb_dim=512, index_dim=128, feat_dim=512,
                                                store_dir=MEMORY_DIR)

            self.memory_manager = _memory_manager
            self.memory_enhancer = MemoryEnhancedInference(
                manager=_memory_manager,
                model=self.model,
                device=str(self.device),
                use_text_emb=self.use_text_emb,
            )
        except Exception as e:
            print(f"[InferenceBackend] Memory init failed: {e}")
            self.memory_enhancer = None

    def predict(self, img, point, sub_box, obj_box, affordance_label=None,
                object_category="", use_image_memory=False):
        """Run inference on a single sample with optional memory enhancement."""
        with torch.no_grad():
            if isinstance(point, np.ndarray):
                point = torch.from_numpy(point).float()
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img).float()
            if isinstance(sub_box, np.ndarray):
                sub_box = torch.from_numpy(sub_box).float()
            if isinstance(obj_box, np.ndarray):
                obj_box = torch.from_numpy(obj_box).float()

            img_batch = img.unsqueeze(0).to(self.device)
            point_batch = point.unsqueeze(0).to(self.device)
            sub_box_batch = sub_box.unsqueeze(0).to(self.device)
            obj_box_batch = obj_box.unsqueeze(0).to(self.device)

            # ── Image memory feature averaging ───────────────────────────
            memory_img_feature = None
            image_memory_applied = False

            if use_image_memory and object_category and affordance_label:
                global _image_memory_manager
                if _image_memory_manager is not None:
                    try:
                        # Extract current image feature
                        F_I = self.model.img_encoder(img_batch)  # [1, C, h, w]

                        # Retrieve memory features and compute averaged feature
                        averaged_F_I, info = _image_memory_manager.retrieve_and_average_feature(
                            current_feature=F_I,
                            object_category=object_category,
                            affordance_label=affordance_label,
                            device=self.device,
                        )

                        if info.get("memory_applied", False):
                            memory_img_feature = averaged_F_I
                            image_memory_applied = True
                    except Exception as e:
                        print(f"[InferenceBackend] Image memory retrieval failed: {e}")

            # ── Model forward with optional image memory ─────────────────
            if image_memory_applied and memory_img_feature is not None:
                if self.use_text_emb:
                    B = img_batch.size(0)
                    text_emb = torch.zeros(B, 300, device=self.device)
                    pred, logits, _, _ = self.model.forward_with_image_memory(
                        img_batch, point_batch, sub_box_batch, obj_box_batch,
                        text_emb, memory_img_feature=memory_img_feature
                    )
                else:
                    pred, logits, _, _ = self.model.forward_with_image_memory(
                        img_batch, point_batch, sub_box_batch, obj_box_batch,
                        memory_img_feature=memory_img_feature
                    )
            else:
                if self.use_text_emb:
                    B = img_batch.size(0)
                    text_emb = torch.zeros(B, 300, device=self.device)
                    pred, logits, _ = self.model(img_batch, point_batch, sub_box_batch, obj_box_batch, text_emb)
                else:
                    pred, logits, _ = self.model(img_batch, point_batch, sub_box_batch, obj_box_batch)

            # Compute softmax probabilities
            probabilities = F.softmax(logits, dim=1)
            pred_class = torch.argmax(probabilities, dim=1)
            pred_prob = torch.max(probabilities, dim=1).values

            raw_pred = pred.cpu().squeeze().numpy()
            memory_applied = False

            # Point-cloud memory enhancement
            if self.memory_enhancer is not None:
                try:
                    result = self.memory_enhancer.predict(
                        img_batch, point_batch, sub_box_batch, obj_box_batch,
                        affordance_label=affordance_label,
                        return_details=False,
                    )
                    raw_pred = result["prediction"]
                    memory_applied = result.get("memory_applied", False)
                except Exception as e:
                    print(f"[InferenceBackend] Memory enhancement failed: {e}")

            # Store current image to image memory after inference
            if use_image_memory and object_category and affordance_label:
                try:
                    if _image_memory_manager is not None:
                        with torch.no_grad():
                            F_I = self.model.img_encoder(img_batch)
                        F_I_np = F_I.cpu().numpy().squeeze()
                        img_np = img_batch.cpu().numpy().squeeze()
                        if img_np.ndim == 3 and img_np.shape[0] == 3:
                            img_np = img_np.transpose(1, 2, 0)
                        sub_box_np = sub_box_batch.cpu().numpy().squeeze()
                        obj_box_np = obj_box_batch.cpu().numpy().squeeze()
                        conf = pred_prob.cpu().item()

                        _image_memory_manager.store_image(
                            image=img_np,
                            image_feature=F_I_np,
                            object_category=object_category,
                            affordance_label=affordance_label,
                            sub_box=sub_box_np if sub_box_np.size > 0 else None,
                            obj_box=obj_box_np if obj_box_np.size > 0 else None,
                            confidence=conf,
                        )
                except Exception as e:
                    print(f"[InferenceBackend] Image memory store failed: {e}")

            return {
                "point_cloud_pred": raw_pred,
                "class_probabilities": probabilities.cpu().squeeze().numpy(),
                "predicted_class": pred_class.cpu().item(),
                "predicted_class_name": self.affordance_labels[pred_class.cpu().item()],
                "confidence": pred_prob.cpu().item(),
                "memory_applied": memory_applied or image_memory_applied,
                "image_memory_applied": image_memory_applied,
            }


# =============================================================================
# Training runner — full implementation
# =============================================================================

def run_training(state, setting, data_dir, epochs, batch_size, lr,
                 use_gpu=True, start_from_breakpoint=None, model_name=None,
                 log_name=None, few_shot=0, scene_id="default",
                 use_text_emb=False):
    """Run training in a background thread with full training loop."""
    backend = TrainerBackend(state)
    state.reset()
    state.is_training = True
    state.total_epochs = epochs
    state.setting = setting
    now = datetime.now()
    state.model_name = model_name or f"{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}-{setting}-model"
    state.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    state.stop_requested = False

    if few_shot > 0:
        state.model_name = f"{state.model_name.replace('-model', '')}-fewshot{few_shot}-model"
    if use_text_emb:
        state.model_name = f"{state.model_name.replace('-model', '')}-textemb-model"

    if not log_name:
        log_name = f"{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}-{setting}-log"
        if few_shot > 0:
            log_name = f"{log_name.replace('-log', '')}-fewshot{few_shot}-log"
        if use_text_emb:
            log_name = f"{log_name.replace('-log', '')}-textemb-log"
    state.init_log_file(log_name)

    try:
        backend.setup(setting, data_dir, epochs, batch_size, lr, use_gpu, few_shot=few_shot,
                      use_text_emb=use_text_emb)

        if start_from_breakpoint and start_from_breakpoint != "Pure":
            backend.load_breakpoint(start_from_breakpoint)

        # Auto-extend if resuming past original epochs
        start_epoch = state.current_epoch
        if start_epoch >= epochs:
            epochs = start_epoch + epochs
            state.total_epochs = epochs
            backend.config["Epoch"] = epochs
            state.add_log(f"Automatically extended training to {epochs} epochs")

        for epoch in range(state.current_epoch, epochs):
            if backend.should_stop():
                state.add_log("Training stopped by user request")
                break

            state.current_epoch = epoch
            state.lr = backend.optimizer.state_dict()["param_groups"][0]["lr"]
            state.add_log(f"\n=== Epoch {epoch + 1}/{epochs} ===")
            state.add_log(f"Learning rate: {state.lr:.6f}")
            if few_shot > 0:
                state.add_log(f"Few-shot setting: {few_shot} samples per class")

            if isinstance(state, BroadcastTrainingState):
                state.broadcast_state()

            train_loss = backend.train_epoch()
            if train_loss is None:
                state.add_log("Training interrupted")
                break

            state.train_loss = train_loss
            state.history["train_loss"].append(train_loss)
            state.add_log(f"Train Loss: {train_loss:.4f}")

            val_loss, auc, iou, sim, mae = backend.validate()
            state.val_loss = val_loss
            state.auc = auc
            state.iou = iou
            state.sim = sim
            state.mae = mae

            state.history["val_loss"].append(val_loss)
            state.history["val_auc"].append(auc)
            state.history["val_iou"].append(iou)
            state.history["val_sim"].append(sim)
            state.history["val_mae"].append(mae)

            state.add_log(f"Val Loss: {val_loss:.4f} | AUC: {auc:.4f} | IOU: {iou:.4f} | SIM: {sim:.4f} | MAE: {mae:.4f}")

            if isinstance(state, BroadcastTrainingState):
                state.broadcast_state()

            if (epoch + 1) % 5 == 0 and (epoch + 1) < epochs:
                backend.save_breakpoint(epoch + 1)

            backend.scheduler.step()

        if not backend.should_stop():
            state.add_log("\nTraining completed successfully!")
            backend.save_final_model()
            backend.save_curves()
            if isinstance(state, BroadcastTrainingState):
                state.broadcast_training_complete(state.model_name, state.to_dict())
                scene_manager.set_status(scene_id, "completed")
        else:
            state.add_log("\nTraining stopped - checkpoints preserved")
            if isinstance(state, BroadcastTrainingState):
                scene_manager.set_status(scene_id, "stopped")

    except Exception as e:
        state.error_message = str(e)
        state.add_log(f"ERROR: {str(e)}")
        import traceback
        state.add_log(traceback.format_exc())

    finally:
        state.is_training = False
        state.close_log_file()
        with _training_lock:
            _training_threads.pop(scene_id, None)

    return state.to_dict()


def _get_or_create_training_state(scene_id: str, org: str = "default") -> BroadcastTrainingState:
    if scene_id not in _training_states:
        _training_states[scene_id] = BroadcastTrainingState(scene_id=scene_id)
        scene_manager.create_scene(scene_id, org, "training")
    return _training_states[scene_id]


def _get_or_create_inference_state(scene_id: str, org: str = "default") -> dict:
    if scene_id not in _inference_states:
        _inference_states[scene_id] = {
            "backend": InferenceBackend(),
            "dataset": None,
            "sample_index": 0,
            "total_samples": 0,
            "setting": None,
            "auto_play": False,
        }
        scene_manager.create_scene(scene_id, org, "inference")
    return _inference_states[scene_id]


# =============================================================================
# Helper: get / init annotation model
# =============================================================================

def _get_annotation_model():
    """Lazily load the annotation model (singleton)."""
    global _annotation_model
    with _annotation_model_lock:
        if _annotation_model is None:
            _annotation_model = build_annotation_model(num_interactions=17, pretrained=True)
            _annotation_model.eval()
            # Try loading trained weights
            ensure_dir(ANNOTATION_CKPT_DIR)
            for f in os.listdir(ANNOTATION_CKPT_DIR):
                if f.endswith(".pt"):
                    try:
                        ckpt = torch.load(os.path.join(ANNOTATION_CKPT_DIR, f),
                                          map_location="cpu", weights_only=False)
                        _annotation_model.load_state_dict(ckpt["model"])
                        break
                    except Exception:
                        continue
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            _annotation_model = _annotation_model.to(device)
    return _annotation_model


# =============================================================================
# Helper: encode numpy array to base64
# =============================================================================

def _numpy_to_base64(arr: np.ndarray) -> str:
    """Encode numpy array to base64 string."""
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _image_to_base64(img: Image.Image, fmt="PNG") -> str:
    """Encode PIL Image to base64 string."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# =============================================================================
# FastAPI Application
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    ensure_dir(CKPT_DIR)
    ensure_dir(BREAK_POINT_DIR)
    ensure_dir(LOG_DIR)
    ensure_dir(DATA_DIR)
    ensure_dir(MEMORY_DIR)
    ensure_dir(IMAGE_MEMORY_DIR)
    yield


app = FastAPI(
    title="3D Affordance Grounding API",
    version="4.0",
    description="Backend API for the 3D Affordance Grounding System with WebSocket real-time communication",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pipeline_router)

# =============================================================================
# WebSocket Endpoint
# =============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, org: str = Query("default"), uid: str = Query("")):
    """
    Main WebSocket endpoint for bidirectional real-time communication.

    Query params:
      - org: Organization name (used for message routing)
      - uid: Unique identifier for the client

    The frontend connects here and can send/receive JSON messages for:
      - Training control (start/stop)
      - Real-time training progress
      - Log streaming
      - Inference results
      - Scene management (subscribe/join by key)
    """
    if not uid:
        uid = str(uuid.uuid4())[:8]

    await websocket.accept()
    connection_manager.register(org, uid, websocket)

    # Auto-subscribe to org's scenes
    org_scenes = scene_manager.list_org_scenes(org)
    for scene in org_scenes:
        connection_manager.subscribe(org, uid, scene["scene_id"])
        scene_manager.add_subscriber(scene["scene_id"], org, uid)

    # Send initial state
    try:
        await websocket.send_json({
            "type": "ws_connected",
            "org": org,
            "uid": uid,
            "timestamp": datetime.now().isoformat(),
        })
        await websocket.send_json({
            "type": "scenes_list",
            "scenes": scene_manager.list_scenes(),
        })
    except Exception:
        pass

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "")

            # ---- Training commands ----
            if msg_type == "start_training":
                payload = data.get("data", {})
                scene_id = f"train-{org}-{uid}-{datetime.now().strftime('%H%M%S')}"
                state = _get_or_create_training_state(scene_id, org)
                connection_manager.subscribe(org, uid, scene_id)
                scene_manager.add_subscriber(scene_id, org, uid)

                def _train():
                    run_training(
                        state=state,
                        setting=payload.get("setting", "Seen"),
                        data_dir=payload.get("data_dir", DATA_DIR),
                        epochs=payload.get("epochs", 80),
                        batch_size=payload.get("batch_size", 8),
                        lr=payload.get("lr", 0.0001),
                        use_gpu=payload.get("use_gpu", True),
                        start_from_breakpoint=payload.get("start_from_breakpoint"),
                        model_name=payload.get("model_name"),
                        few_shot=payload.get("few_shot", 0),
                        scene_id=scene_id,
                        use_text_emb=payload.get("use_text_emb", False),
                    )

                with _training_lock:
                    t = threading.Thread(target=_train, daemon=True)
                    _training_threads[scene_id] = t
                    t.start()

                await websocket.send_json({
                    "type": "training_started",
                    "scene_id": scene_id,
                    "data": state.to_dict(),
                })

            elif msg_type == "stop_training":
                scene_id_to_stop = data.get("scene_id", "")
                stopped = False
                for sid, state in _training_states.items():
                    if scene_id_to_stop and sid != scene_id_to_stop:
                        continue
                    if state.is_training:
                        state.request_stop()
                        await websocket.send_json({"type": "training_stopped", "scene_id": sid})
                        stopped = True
                        if scene_id_to_stop:
                            break
                if not stopped:
                    await websocket.send_json({"type": "error", "message": "No active training to stop"})

            # ---- Scene management ----
            elif msg_type == "get_scenes":
                await websocket.send_json({
                    "type": "scenes_list",
                    "scenes": scene_manager.list_scenes(),
                })

            elif msg_type == "subscribe_scene":
                scene_id = data.get("scene_id", "")
                if scene_id:
                    connection_manager.subscribe(org, uid, scene_id)
                    scene_manager.add_subscriber(scene_id, org, uid)
                    scene = scene_manager.get_scene(scene_id)
                    await websocket.send_json({
                        "type": "scene_subscribed",
                        "scene_id": scene_id,
                        "scene_info": scene.to_dict() if scene else {},
                    })

            elif msg_type == "unsubscribe_scene":
                scene_id = data.get("scene_id", "")
                if scene_id:
                    connection_manager.unsubscribe(org, uid, scene_id)
                    scene_manager.remove_subscriber(scene_id, org, uid)
                    await websocket.send_json({"type": "scene_unsubscribed", "scene_id": scene_id})

            elif msg_type == "join_scene":
                scene_key = data.get("scene_key", "")
                if scene_key:
                    scene = scene_manager.get_scene_by_key(scene_key)
                    if scene:
                        connection_manager.subscribe(org, uid, scene.scene_id)
                        scene_manager.add_subscriber(scene.scene_id, org, uid)
                        await websocket.send_json({
                            "type": "scene_joined",
                            "scene_id": scene.scene_id,
                            "scene_info": scene.to_dict(),
                        })
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Scene key not found: {scene_key}",
                        })

            # ---- State sync ----
            elif msg_type == "get_state":
                org_scene_ids = [s["scene_id"] for s in scene_manager.list_org_scenes(org)]
                training_data = {}
                for sid in org_scene_ids:
                    if sid in _training_states:
                        training_data[sid] = _training_states[sid].to_dict()
                await websocket.send_json({
                    "type": "state_update",
                    "data": {
                        "training_status": training_data,
                        "subscribed_scenes": {
                            sid: scene_manager.get_scene(sid).to_dict()
                            for sid in org_scene_ids if scene_manager.get_scene(sid)
                        },
                        "available_scenes": scene_manager.list_scenes(),
                    },
                })

            # ---- Inference ----
            elif msg_type == "inference_next":
                scene_id = data.get("scene_id", "")
                direction = data.get("direction", "next")
                if scene_id and scene_id in _inference_states:
                    inf_state = _inference_states[scene_id]
                    backend = inf_state["backend"]
                    dataset = inf_state["dataset"]

                    if dataset is None:
                        await websocket.send_json({"type": "error", "message": "No inference dataset loaded"})
                    else:
                        # Navigate
                        if direction == "next":
                            inf_state["sample_index"] = (inf_state["sample_index"] + 1) % inf_state["total_samples"]
                        elif direction == "prev":
                            inf_state["sample_index"] = (inf_state["sample_index"] - 1) % inf_state["total_samples"]
                        elif direction == "stop":
                            inf_state["auto_play"] = False

                        # Run inference
                        idx = inf_state["sample_index"]
                        try:
                            sample = dataset[idx]
                            img = sample[0].float()
                            point = sample[1].float()
                            sub_box = sample[4].float()
                            obj_box = sample[5].float()

                            result = backend.predict(img, point, sub_box, obj_box)

                            await websocket.send_json({
                                "type": "inference_result",
                                "scene_id": scene_id,
                                "data": {
                                    "sample_index": idx,
                                    "total_samples": inf_state["total_samples"],
                                    "predicted_class": result["predicted_class"],
                                    "predicted_class_name": result["predicted_class_name"],
                                    "confidence": result["confidence"],
                                    "memory_applied": result["memory_applied"],
                                },
                            })
                        except Exception as e:
                            await websocket.send_json({"type": "error", "message": f"Inference error: {e}"})
                else:
                    await websocket.send_json({"type": "error", "message": "No active inference session"})

            elif msg_type.startswith("pipeline_"):
                from model.pipeline_api_server import handle_pipeline_ws_command
                await handle_pipeline_ws_command(msg_type, data, websocket, org, uid)

            # ---- Unknown ----
            else:
                await websocket.send_json({"type": "ack", "original_type": msg_type})

    except WebSocketDisconnect:
        connection_manager.unregister(org, uid)
        scene_manager.remove_subscriber_from_all(org, uid)
    except Exception as e:
        connection_manager.unregister(org, uid)
        scene_manager.remove_subscriber_from_all(org, uid)


# =============================================================================
# REST API — Health / System
# =============================================================================

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Check if the backend is running."""
    uptime = (datetime.now() - _start_time).total_seconds()
    return HealthResponse(status="ok", version="4.0", uptime=uptime)


@app.get("/api/gpu/status", response_model=GPUStatusResponse)
async def gpu_status():
    """Check GPU availability."""
    available = torch.cuda.is_available()
    return GPUStatusResponse(
        available=available,
        device_name=torch.cuda.get_device_name(0) if available else "",
        device_count=torch.cuda.device_count() if available else 0,
    )


@app.get("/api/dirs", response_model=DirsResponse)
async def get_dirs():
    """Return data directory paths."""
    return DirsResponse(
        data_dir=DATA_DIR,
        ckpt_dir=CKPT_DIR,
        log_dir=LOG_DIR,
        break_point_dir=BREAK_POINT_DIR,
    )


# =============================================================================
# REST API — Scene Keys
# =============================================================================

@app.get("/api/scenes/keys", response_model=SceneKeyListResponse)
async def list_scene_keys():
    """List all stored scene keys."""
    keys = [SceneKeyItem(name=n, key=k) for n, k in _scene_key_store.items()]
    return SceneKeyListResponse(keys=keys)


@app.post("/api/scenes/keys", response_model=MessageResponse)
async def create_scene_key(req: SceneKeyCreateRequest):
    """Create or update a scene key."""
    key_val = req.key.strip() if req.key.strip() else str(uuid.uuid4())[:12]
    _scene_key_store[req.name] = key_val
    return MessageResponse(success=True, message=f"Key '{req.name}' created")


@app.post("/api/scenes/keys/delete", response_model=MessageResponse)
async def delete_scene_key(req: SceneKeyDeleteRequest):
    """Delete a scene key by name."""
    if req.name in _scene_key_store:
        del _scene_key_store[req.name]
        return MessageResponse(success=True, message=f"Key '{req.name}' deleted")
    return MessageResponse(success=False, message=f"Key '{req.name}' not found")


# =============================================================================
# REST API — Inference (Work Interface)
# =============================================================================

@app.post("/api/inference/load", response_model=MessageResponse)
async def load_inference_model(req: InferenceLoadRequest):
    """Load a model for inference and optionally initialize the dataset."""
    if not req.model_path or not os.path.exists(req.model_path):
        # Try to find the latest model
        ensure_dir(CKPT_DIR)
        models = sorted(
            [f for f in os.listdir(CKPT_DIR) if f.endswith(".pt") and "model" in f],
            key=lambda x: os.path.getmtime(os.path.join(CKPT_DIR, x)),
            reverse=True,
        )
        if models:
            req.model_path = os.path.join(CKPT_DIR, models[0])
        else:
            raise HTTPException(status_code=404, detail="No trained model found")

    scene_id = f"infer-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    inf_state = _get_or_create_inference_state(scene_id, "default")
    backend = inf_state["backend"]

    try:
        setting = backend.load_model(req.model_path, use_memory=req.use_memory,
                                     memory_dir=req.memory_dir)
        inf_state["setting"] = setting

        # Load the inference dataset
        data_path = os.path.join(DATA_DIR, setting)
        if os.path.exists(os.path.join(data_path, "Point_Test.txt")):
            dataset = PIADInference(
                run_type="val", setting_type=setting,
                point_path=os.path.join(data_path, "Point_Test.txt"),
                img_path=os.path.join(data_path, "Img_Test.txt"),
                box_path=os.path.join(data_path, "Box_Test.txt"),
            )
            inf_state["dataset"] = dataset
            inf_state["total_samples"] = len(dataset)
            inf_state["sample_index"] = 0
        else:
            inf_state["dataset"] = None
            inf_state["total_samples"] = 0

        return MessageResponse(success=True,
                               message=f"Model loaded (scene: {scene_id}, setting: {setting}, samples: {inf_state['total_samples']})")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


@app.get("/api/inference/current", response_model=InferenceCurrentResponse)
async def get_current_inference(scene_id: str = Query("")):
    """Get current inference data: point cloud, prediction scores, images."""
    # Find the most recent inference session if scene_id not specified
    if not scene_id:
        for sid in reversed(list(_inference_states.keys())):
            if _inference_states[sid].get("backend") and _inference_states[sid]["backend"].model is not None:
                scene_id = sid
                break

    if not scene_id or scene_id not in _inference_states:
        return InferenceCurrentResponse()

    inf_state = _inference_states[scene_id]
    backend = inf_state["backend"]
    dataset = inf_state["dataset"]

    if backend.model is None or dataset is None:
        return InferenceCurrentResponse()

    idx = inf_state["sample_index"]
    try:
        sample = dataset[idx]
        img = sample[0].float()
        point = sample[1].float()
        sub_box = sample[4].float()
        obj_box = sample[5].float()

        result = backend.predict(img, point, sub_box, obj_box)

        # Encode point cloud and prediction as base64
        point_np = point.cpu().numpy()
        pred_np = result["point_cloud_pred"]

        # Encode image as base64
        img_pil = transforms.ToPILImage()(img.cpu())
        img_b64 = _image_to_base64(img_pil)

        return InferenceCurrentResponse(
            point_cloud=_numpy_to_base64(point_np),
            pred_scores=pred_np.tolist() if isinstance(pred_np, np.ndarray) else pred_np,
            images=[{"base64": img_b64}],
            affordance=result["predicted_class_name"],
            sample_index=idx,
            total_samples=inf_state["total_samples"],
            memory_applied=result["memory_applied"],
            class_probabilities=result["class_probabilities"].tolist() if isinstance(result["class_probabilities"], np.ndarray) else None,
            predicted_class=result["predicted_class"],
            predicted_class_name=result["predicted_class_name"],
            confidence=result["confidence"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")


@app.post("/api/inference/navigate", response_model=MessageResponse)
async def inference_navigate(req: InferenceNavigateRequest):
    """Navigate inference: next, prev, auto-play, stop."""
    scene_id = req.scene_id

    # Find the most recent inference session if scene_id not specified
    if not scene_id:
        for sid in reversed(list(_inference_states.keys())):
            if _inference_states[sid].get("backend") and _inference_states[sid]["backend"].model is not None:
                scene_id = sid
                break

    if not scene_id or scene_id not in _inference_states:
        return MessageResponse(success=False, message="No active inference session")

    inf_state = _inference_states[scene_id]

    if req.direction == "next":
        inf_state["sample_index"] = (inf_state["sample_index"] + 1) % inf_state["total_samples"]
    elif req.direction == "prev":
        inf_state["sample_index"] = (inf_state["sample_index"] - 1) % inf_state["total_samples"]
    elif req.direction == "auto":
        inf_state["auto_play"] = True
    elif req.direction == "stop":
        inf_state["auto_play"] = False
    elif req.direction.startswith("goto:"):
        try:
            goto_idx = int(req.direction.split(":")[1])
            inf_state["sample_index"] = min(max(goto_idx, 0), inf_state["total_samples"] - 1)
        except (ValueError, IndexError):
            pass

    return MessageResponse(success=True, message=f"Navigated to sample {inf_state['sample_index']}")


@app.get("/api/inference/images")
async def get_inference_images(scene_id: str = Query("")):
    """Get images currently being processed by the model."""
    if not scene_id:
        for sid in reversed(list(_inference_states.keys())):
            if _inference_states[sid].get("backend") and _inference_states[sid]["backend"].model is not None:
                scene_id = sid
                break

    if not scene_id or scene_id not in _inference_states:
        return {"images": []}

    inf_state = _inference_states[scene_id]
    dataset = inf_state["dataset"]
    if dataset is None:
        return {"images": []}

    idx = inf_state["sample_index"]
    try:
        sample = dataset[idx]
        img = sample[0].float()
        img_pil = transforms.ToPILImage()(img.cpu())
        img_b64 = _image_to_base64(img_pil)
        return {"images": [{"base64": img_b64, "sample_index": idx}]}
    except Exception:
        return {"images": []}


# =============================================================================
# REST API — Annotation
# =============================================================================

@app.post("/api/annotation/image", response_model=ImageAnnotationResponse)
async def annotate_image(file: UploadFile = File(...)):
    """Upload an image and run the annotation model to detect boxes and interactions."""
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    model = _get_annotation_model()
    device = next(model.parameters()).device

    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        detections = model(img_tensor)

    boxes = []
    objects = []
    actions = []

    # Process subject boxes
    if hasattr(detections, "get") and "subject_boxes" in detections:
        subj_boxes = detections["subject_boxes"]
        obj_boxes = detections["object_boxes"]

        if isinstance(subj_boxes, torch.Tensor) and subj_boxes.numel() > 0:
            for box in subj_boxes.cpu().numpy():
                boxes.append({
                    "label": "subject",
                    "confidence": float(detections.get("subject_scores", torch.tensor([1.0]))[0].cpu()) if "subject_scores" in detections else 1.0,
                    "x1": float(box[0]), "y1": float(box[1]),
                    "x2": float(box[2]), "y2": float(box[3]),
                })
                objects.append("subject")

        if isinstance(obj_boxes, torch.Tensor) and obj_boxes.numel() > 0:
            for box in obj_boxes.cpu().numpy():
                boxes.append({
                    "label": "object",
                    "confidence": float(detections.get("object_scores", torch.tensor([1.0]))[0].cpu()) if "object_scores" in detections else 1.0,
                    "x1": float(box[0]), "y1": float(box[1]),
                    "x2": float(box[2]), "y2": float(box[3]),
                })
                objects.append("object")

        # Process interaction predictions
        if detections.get("interaction_logits") is not None and isinstance(detections["interaction_logits"], torch.Tensor):
            probs = F.softmax(detections["interaction_logits"], dim=1)
            top_classes = torch.argmax(probs, dim=1).cpu().numpy()
            for cls_idx in top_classes:
                actions.append(AFFORDANCE_LABELS[cls_idx] if cls_idx < len(AFFORDANCE_LABELS) else "unknown")

    return ImageAnnotationResponse(boxes=boxes, objects=objects, actions=actions)


@app.post("/api/annotation/approve", response_model=MessageResponse)
async def approve_annotation(req: AnnotationApproveRequest):
    """Approve and save annotation results to the dataset."""
    if not req.approved:
        return MessageResponse(success=True, message="Annotation rejected")

    if req.annotation_data:
        try:
            annotation = json.loads(req.annotation_data)
            # Save to annotation output directory
            annot_dir = os.path.join(DATA_DIR, "annotations")
            ensure_dir(annot_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(annot_dir, f"annotation_{timestamp}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(annotation, f, indent=2, ensure_ascii=False)
            return MessageResponse(success=True, message=f"Annotation saved to {out_path}")
        except Exception as e:
            return MessageResponse(success=False, message=f"Failed to save annotation: {e}")

    return MessageResponse(success=True, message="Annotation approved (no data to save)")


@app.post("/api/annotation/pointcloud", response_model=MessageResponse)
async def submit_pointcloud_annotation(req: PointCloudAnnotationRequest):
    """Submit point cloud annotations (selected point indices with affordance label)."""
    annot_dir = os.path.join(DATA_DIR, "annotations")
    ensure_dir(annot_dir)

    annotation = {
        "indices": req.indices,
        "point_cloud_shape": req.point_cloud_shape,
        "affordance_label": req.affordance_label,
        "scene_id": req.scene_id,
        "timestamp": datetime.now().isoformat(),
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(annot_dir, f"pc_annotation_{timestamp}.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(annotation, f, indent=2)
        return MessageResponse(
            success=True,
            message=f"Saved {len(req.indices)} annotated points with label '{req.affordance_label}' to {out_path}",
        )
    except Exception as e:
        return MessageResponse(success=False, message=f"Failed to save: {e}")


# =============================================================================
# REST API — Training
# =============================================================================

@app.post("/api/training/start", response_model=MessageResponse)
async def start_training(req: TrainingStartRequest):
    """Start model training with full pipeline."""
    # Check if training is already running
    for sid, state in _training_states.items():
        if state.is_training:
            return MessageResponse(success=False, message="Training already in progress")

    scene_id = f"train-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    state = _get_or_create_training_state(scene_id, "default")

    def _train():
        run_training(
            state=state,
            setting=req.setting,
            data_dir=req.data_dir or DATA_DIR,
            epochs=req.epochs,
            batch_size=req.batch_size,
            lr=req.lr,
            use_gpu=req.use_gpu,
            start_from_breakpoint=req.start_from_breakpoint,
            model_name=req.model_name,
            few_shot=req.few_shot,
            scene_id=scene_id,
            use_text_emb=req.use_text_emb,
        )

    with _training_lock:
        t = threading.Thread(target=_train, daemon=True)
        _training_threads[scene_id] = t
        t.start()

    return MessageResponse(success=True, message=f"Training started (scene: {scene_id})")


@app.post("/api/training/stop", response_model=MessageResponse)
async def stop_training():
    """Stop the currently running training."""
    stopped = False
    for sid, state in _training_states.items():
        if state.is_training:
            state.request_stop()
            stopped = True
    if stopped:
        return MessageResponse(success=True, message="Training stop requested")
    return MessageResponse(success=False, message="No training in progress")


@app.get("/api/training/status", response_model=TrainingStatusResponse)
async def training_status():
    """Get current training status."""
    for sid, state in _training_states.items():
        if state.is_training or len(state.history.get("train_loss", [])) > 0:
            d = state.to_dict()
            return TrainingStatusResponse(**d)
    return TrainingStatusResponse()


@app.get("/api/breakpoints", response_model=BreakpointListResponse)
async def list_breakpoints(setting: str = Query(None)):
    """List available training breakpoints."""
    ensure_dir(BREAK_POINT_DIR)
    bps = []
    for f in os.listdir(BREAK_POINT_DIR):
        if f.endswith(".pt"):
            path = os.path.join(BREAK_POINT_DIR, f)
            stat = os.stat(path)
            parts = f.replace(".pt", "").split("-")
            bp_setting = parts[5] if len(parts) > 5 else ""
            epoch = int(parts[-1]) if parts[-1].isdigit() else 0
            if setting and bp_setting != setting:
                continue
            bps.append(BreakpointItem(
                name=f.replace(".pt", ""),
                path=path,
                epoch=epoch,
                setting=bp_setting,
                size_mb=stat.st_size / (1024 * 1024),
            ))
    return BreakpointListResponse(breakpoints=bps)


@app.get("/api/models", response_model=ModelListResponse)
async def list_models():
    """List available trained models."""
    ensure_dir(CKPT_DIR)
    models = []
    for f in os.listdir(CKPT_DIR):
        if f.endswith(".pt") and "model" in f:
            path = os.path.join(CKPT_DIR, f)
            stat = os.stat(path)
            # Try to read setting from checkpoint metadata
            setting = ""
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                setting = ckpt.get("config", {}).get("Setting", "")
            except Exception:
                pass
            models.append(ModelItem(
                name=f.replace(".pt", ""),
                path=path,
                size_mb=stat.st_size / (1024 * 1024),
                setting=setting,
                mtime=datetime.fromtimestamp(stat.st_mtime).isoformat(),
            ))
    return ModelListResponse(models=models)


@app.get("/api/logs", response_model=LogListResponse)
async def list_logs():
    """List available log files."""
    ensure_dir(LOG_DIR)
    logs = []
    for f in os.listdir(LOG_DIR):
        if f.endswith(".txt"):
            path = os.path.join(LOG_DIR, f)
            stat = os.stat(path)
            parts = f.replace(".txt", "").split("-")
            log_setting = parts[4] if len(parts) > 5 else "Unknown"
            logs.append(LogItem(
                name=f.replace(".txt", ""),
                path=path,
                setting=log_setting,
                size_kb=stat.st_size / 1024,
                mtime=datetime.fromtimestamp(stat.st_mtime).isoformat(),
            ))
    return LogListResponse(logs=logs)


@app.get("/api/logs/content", response_model=LogContentResponse)
async def get_log_content(path: str = Query(...)):
    """Get content of a log file."""
    if not os.path.exists(path):
        return LogContentResponse(content="File not found")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return LogContentResponse(content=f.read())
    except Exception as e:
        return LogContentResponse(content=f"Error: {e}")


# =============================================================================
# REST API — Datasets
# =============================================================================

@app.get("/api/datasets", response_model=DatasetListResponse)
async def list_datasets():
    """List available datasets by scanning DATA_DIR."""
    datasets = []
    if os.path.exists(DATA_DIR):
        for d in os.listdir(DATA_DIR):
            dp = os.path.join(DATA_DIR, d)
            if os.path.isdir(dp):
                # Count sample files to estimate sample count
                sample_count = 0
                for sub in ["Seen", "Unseen"]:
                    sub_path = os.path.join(dp, sub)
                    if os.path.exists(sub_path):
                        pt_file = os.path.join(sub_path, "Point_Train.txt")
                        if os.path.exists(pt_file):
                            try:
                                with open(pt_file, "r") as f:
                                    sample_count += sum(1 for _ in f)
                            except Exception:
                                pass
                datasets.append(DatasetItem(name=d, path=dp, setting=d, sample_count=sample_count))
    return DatasetListResponse(datasets=datasets)


@app.post("/api/datasets/import", response_model=DatasetImportResponse)
async def import_dataset(file: UploadFile = File(...), name: str = Form("")):
    """Import a dataset from an uploaded zip file."""
    dataset_name = name or (file.filename or "unnamed").replace(".zip", "")
    if not dataset_name:
        dataset_name = "imported_dataset"

    dest_dir = os.path.join(DATA_DIR, dataset_name)
    ensure_dir(dest_dir)

    try:
        contents = await file.read()
        zip_path = os.path.join(dest_dir, f"{dataset_name}.zip")
        with open(zip_path, "wb") as f:
            f.write(contents)

        # Extract zip
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)

        # Remove the zip after extraction
        os.remove(zip_path)

        return DatasetImportResponse(
            success=True,
            message=f"Dataset '{dataset_name}' imported successfully",
            name=dataset_name,
        )
    except zipfile.BadZipFile:
        return DatasetImportResponse(
            success=False,
            message="Uploaded file is not a valid zip archive",
            name=dataset_name,
        )
    except Exception as e:
        return DatasetImportResponse(
            success=False,
            message=f"Import failed: {e}",
            name=dataset_name,
        )


# =============================================================================
# REST API — Configuration
# =============================================================================

@app.get("/api/config/user", response_model=UserConfigResponse)
async def get_user_config():
    """Get current user configuration."""
    return UserConfigResponse(**_user_config)


@app.post("/api/config/user", response_model=MessageResponse)
async def save_user_config(req: UserConfigRequest):
    """Save user configuration."""
    _user_config["org_name"] = req.org_name
    _user_config["uid"] = req.uid
    _user_config["scene_keys"] = req.scene_keys
    _user_config["model_parallel"] = req.model_parallel
    _scene_key_store.update(req.scene_keys)
    return MessageResponse(success=True, message="Configuration saved")


@app.get("/api/config/memory", response_model=MemoryLibraryResponse)
async def get_memory_libraries():
    """Get available memory libraries by scanning the memory store directory."""
    libraries = []
    if os.path.exists(MEMORY_DIR):
        for d in os.listdir(MEMORY_DIR):
            dp = os.path.join(MEMORY_DIR, d)
            if os.path.isdir(dp):
                # Count memory entries
                num_memories = 0
                faiss_file = os.path.join(dp, "faiss_index.bin")
                sqlite_file = os.path.join(dp, "memory_store.db")
                size_str = ""
                if os.path.exists(faiss_file):
                    size_str += f"FAISS: {os.path.getsize(faiss_file) / 1024:.1f}KB"
                if os.path.exists(sqlite_file):
                    if size_str:
                        size_str += ", "
                    size_str += f"SQLite: {os.path.getsize(sqlite_file) / 1024:.1f}KB"
                libraries.append(MemoryLibraryItem(
                    name=d, path=dp, size=size_str or "Empty", num_memories=num_memories,
                ))

    # Also include the default memory manager if active
    if _memory_manager is not None:
        stats = _memory_manager.get_stats()
        libraries.append(MemoryLibraryItem(
            name="active_session",
            path=MEMORY_DIR,
            size=f"{stats['total_memories']} entries",
            num_memories=stats["total_memories"],
        ))

    return MemoryLibraryResponse(libraries=libraries)


@app.get("/api/config/memory/stats", response_model=MemoryStatsResponse)
async def get_memory_stats():
    """Get current memory system statistics."""
    if _memory_manager is None:
        return MemoryStatsResponse()
    stats = _memory_manager.get_stats()
    return MemoryStatsResponse(**stats)


@app.post("/api/config/memory/select", response_model=MessageResponse)
async def select_memory_library(req: MemorySelectRequest):
    """Select and load a memory library for use in inference."""
    global _memory_manager

    memory_dir = os.path.join(MEMORY_DIR, req.name)
    if not os.path.exists(memory_dir):
        # Check if it's the active session
        if req.name == "active_session":
            return MessageResponse(success=True, message="Active session already selected")
        return MessageResponse(success=False, message=f"Memory library '{req.name}' not found")

    try:
        with _memory_manager_lock:
            _memory_manager = MemoryManager(
                emb_dim=512, index_dim=128, feat_dim=512,
                store_dir=memory_dir,
            )
            _memory_manager.load()
        return MessageResponse(success=True, message=f"Memory library '{req.name}' loaded successfully")
    except Exception as e:
        return MessageResponse(success=False, message=f"Failed to load memory library: {e}")


@app.post("/api/config/memory/import", response_model=MemoryImportResponse)
async def import_memory_library(file: UploadFile = File(...), name: str = Form("")):
    """Import a memory library file (zip archive of FAISS + SQLite)."""
    lib_name = name or (file.filename or "unnamed").replace(".zip", "")
    if not lib_name:
        lib_name = "imported_memory"

    dest_dir = os.path.join(MEMORY_DIR, lib_name)
    ensure_dir(dest_dir)

    try:
        contents = await file.read()
        zip_path = os.path.join(dest_dir, f"{lib_name}.zip")
        with open(zip_path, "wb") as f:
            f.write(contents)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)

        os.remove(zip_path)
        return MemoryImportResponse(success=True, message=f"Memory library '{lib_name}' imported successfully")
    except zipfile.BadZipFile:
        return MemoryImportResponse(success=False, message="Uploaded file is not a valid zip archive")
    except Exception as e:
        return MemoryImportResponse(success=False, message=f"Import failed: {e}")


@app.post("/api/config/memory/populate", response_model=MessageResponse)
async def populate_memory_from_dataset(
    model_path: str = Form(""),
    setting: str = Form("Seen"),
    max_samples: int = Form(100),
):
    """Pre-populate the memory store from the training dataset."""
    global _memory_manager

    # Find the latest model if not specified
    if not model_path or not os.path.exists(model_path):
        ensure_dir(CKPT_DIR)
        models = sorted(
            [f for f in os.listdir(CKPT_DIR) if f.endswith(".pt") and "model" in f],
            key=lambda x: os.path.getmtime(os.path.join(CKPT_DIR, x)),
            reverse=True,
        )
        if not models:
            return MessageResponse(success=False, message="No trained model available for population")
        model_path = os.path.join(CKPT_DIR, models[0])

    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        config = checkpoint.get("config", {})
        use_text_emb = config.get("use_text_emb", False)

        if use_text_emb:
            model = get_IAG_TextEmb(pre_train=False, N_p=64, emb_dim=512, proj_dim=512,
                                    num_heads=4, N_raw=2048, num_affordance=17)
        else:
            model = get_MyNet(pre_train=False, N_p=64, emb_dim=512, proj_dim=512,
                              num_heads=4, N_raw=2048, num_affordance=17)
        model.load_state_dict(checkpoint["model"])
        model = model.to(device)
        model.eval()

        # Load training dataset
        data_path = os.path.join(DATA_DIR, setting)
        train_dataset = PIAD("train", setting,
                             os.path.join(data_path, "Point_Train.txt"),
                             os.path.join(data_path, "Img_Train.txt"),
                             os.path.join(data_path, "Box_Train.txt"),
                             2)

        # Populate
        with _memory_manager_lock:
            manager = prepopulate_from_dataset(
                model=model,
                dataset=train_dataset,
                device=device,
                setting=setting,
                manager=_memory_manager,
                max_samples=max_samples,
            )
            _memory_manager = manager
            manager.save()

        return MessageResponse(success=True,
                               message=f"Memory populated with {min(max_samples, len(train_dataset))} samples")
    except Exception as e:
        return MessageResponse(success=False, message=f"Population failed: {e}")


# ---------------------------------------------------------------------------
# Memory Visualization Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/memory/entries", response_model=MemoryEntryListResponse)
async def list_memory_entries(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    affordance_label: str = Query(""),
    outcome: str = Query(""),
    sort_by: str = Query("timestamp"),
    sort_order: str = Query("desc"),
):
    """List memory entries with pagination and filtering.

    Reads directly from SQLite for efficiency (avoids deserialising all entries).
    """
    if _memory_manager is None:
        return MemoryEntryListResponse()

    store = _memory_manager._store
    db_path = store._db_path

    # Validate sort column
    allowed_sort = {"timestamp", "reward", "confidence", "access_count", "id"}
    if sort_by not in allowed_sort:
        sort_by = "timestamp"
    order = "DESC" if sort_order.lower() == "desc" else "ASC"

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        # Build WHERE clause
        conditions = []
        params: list = []
        if affordance_label:
            conditions.append("affordance_label = ?")
            params.append(affordance_label)
        if outcome:
            conditions.append("outcome = ?")
            params.append(outcome)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        # Count total
        count_sql = f"SELECT COUNT(*) FROM memories {where_clause}"
        cursor = conn.execute(count_sql, params)
        total = cursor.fetchone()[0]

        # Fetch page
        offset = (page - 1) * per_page
        data_sql = (
            f"SELECT id, affordance_label, outcome, reward, confidence, "
            f"timestamp, object_category, access_count "
            f"FROM memories {where_clause} "
            f"ORDER BY {sort_by} {order} "
            f"LIMIT ? OFFSET ?"
        )
        cursor = conn.execute(data_sql, params + [per_page, offset])
        rows = cursor.fetchall()
        conn.close()

        entries = [
            MemoryEntrySummary(
                id=r["id"],
                affordance_label=r["affordance_label"] or "",
                outcome=r["outcome"] or "",
                reward=r["reward"] or 0.0,
                confidence=r["confidence"] or 0.0,
                timestamp=r["timestamp"] or 0.0,
                object_category=r["object_category"] or "",
                access_count=r["access_count"] or 0,
            )
            for r in rows
        ]

        total_pages = max(1, (total + per_page - 1) // per_page)

        return MemoryEntryListResponse(
            entries=entries,
            total=total,
            page=page,
            per_page=per_page,
            total_pages=total_pages,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list memory entries: {e}")


@app.get("/api/memory/entry/{entry_id}", response_model=MemoryEntryDetailResponse)
async def get_memory_entry(entry_id: str):
    """Get a specific memory entry with full data including point cloud."""
    if _memory_manager is None:
        raise HTTPException(status_code=404, detail="Memory system not initialised")

    try:
        entry = _memory_manager._store.get(entry_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Memory entry '{entry_id}' not found")

        # Encode numpy arrays as base64 for JSON transport
        point_cloud_b64 = None
        if entry.point_cloud is not None and entry.point_cloud.size > 0:
            point_cloud_b64 = _numpy_to_base64(entry.point_cloud)

        preference_b64 = None
        if entry.preference_matrix is not None and entry.preference_matrix.size > 0:
            preference_b64 = _numpy_to_base64(entry.preference_matrix)

        index_vector_b64 = None
        if entry.index_vector is not None and entry.index_vector.size > 0:
            index_vector_b64 = _numpy_to_base64(entry.index_vector)

        point_features_b64 = None
        if entry.point_features is not None and entry.point_features.size > 0:
            point_features_b64 = _numpy_to_base64(entry.point_features)

        return MemoryEntryDetailResponse(
            id=entry.id,
            affordance_label=entry.affordance_label,
            outcome=entry.outcome,
            reward=entry.reward,
            confidence=entry.confidence,
            timestamp=entry.timestamp,
            object_category=entry.object_category,
            access_count=entry.access_count,
            point_cloud=point_cloud_b64,
            preference_matrix=preference_b64,
            index_vector=index_vector_b64,
            point_features=point_features_b64,
            action_parameters=entry.action_parameters,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get memory entry: {e}")


@app.get("/api/memory/entry/{entry_id}/pointcloud", response_model=MemoryPointCloudResponse)
async def get_memory_entry_pointcloud(entry_id: str):
    """Get just the point cloud and preference for 3D visualization.

    Lightweight endpoint optimised for the 3D viewer.
    """
    if _memory_manager is None:
        raise HTTPException(status_code=404, detail="Memory system not initialised")

    try:
        entry = _memory_manager._store.get(entry_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Memory entry '{entry_id}' not found")

        point_cloud_b64 = None
        if entry.point_cloud is not None and entry.point_cloud.size > 0:
            point_cloud_b64 = _numpy_to_base64(entry.point_cloud)

        preference_b64 = None
        if entry.preference_matrix is not None and entry.preference_matrix.size > 0:
            preference_b64 = _numpy_to_base64(entry.preference_matrix)

        return MemoryPointCloudResponse(
            point_cloud=point_cloud_b64,
            preference_matrix=preference_b64,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get point cloud: {e}")


@app.post("/api/memory/search", response_model=MemorySearchResponse)
async def search_memory(req: MemorySearchRequest):
    """Search for similar memories using a query vector."""
    if _memory_manager is None:
        return MemorySearchResponse()

    if not req.query_vector:
        raise HTTPException(status_code=400, detail="query_vector is required")

    try:
        query = np.array(req.query_vector, dtype=np.float32)
        top_k = max(1, min(req.top_k, 50))

        distances, entries = _memory_manager._store.search(query, top_k)

        results = []
        for dist, entry in zip(distances, entries):
            # Apply optional filters
            if req.affordance_label and entry.affordance_label != req.affordance_label:
                continue
            if entry.reward < req.min_reward:
                continue
            results.append(MemorySearchResultItem(
                id=entry.id,
                affordance_label=entry.affordance_label,
                outcome=entry.outcome,
                reward=entry.reward,
                confidence=entry.confidence,
                timestamp=entry.timestamp,
                object_category=entry.object_category,
                access_count=entry.access_count,
                distance=float(dist),
            ))

        return MemorySearchResponse(results=results)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory search failed: {e}")


@app.delete("/api/memory/entry/{entry_id}", response_model=MessageResponse)
async def delete_memory_entry(entry_id: str):
    """Delete a specific memory entry."""
    if _memory_manager is None:
        return MessageResponse(success=False, message="Memory system not initialised")

    try:
        entry = _memory_manager._store.get(entry_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Memory entry '{entry_id}' not found")

        _memory_manager._store.remove(entry_id)
        return MessageResponse(success=True, message=f"Memory entry '{entry_id}' deleted")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete memory entry: {e}")


@app.delete("/api/memory/clear", response_model=MessageResponse)
async def clear_all_memories():
    """Clear all memories from the store."""
    if _memory_manager is None:
        return MessageResponse(success=False, message="Memory system not initialised")

    try:
        _memory_manager._store.clear()
        return MessageResponse(success=True, message="All memories cleared")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear memories: {e}")


@app.get("/api/config/parallel", response_model=ParallelConfigResponse)
async def get_parallel_config():
    """Get model parallelism configuration."""
    return ParallelConfigResponse(parallel=_user_config.get("model_parallel", 1))


@app.post("/api/config/parallel", response_model=MessageResponse)
async def set_parallel_config(req: ParallelConfigRequest):
    """Set model parallelism configuration."""
    _user_config["model_parallel"] = max(1, req.parallel)
    return MessageResponse(success=True, message=f"Parallel set to {_user_config['model_parallel']}")


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


# =============================================================================
# REST API — Image Memory System
# =============================================================================

class ImageMemoryStatsResponse(BaseModel):
    """Stats response for the image memory store."""
    total_images: int = 0
    max_per_key: int = 0
    feature_dim: int = 512
    use_faiss: bool = False
    categories: List[Dict[str, Any]] = []


class ImageMemoryCategoryItem(BaseModel):
    object_category: str = ""
    affordance_label: str = ""
    count: int = 0


class ImageMemoryEntrySummary(BaseModel):
    id: str = ""
    object_category: str = ""
    affordance_label: str = ""
    confidence: float = 0.0
    timestamp: float = 0.0
    access_count: int = 0


class ImageMemoryEntryListResponse(BaseModel):
    entries: List[ImageMemoryEntrySummary] = []
    total: int = 0
    page: int = 1
    per_page: int = 20
    total_pages: int = 0


class ImageMemoryInitRequest(BaseModel):
    store_dir: str = ""
    feature_dim: int = 512
    use_faiss: bool = True
    max_images_per_key: int = 50
    averaging_strategy: str = "mean"
    alpha: float = 0.5
    max_memory_images: int = 3


class ImageMemoryPopulateRequest(BaseModel):
    max_samples: int = 500


def _get_or_init_image_memory_manager(
    store_dir: str = "",
    feature_dim: int = 512,
    use_faiss: bool = True,
    max_images_per_key: int = 50,
    averaging_strategy: str = "mean",
    alpha: float = 0.5,
    max_memory_images: int = 3,
) -> ImageMemoryManager:
    """Lazily initialise the global image memory manager."""
    global _image_memory_manager
    with _image_memory_manager_lock:
        if _image_memory_manager is None:
            sd = store_dir or IMAGE_MEMORY_DIR
            _image_memory_manager = ImageMemoryManager(
                store_dir=sd,
                feature_dim=feature_dim,
                use_faiss=use_faiss,
                max_images_per_key=max_images_per_key,
                averaging_strategy=averaging_strategy,
                alpha=alpha,
                max_memory_images=max_memory_images,
            )
    return _image_memory_manager


@app.post("/api/image-memory/init", response_model=MessageResponse)
async def init_image_memory(req: ImageMemoryInitRequest):
    """Initialise the image memory store."""
    try:
        mgr = _get_or_init_image_memory_manager(
            store_dir=req.store_dir,
            feature_dim=req.feature_dim,
            use_faiss=req.use_faiss,
            max_images_per_key=req.max_images_per_key,
            averaging_strategy=req.averaging_strategy,
            alpha=req.alpha,
            max_memory_images=req.max_memory_images,
        )
        return MessageResponse(success=True,
                               message=f"Image memory initialised ({mgr.store.count()} existing images)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to init image memory: {e}")


@app.get("/api/image-memory/stats", response_model=ImageMemoryStatsResponse)
async def get_image_memory_stats():
    """Return image memory store statistics."""
    if _image_memory_manager is None:
        return ImageMemoryStatsResponse()
    stats = _image_memory_manager.get_stats()
    return ImageMemoryStatsResponse(
        total_images=stats.get("total_images", 0),
        max_per_key=stats.get("max_per_key", 0),
        feature_dim=stats.get("feature_dim", 512),
        use_faiss=stats.get("use_faiss", False),
        categories=stats.get("categories", []),
    )


@app.get("/api/image-memory/categories")
async def list_image_memory_categories():
    """List all (object_category, affordance_label) pairs with counts."""
    if _image_memory_manager is None:
        return []
    return _image_memory_manager.list_categories()


@app.get("/api/image-memory/entries", response_model=ImageMemoryEntryListResponse)
async def list_image_memory_entries(page: int = Query(1, ge=1),
                                     per_page: int = Query(20, ge=1, le=100)):
    """List image memory entries with pagination."""
    if _image_memory_manager is None:
        return ImageMemoryEntryListResponse()
    result = _image_memory_manager.list_entries(page=page, per_page=per_page)
    entries = [
        ImageMemoryEntrySummary(
            id=e["id"],
            object_category=e.get("object_category", ""),
            affordance_label=e.get("affordance_label", ""),
            confidence=e.get("confidence", 0.0),
            timestamp=e.get("timestamp", 0.0),
            access_count=e.get("access_count", 0),
        )
        for e in result.get("entries", [])
    ]
    return ImageMemoryEntryListResponse(
        entries=entries,
        total=result.get("total", 0),
        page=result.get("page", page),
        per_page=result.get("per_page", per_page),
        total_pages=result.get("total_pages", 0),
    )


@app.get("/api/image-memory/entry/{entry_id}")
async def get_image_memory_entry(entry_id: str):
    """Get a single image memory entry (including the stored image)."""
    if _image_memory_manager is None:
        raise HTTPException(status_code=404, detail="Image memory not initialised")
    entry = _image_memory_manager.retrieve_image(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Entry {entry_id} not found")
    # Return metadata; image data is base64-encoded
    result = {
        "id": entry.get("id", ""),
        "object_category": entry.get("object_category", ""),
        "affordance_label": entry.get("affordance_label", ""),
        "confidence": entry.get("confidence", 0.0),
        "timestamp": entry.get("timestamp", 0.0),
        "access_count": entry.get("access_count", 0),
    }
    image_data = entry.get("image_data")
    if image_data is not None:
        result["image"] = _numpy_to_base64(image_data)
    return result


@app.get("/api/image-memory/retrieve")
async def retrieve_image_memories(
    object_category: str = Query(...),
    affordance_label: str = Query(...),
    top_k: int = Query(5, ge=1, le=50),
):
    """Retrieve image memories by (object_category, affordance_label)."""
    if _image_memory_manager is None:
        raise HTTPException(status_code=404, detail="Image memory not initialised")
    entries = _image_memory_manager.store.retrieve_by_key(
        object_category=object_category,
        affordance_label=affordance_label,
        top_k=top_k,
    )
    results = []
    for e in entries:
        item = {
            "id": e["id"],
            "object_category": e["object_category"],
            "affordance_label": e["affordance_label"],
            "confidence": e["confidence"],
            "timestamp": e["timestamp"],
            "access_count": e["access_count"],
        }
        # Load and encode the stored image
        img_path = e.get("image_path", "")
        if img_path and os.path.exists(img_path):
            img_data = np.load(img_path)
            item["image"] = _numpy_to_base64(img_data)
        results.append(item)
    return results


@app.post("/api/image-memory/populate", response_model=MessageResponse)
async def populate_image_memory(req: ImageMemoryPopulateRequest):
    """Pre-populate the image memory from the training dataset.

    Requires an active inference session with a loaded model.
    """
    if _image_memory_manager is None:
        _get_or_init_image_memory_manager()

    # Find any active inference backend that has a loaded model
    model = None
    for sid, istate in _inference_states.items():
        backend = istate.get("backend")
        if backend and backend.model is not None:
            model = backend.model
            break

    if model is None:
        raise HTTPException(status_code=400, detail="No model loaded. Load a model first.")

    # Try to load dataset
    setting = "Seen"
    data_path = os.path.join(DATA_DIR, setting)
    try:
        from data_utils.dataset import PIAD
        dataset = PIAD("train", setting,
                       os.path.join(data_path, "Point_Train.txt"),
                       os.path.join(data_path, "Img_Train.txt"),
                       os.path.join(data_path, "Box_Train.txt"),
                       2)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load dataset: {e}")

    try:
        count = _image_memory_manager.prepopulate_from_dataset(
            model=model,
            dataset=dataset,
            device=str(next(model.parameters()).device),
            setting=setting,
            max_samples=req.max_samples,
        )
        return MessageResponse(success=True,
                               message=f"Populated {count} image memories")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Populate failed: {e}")


@app.delete("/api/image-memory/entry/{entry_id}", response_model=MessageResponse)
async def delete_image_memory_entry(entry_id: str):
    """Delete a single image memory entry."""
    if _image_memory_manager is None:
        raise HTTPException(status_code=404, detail="Image memory not initialised")
    _image_memory_manager.store.remove(entry_id)
    return MessageResponse(success=True, message=f"Entry {entry_id} deleted")


@app.delete("/api/image-memory/clear", response_model=MessageResponse)
async def clear_image_memory():
    """Clear all image memories."""
    if _image_memory_manager is None:
        return MessageResponse(success=False, message="Image memory not initialised")
    _image_memory_manager.clear()
    return MessageResponse(success=True, message="All image memories cleared")
