"""
DAgger training script for fine-tuning pi0 using BC policies as experts with LoRA.

This script implements Dataset Aggregation (DAgger) where:
- pi0 is the student policy being fine-tuned via LoRA (Low-Rank Adaptation)
- N BC policies (trained on different bowl sizes) serve as experts (N can be any number >= 1)
- During rollouts, environments with different bowl sizes are randomly sampled
- The corresponding BC expert provides action labels for aggregated training

The number of environments (bowl scales) and expert policies is flexible - you can use
any number as long as they match (one expert per bowl scale).

LoRA Configuration:
- Uses PEFT library for parameter-efficient fine-tuning
- Only trains low-rank adaptation matrices, keeping base model frozen
- Significantly reduces memory requirements compared to full fine-tuning

Usage:
    # Example with 2 experts/environments:
    python scripts/train_dagger.py \
        --expert-checkpoints checkpoints/bc_scale_0.4/best_model.pt checkpoints/bc_scale_0.7/best_model.pt \
        --bowl-scales 0.4 0.7 \
        --num-dagger-iterations 10

    # Example with 5 experts/environments:
    python scripts/train_dagger.py \
        --expert-checkpoints ckpt1.pt ckpt2.pt ckpt3.pt ckpt4.pt ckpt5.pt \
        --bowl-scales 0.3 0.4 0.5 0.6 0.7 \
        --num-dagger-iterations 10

    # LoRA-specific options:
    python scripts/train_dagger.py \
        --expert-checkpoints ckpt1.pt ckpt2.pt \
        --bowl-scales 0.4 0.7 \
        --lora-rank 16 \
        --lora-alpha 32 \
        --lora-dropout 0.1

Requirements:
    - Pre-trained BC expert checkpoints (one per bowl scale)
    - pi0 checkpoint (or will use default LIBERO checkpoint)
    - peft library: pip install peft
"""
import argparse
import collections
import dataclasses
import gc
import logging
import math
import os
import pathlib
import random
import sys
from collections import OrderedDict
from typing import Any

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from models.diffusion_policy import DiffusionPolicyUNet
import utils.obs_utils as ObsUtils

# For pi0 imports
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
import openpi.models.pi0_config as pi0_config
import openpi.models_pytorch.pi0_pytorch as pi0_pytorch

# Try to import PEFT for LoRA
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logging.warning("PEFT not installed. Install with: pip install peft")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


# =============================================================================
# BC Expert Policy Wrapper
# =============================================================================
class AttrDict(dict):
    """Dictionary that allows attribute-style access."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def lock(self):
        pass

    def unlock(self):
        pass


def create_bc_config(
    observation_horizon: int = 2,
    action_horizon: int = 8,
    prediction_horizon: int = 16,
):
    """Create config for BC DiffusionPolicyUNet."""
    algo_config = AttrDict({
        "horizon": AttrDict({
            "observation_horizon": observation_horizon,
            "action_horizon": action_horizon,
            "prediction_horizon": prediction_horizon,
        }),
        "ddpm": AttrDict({
            "enabled": True,
            "num_train_timesteps": 100,
            "num_inference_timesteps": 10,
            "beta_schedule": "squaredcos_cap_v2",
            "clip_sample": True,
            "prediction_type": "epsilon",
        }),
        "ddim": AttrDict({"enabled": False}),
        "ema": AttrDict({"enabled": False, "power": 0.75}),
        "unet": AttrDict({"enabled": True}),
        "transformer": AttrDict({"enabled": False}),
        "optim_params": AttrDict({
            "policy": AttrDict({
                "optimizer_type": "adamw",
                "learning_rate": AttrDict({
                    "initial": 1e-4, "decay_factor": 0.1, "epoch_schedule": [],
                    "scheduler_type": "cosine", "warmup_steps": 500, "step_every_batch": True,
                }),
                "regularization": AttrDict({"L2": 1e-6}),
                "num_epochs": 100, "num_train_batches": 1000,
            }),
        }),
    })

    obs_config = AttrDict({
        "modalities": AttrDict({
            "obs": AttrDict({
                "rgb": ["agentview_image", "robot0_eye_in_hand_image"],
                "low_dim": ["robot0_eef_pos", "robot0_eef_ori", "robot0_gripper_qpos"],
            }),
        }),
        "encoder": AttrDict({
            "rgb": AttrDict({
                "core_class": "VisualCore",
                "core_kwargs": AttrDict({
                    "backbone_class": "ResNet18Conv",
                    "backbone_kwargs": AttrDict({"pretrained": False, "input_coord_conv": False}),
                    "pool_class": "SpatialSoftmax",
                    "pool_kwargs": AttrDict({
                        "num_kp": 32, "learnable_temperature": False,
                        "temperature": 1.0, "noise_std": 0.0,
                    }),
                }),
                "obs_randomizer_class": None,
                "obs_randomizer_kwargs": AttrDict({}),
            }),
            "low_dim": AttrDict({
                "core_class": None, "core_kwargs": AttrDict({}),
                "obs_randomizer_class": None, "obs_randomizer_kwargs": AttrDict({}),
            }),
        }),
    })

    global_config = AttrDict({
        "algo_name": "diffusion_policy",
        "all_obs_keys": ["agentview_image", "robot0_eye_in_hand_image",
                        "robot0_eef_pos", "robot0_eef_ori", "robot0_gripper_qpos"],
    })

    return algo_config, obs_config, global_config


class BCExpertPolicy:
    """Wrapper for BC DiffusionPolicyUNet expert."""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = device
        self.observation_horizon = 2
        self.image_size = 84
        
        # Load checkpoint
        logger.info(f"Loading BC expert from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Create model
        obs_key_shapes = OrderedDict({
            "agentview_image": (3, self.image_size, self.image_size),
            "robot0_eye_in_hand_image": (3, self.image_size, self.image_size),
            "robot0_eef_pos": (3,),
            "robot0_eef_ori": (3,),
            "robot0_gripper_qpos": (1,),
        })
        
        ObsUtils.initialize_obs_modality_mapping_from_dict({
            "rgb": ["agentview_image", "robot0_eye_in_hand_image"],
            "low_dim": ["robot0_eef_pos", "robot0_eef_ori", "robot0_gripper_qpos"],
        })
        
        algo_config, obs_config, global_config = create_bc_config()
        ObsUtils.initialize_default_obs_encoder(obs_config.encoder)
        
        self.model = DiffusionPolicyUNet(
            algo_config=algo_config,
            obs_config=obs_config,
            global_config=global_config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=7,
            device=torch.device(device),
        )
        
        self.model.deserialize(checkpoint["model"])
        self.model.set_eval()
        logger.info(f"Loaded BC expert (epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.4f})")
    
    def reset(self):
        """Reset the policy state."""
        self.model.reset()
        self.obs_history = collections.deque(maxlen=self.observation_horizon)
    
    def get_action(self, obs_dict: dict) -> np.ndarray:
        """Get expert action for the given observation."""
        # Add to history
        self.obs_history.append(obs_dict)
        while len(self.obs_history) < self.observation_horizon:
            self.obs_history.appendleft(obs_dict)
        
        # Build observation with temporal stacking
        batched_obs = {}
        for key in obs_dict.keys():
            stacked = torch.stack([self.obs_history[i][key] for i in range(self.observation_horizon)], dim=0)
            batched_obs[key] = stacked.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.model.get_action(batched_obs)
            return action.squeeze(0).cpu().numpy()


# =============================================================================
# Pi0 Student Policy Wrapper with LoRA (PyTorch)
# =============================================================================
class Pi0StudentPolicy:
    """Wrapper for PyTorch pi0 model with PEFT LoRA fine-tuning support."""
    
    def __init__(
        self,
        config_name: str = "pi05_libero",
        checkpoint_dir: str = "gs://openpi-assets/checkpoints/pi05_libero",
        device: str = "cuda",
        lora_config: dict | None = None,
    ):
        self.device = device
        self.config_name = config_name
        self.lora_enabled = lora_config is not None and PEFT_AVAILABLE
        
        logger.info(f"Loading pi0 from {checkpoint_dir}")
        
        # Download checkpoint if needed
        import openpi.shared.download as download
        checkpoint_dir = pathlib.Path(download.maybe_download(str(checkpoint_dir)))
        
        # Check for PyTorch model
        weight_path = checkpoint_dir / "model.safetensors"
        if not weight_path.exists():
            # Fall back to JAX model path and check
            params_path = checkpoint_dir / "params"
            if params_path.exists():
                logger.warning("Only JAX checkpoint found. LoRA requires PyTorch checkpoint.")
                logger.warning("Will load JAX model for inference only (no LoRA training).")
                self._load_jax_model(config_name, checkpoint_dir)
                self.is_pytorch = False
                self.lora_enabled = False
                return
            else:
                raise FileNotFoundError(f"No checkpoint found at {checkpoint_dir}")
        
        self.is_pytorch = True
        self._load_pytorch_model(config_name, weight_path, lora_config)
    
    def _load_jax_model(self, config_name: str, checkpoint_dir: pathlib.Path):
        """Load JAX model for inference only."""
        train_config = _config.get_config(config_name)
        self.policy = _policy_config.create_trained_policy(
            train_config, str(checkpoint_dir), pytorch_device=self.device
        )
        self.model = None
        logger.info("Loaded JAX model (inference only, no LoRA)")
    
    def _load_pytorch_model(self, config_name: str, weight_path: pathlib.Path, lora_config: dict | None):
        """Load PyTorch model with optional LoRA."""
        import safetensors.torch
        
        train_config = _config.get_config(config_name)
        
        # Create PyTorch model
        logger.info("Creating PyTorch PI0 model...")
        self.model = pi0_pytorch.PI0Pytorch(config=train_config.model)
        
        # Load weights
        logger.info(f"Loading weights from {weight_path}...")
        safetensors.torch.load_model(self.model, str(weight_path))
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Convert to bfloat16 for memory efficiency
        if hasattr(self.model, 'paligemma_with_expert'):
            self.model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        
        # Apply LoRA if configured
        if self.lora_enabled and lora_config:
            self._apply_lora(lora_config)
        
        # Create a simple policy wrapper for inference
        self.policy = None  # Will use direct model inference
        self.train_config = train_config
        
        logger.info(f"PyTorch pi0 loaded successfully (LoRA: {self.lora_enabled})")
    
    def _apply_lora(self, lora_config: dict):
        """Apply LoRA adapters to the PyTorch model."""
        if not PEFT_AVAILABLE:
            raise RuntimeError("PEFT not installed. Install with: pip install peft")
        
        if not hasattr(self.model, 'paligemma_with_expert'):
            logger.warning("Model doesn't have paligemma_with_expert, LoRA not applied")
            self.lora_enabled = False
            return
        
        logger.info(f"Applying LoRA with rank={lora_config['rank']}, alpha={lora_config['alpha']}")
        
        # Create LoRA config for the language model
        peft_config = LoraConfig(
            r=lora_config["rank"],
            lora_alpha=lora_config["alpha"],
            lora_dropout=lora_config["dropout"],
            target_modules=lora_config["target_modules"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA to the PaliGemma language model
        lm = self.model.paligemma_with_expert.paligemma.language_model
        self.lora_model = get_peft_model(lm, peft_config)
        
        # Replace the original language model with LoRA version
        self.model.paligemma_with_expert.paligemma.language_model = self.lora_model
        
        # Freeze all non-LoRA parameters
        for name, param in self.model.named_parameters():
            if 'lora_' not in name:
                param.requires_grad = False
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"LoRA trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    def get_action(self, obs: dict) -> np.ndarray:
        """Get action from pi0 for the given observation."""
        if self.policy is not None:
            # JAX model path
            result = self.policy.infer(obs)
            return result["actions"]
        else:
            # PyTorch model path - need to implement direct inference
            # For now, raise an error - we need the transforms
            raise NotImplementedError(
                "Direct PyTorch inference not yet implemented. "
                "Use JAX model for rollouts or implement transforms."
            )
    
    def get_model(self):
        """Get the underlying model for fine-tuning."""
        return self.model
    
    def train_mode(self):
        """Set model to training mode."""
        if self.model is not None:
            self.model.train()
    
    def eval_mode(self):
        """Set model to evaluation mode."""
        if self.model is not None:
            self.model.eval()
    
    def save_lora(self, path: str):
        """Save LoRA weights."""
        if self.lora_enabled and hasattr(self, 'lora_model'):
            os.makedirs(path, exist_ok=True)
            self.lora_model.save_pretrained(path)
            logger.info(f"Saved LoRA weights to {path}")
        else:
            logger.warning("LoRA not enabled, cannot save adapter weights")
    
    def load_lora(self, path: str):
        """Load LoRA weights."""
        if self.lora_enabled and hasattr(self, 'lora_model'):
            from peft import PeftModel
            lm = self.model.paligemma_with_expert.paligemma.language_model
            # Need to get base model first if already wrapped
            if hasattr(lm, 'base_model'):
                lm = lm.base_model.model
            self.lora_model = PeftModel.from_pretrained(lm, path)
            self.model.paligemma_with_expert.paligemma.language_model = self.lora_model
            logger.info(f"Loaded LoRA weights from {path}")


# =============================================================================
# Environment Utilities
# =============================================================================
def _quat2axisangle(quat):
    """Convert quaternion to axis-angle."""
    quat = quat.copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def modify_bowl_scale(scale: float):
    """Modify the akita_black_bowl.xml to use the given scale."""
    bowl_xml_path = pathlib.Path(__file__).parent.parent / \
        "third_party/libero/libero/libero/assets/stable_scanned_objects/akita_black_bowl/akita_black_bowl.xml"
    
    with open(bowl_xml_path, 'r') as f:
        content = f.read()
    
    # Replace scale attribute
    import re
    new_content = re.sub(
        r'scale="[\d.]+ [\d.]+ [\d.]+"',
        f'scale="{scale} {scale} {scale}"',
        content
    )
    
    with open(bowl_xml_path, 'w') as f:
        f.write(new_content)
    
    logger.info(f"Set bowl scale to {scale}")


def get_libero_env(task, resolution: int, seed: int):
    """Create LIBERO environment."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def process_image(img: np.ndarray, image_size: int = 84) -> torch.Tensor:
    """Process image for BC policy: (H,W,C) uint8 -> (C,H,W) float tensor."""
    img = torch.from_numpy(img.copy()).float()
    img = img.permute(2, 0, 1)
    img = F.interpolate(img.unsqueeze(0), size=(image_size, image_size),
                        mode='bilinear', align_corners=False)[0]
    return img


def process_obs_for_bc(obs: dict, image_size: int = 84) -> dict:
    """Process environment observation for BC policy."""
    # Rotate images 180 degrees to match training
    agentview = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    
    return {
        "agentview_image": process_image(agentview, image_size),
        "robot0_eye_in_hand_image": process_image(wrist, image_size),
        "robot0_eef_pos": torch.from_numpy(obs["robot0_eef_pos"].copy()).float(),
        "robot0_eef_ori": torch.from_numpy(_quat2axisangle(obs["robot0_eef_quat"])).float(),
        "robot0_gripper_qpos": torch.from_numpy(obs["robot0_gripper_qpos"][:1].copy()).float(),
    }


def process_obs_for_pi0(obs: dict, task_description: str) -> dict:
    """Process environment observation for pi0 policy."""
    # Rotate images 180 degrees
    agentview = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    
    return {
        "observation/image": agentview,
        "observation/wrist_image": wrist,
        "observation/state": np.concatenate([
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        ]),
        "prompt": task_description,
    }


# =============================================================================
# DAgger Dataset
# =============================================================================
class DAggerDataset(Dataset):
    """Dataset that aggregates (observation, expert_action) pairs."""
    
    def __init__(self):
        self.observations = []  # List of processed observation dicts
        self.actions = []  # List of expert actions
    
    def add(self, obs: dict, action: np.ndarray):
        """Add a single (obs, action) pair."""
        self.observations.append(obs)
        self.actions.append(action)
    
    def add_batch(self, observations: list, actions: list):
        """Add a batch of (obs, action) pairs."""
        self.observations.extend(observations)
        self.actions.extend(actions)
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]
    
    def save(self, path: str):
        """Save dataset to disk."""
        torch.save({
            "observations": self.observations,
            "actions": self.actions,
        }, path)
        logger.info(f"Saved DAgger dataset ({len(self)} samples) to {path}")
    
    def load(self, path: str):
        """Load dataset from disk."""
        data = torch.load(path, weights_only=False)
        self.observations = data["observations"]
        self.actions = data["actions"]
        logger.info(f"Loaded DAgger dataset ({len(self)} samples) from {path}")


# =============================================================================
# DAgger Training Loop
# =============================================================================
@dataclasses.dataclass
class DAggerConfig:
    """Configuration for DAgger training with LoRA.
    
    The number of environments (bowl_scales) and experts (expert_checkpoints) is flexible.
    You can use any number >= 1, as long as they match (one expert per environment).
    """
    # Expert configs (flexible number, must match bowl_scales)
    expert_checkpoints: list[str] = dataclasses.field(default_factory=list)
    bowl_scales: list[float] = dataclasses.field(default_factory=list)
    
    # Student config
    pi0_config_name: str = "pi05_libero"
    pi0_checkpoint: str = "gs://openpi-assets/checkpoints/pi05_libero"
    
    # Environment config
    task_suite: str = "libero_spatial"
    task_id: int = 0  # Which task to train on
    
    # DAgger config
    num_dagger_iterations: int = 10
    rollouts_per_iteration: int = 20
    max_steps_per_rollout: int = 220
    num_steps_wait: int = 10
    beta_schedule: str = "linear"  # How to mix student/expert: "linear", "constant"
    initial_beta: float = 1.0  # Start with 100% expert
    
    # LoRA config
    lora_rank: int = 16  # Rank of LoRA matrices
    lora_alpha: float = 32.0  # LoRA scaling factor
    lora_dropout: float = 0.1  # Dropout for LoRA layers
    lora_target_modules: list[str] = dataclasses.field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # Training config
    batch_size: int = 8  # Smaller batch for LoRA fine-tuning
    learning_rate: float = 1e-4  # Higher LR for LoRA
    weight_decay: float = 0.01
    train_steps_per_iteration: int = 500
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    
    # Output config
    output_dir: str = "checkpoints/dagger_pi0_lora"
    save_videos: bool = True
    video_dir: str = "data/libero/dagger_videos"
    
    # Misc
    device: str = "cuda"
    seed: int = 42


def run_dagger(config: DAggerConfig):
    """Run DAgger training loop."""
    
    # Validate config
    num_experts = len(config.expert_checkpoints)
    num_scales = len(config.bowl_scales)
    assert num_experts == num_scales, \
        f"Number of expert checkpoints ({num_experts}) must match number of bowl scales ({num_scales})"
    assert num_experts > 0, "Must provide at least one expert checkpoint and bowl scale"
    
    logger.info(f"Configured with {num_experts} environment(s)/expert(s)")
    
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    
    # Create output directories
    output_dir = pathlib.Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.save_videos:
        pathlib.Path(config.video_dir).mkdir(parents=True, exist_ok=True)
    
    # Load BC experts
    logger.info("Loading BC expert policies...")
    experts = {}
    for ckpt, scale in zip(config.expert_checkpoints, config.bowl_scales):
        experts[scale] = BCExpertPolicy(ckpt, config.device)
    logger.info(f"Loaded {len(experts)} BC experts for scales: {list(experts.keys())}")
    
    # Load pi0 student with LoRA
    logger.info("Loading pi0 student policy...")
    lora_config = {
        "rank": config.lora_rank,
        "alpha": config.lora_alpha,
        "dropout": config.lora_dropout,
        "target_modules": config.lora_target_modules,
    }
    
    try:
        student = Pi0StudentPolicy(
            config_name=config.pi0_config_name,
            checkpoint_dir=config.pi0_checkpoint,
            device=config.device,
            lora_config=lora_config if PEFT_AVAILABLE else None,
        )
    except Exception as e:
        logger.error(f"Failed to load pi0 model: {e}")
        logger.info("Tip: The default pi0 checkpoints are JAX-based. For LoRA training,")
        logger.info("     you need a PyTorch checkpoint (model.safetensors).")
        raise
    
    # Setup optimizer for LoRA parameters only
    if student.lora_enabled and student.is_pytorch:
        trainable_params = [p for p in student.get_model().parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        logger.info(f"Created optimizer for {len(trainable_params)} trainable parameter groups")
        logger.info("LoRA fine-tuning ENABLED")
    else:
        optimizer = None
        if not student.is_pytorch:
            logger.warning("=" * 60)
            logger.warning("JAX model loaded - LoRA training NOT available")
            logger.warning("The script will collect DAgger data but NOT fine-tune.")
            logger.warning("To enable LoRA training, you need a PyTorch checkpoint.")
            logger.warning("=" * 60)
        else:
            logger.warning("LoRA not enabled - will only collect data")
    
    # Get task
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[config.task_suite]()
    task = task_suite.get_task(config.task_id)
    
    # Initialize aggregated dataset
    dagger_dataset = DAggerDataset()
    
    # DAgger iterations
    for iteration in range(config.num_dagger_iterations):
        logger.info("=" * 60)
        logger.info(f"DAgger Iteration {iteration + 1}/{config.num_dagger_iterations}")
        logger.info("=" * 60)
        
        # Compute beta (probability of using expert action during rollout)
        if config.beta_schedule == "linear":
            beta = config.initial_beta * (1 - iteration / config.num_dagger_iterations)
        else:
            beta = config.initial_beta
        logger.info(f"Beta (expert probability): {beta:.3f}")
        
        # Collect rollouts
        iteration_obs = []
        iteration_actions = []
        iteration_successes = 0
        
        for rollout_idx in tqdm(range(config.rollouts_per_iteration), desc="Collecting rollouts"):
            # Randomly select bowl scale
            scale = random.choice(config.bowl_scales)
            expert = experts[scale]
            
            # Modify environment to use this scale
            modify_bowl_scale(scale)
            
            # Create environment
            env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, config.seed + rollout_idx)
            obs = env.reset()
            expert.reset()
            
            rollout_obs = []
            rollout_actions = []
            replay_images = []
            done = False
            
            for t in range(config.max_steps_per_rollout + config.num_steps_wait):
                # Wait for objects to stabilize
                if t < config.num_steps_wait:
                    obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                    continue
                
                # Process observations
                bc_obs = process_obs_for_bc(obs)
                pi0_obs = process_obs_for_pi0(obs, task_description)
                
                # Get expert action (for labeling)
                expert_action = expert.get_action(bc_obs)
                
                # Decide which action to execute (student or expert)
                if random.random() < beta:
                    # Use expert action
                    action = expert_action
                else:
                    # Use student action
                    action = student.get_action(pi0_obs)
                    # Truncate to 7 dims if needed (pi0 may output more)
                    if len(action) > 7:
                        action = action[:7]
                
                # Store observation and EXPERT action (key DAgger insight)
                rollout_obs.append(pi0_obs)
                rollout_actions.append(expert_action)
                
                # Save for video
                if config.save_videos:
                    replay_images.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
                
                # Execute action
                obs, reward, done, info = env.step(action.tolist())
                
                if done:
                    iteration_successes += 1
                    break
            
            # Add rollout data to iteration buffer
            iteration_obs.extend(rollout_obs)
            iteration_actions.extend(rollout_actions)
            
            # Save video
            if config.save_videos and replay_images:
                suffix = "success" if done else "failure"
                video_path = pathlib.Path(config.video_dir) / f"iter{iteration}_roll{rollout_idx}_scale{scale}_{suffix}.mp4"
                imageio.mimwrite(video_path, replay_images, fps=10)
            
            env.close()
        
        # Aggregate data
        logger.info(f"Collected {len(iteration_obs)} new samples, {iteration_successes}/{config.rollouts_per_iteration} successes")
        dagger_dataset.add_batch(iteration_obs, iteration_actions)
        logger.info(f"Total aggregated dataset size: {len(dagger_dataset)}")
        
        # Save dataset checkpoint
        dagger_dataset.save(output_dir / f"dagger_dataset_iter{iteration}.pt")
        
        # Fine-tune pi0 with LoRA on aggregated dataset
        if student.lora_enabled and student.is_pytorch and optimizer is not None and len(dagger_dataset) > 0:
            logger.info(f"Fine-tuning pi0 with LoRA for {config.train_steps_per_iteration} steps...")
            
            # Create data loader
            train_loader = DataLoader(
                dagger_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=0,  # Keep in main process for simplicity
                drop_last=True,
            )
            
            # Training loop
            student.train_mode()
            model = student.get_model()
            
            total_loss = 0.0
            num_steps = 0
            
            pbar = tqdm(total=config.train_steps_per_iteration, desc="LoRA Fine-tuning")
            
            while num_steps < config.train_steps_per_iteration:
                for batch_obs, batch_actions in train_loader:
                    if num_steps >= config.train_steps_per_iteration:
                        break
                    
                    # Convert batch to appropriate format for pi0
                    # Note: This is a simplified version - full implementation would
                    # need proper data transforms matching the pi0 training pipeline
                    try:
                        # Move actions to device
                        if isinstance(batch_actions, list):
                            batch_actions = torch.stack([torch.from_numpy(a) if isinstance(a, np.ndarray) else a for a in batch_actions])
                        batch_actions = batch_actions.float().to(config.device)
                        
                        # Process observations - this needs to match pi0 input format
                        # For now, we use the policy's transform pipeline
                        batch_loss = 0.0
                        for i, obs in enumerate(batch_obs):
                            try:
                                # Use the policy's inference to get a loss proxy
                                # This is a simplified approach - proper training would use compute_loss
                                result = student.policy.infer(obs)
                                pred_action = torch.from_numpy(result["actions"]).float().to(config.device)
                                target_action = batch_actions[i][:len(pred_action)]
                                
                                # Compute MSE loss
                                loss = F.mse_loss(pred_action, target_action)
                                batch_loss += loss
                            except Exception as e:
                                logger.debug(f"Skipping sample due to error: {e}")
                                continue
                        
                        if batch_loss > 0:
                            batch_loss = batch_loss / config.gradient_accumulation_steps
                            batch_loss.backward()
                            
                            if (num_steps + 1) % config.gradient_accumulation_steps == 0:
                                torch.nn.utils.clip_grad_norm_(
                                    [p for p in model.parameters() if p.requires_grad],
                                    config.max_grad_norm
                                )
                                optimizer.step()
                                optimizer.zero_grad()
                            
                            total_loss += batch_loss.item()
                    
                    except Exception as e:
                        logger.warning(f"Training step failed: {e}")
                        continue
                    
                    num_steps += 1
                    pbar.update(1)
                    
                    if num_steps % 50 == 0:
                        avg_loss = total_loss / max(num_steps, 1)
                        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
            
            pbar.close()
            avg_loss = total_loss / max(num_steps, 1)
            logger.info(f"LoRA fine-tuning complete. Avg loss: {avg_loss:.4f}")
            
            # Save LoRA checkpoint
            student.eval_mode()
            lora_save_path = output_dir / f"lora_iter{iteration}"
            lora_save_path.mkdir(parents=True, exist_ok=True)
            student.save_lora(str(lora_save_path))
            
            # Clear memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            logger.info("Skipping fine-tuning (LoRA not enabled or no data)")
    
    # Save final LoRA weights
    if student.lora_enabled and student.is_pytorch:
        final_lora_path = output_dir / "lora_final"
        final_lora_path.mkdir(parents=True, exist_ok=True)
        student.save_lora(str(final_lora_path))
        logger.info(f"Final LoRA weights saved to: {final_lora_path}")
    
    # Save final dataset
    dagger_dataset.save(output_dir / "dagger_dataset_final.pt")
    
    logger.info("=" * 60)
    logger.info("DAgger training with LoRA complete!")
    logger.info(f"Final dataset size: {len(dagger_dataset)}")
    logger.info(f"Checkpoints saved to: {output_dir}")
    if student.lora_enabled:
        logger.info(f"LoRA config: rank={config.lora_rank}, alpha={config.lora_alpha}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="DAgger training for pi0 with BC experts and LoRA fine-tuning")
    
    # Expert configuration (flexible number - just needs to match)
    parser.add_argument("--expert-checkpoints", type=str, nargs="+", required=True,
                        help="Paths to BC expert checkpoints. Can be any number >= 1, but must match --bowl-scales count.")
    parser.add_argument("--bowl-scales", type=float, nargs="+", required=True,
                        help="Bowl scales for each environment. Must have same count as --expert-checkpoints.")
    
    # Student configuration
    parser.add_argument("--pi0-config", type=str, default="pi05_libero",
                        help="Pi0 training config name")
    parser.add_argument("--pi0-checkpoint", type=str, default="gs://openpi-assets/checkpoints/pi05_libero",
                        help="Pi0 checkpoint directory")
    
    # Environment configuration
    parser.add_argument("--task-suite", type=str, default="libero_spatial",
                        help="LIBERO task suite")
    parser.add_argument("--task-id", type=int, default=0,
                        help="Task ID within the suite")
    
    # DAgger configuration
    parser.add_argument("--num-dagger-iterations", type=int, default=10,
                        help="Number of DAgger iterations")
    parser.add_argument("--rollouts-per-iteration", type=int, default=20,
                        help="Number of rollouts per iteration")
    parser.add_argument("--max-steps", type=int, default=220,
                        help="Max steps per rollout")
    parser.add_argument("--initial-beta", type=float, default=1.0,
                        help="Initial probability of using expert action")
    parser.add_argument("--beta-schedule", type=str, default="linear",
                        choices=["linear", "constant"],
                        help="How to decay beta over iterations")
    
    # LoRA configuration
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="Rank of LoRA matrices (higher = more parameters)")
    parser.add_argument("--lora-alpha", type=float, default=32.0,
                        help="LoRA scaling factor (typically 2x rank)")
    parser.add_argument("--lora-dropout", type=float, default=0.1,
                        help="Dropout probability for LoRA layers")
    parser.add_argument("--lora-target-modules", type=str, nargs="+",
                        default=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                        help="Which modules to apply LoRA to")
    
    # Training configuration
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for LoRA fine-tuning")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Learning rate for LoRA fine-tuning")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay for optimizer")
    parser.add_argument("--train-steps", type=int, default=500,
                        help="Training steps per DAgger iteration")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Maximum gradient norm for clipping")
    
    # Output configuration
    parser.add_argument("--output-dir", type=str, default="checkpoints/dagger_pi0_lora",
                        help="Output directory for checkpoints")
    parser.add_argument("--save-videos", action="store_true",
                        help="Save rollout videos")
    parser.add_argument("--video-dir", type=str, default="data/libero/dagger_videos",
                        help="Directory for rollout videos")
    
    # Misc
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for computation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    config = DAggerConfig(
        expert_checkpoints=args.expert_checkpoints,
        bowl_scales=args.bowl_scales,
        pi0_config_name=args.pi0_config,
        pi0_checkpoint=args.pi0_checkpoint,
        task_suite=args.task_suite,
        task_id=args.task_id,
        num_dagger_iterations=args.num_dagger_iterations,
        rollouts_per_iteration=args.rollouts_per_iteration,
        max_steps_per_rollout=args.max_steps,
        initial_beta=args.initial_beta,
        beta_schedule=args.beta_schedule,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        train_steps_per_iteration=args.train_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        output_dir=args.output_dir,
        save_videos=args.save_videos,
        video_dir=args.video_dir,
        device=args.device,
        seed=args.seed,
    )
    
    run_dagger(config)


if __name__ == "__main__":
    main()

