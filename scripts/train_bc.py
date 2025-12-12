"""
BC training script for DiffusionPolicyUNet on Libero libero_spatial data.

This script uses the existing robomimic-style DiffusionPolicyUNet from models/diffusion_policy.py
and loads data directly from libero's HDF5 demo files (~2-3GB for libero_spatial only).

Usage:
    python scripts/train_bc.py
"""
import logging
import os
import sys
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm import tqdm

# Add the project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Robomimic imports
from models.diffusion_policy import DiffusionPolicyUNet
import utils.obs_utils as ObsUtils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Classes (minimal robomimic-style configs)
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


def create_diffusion_policy_config(
    observation_horizon: int = 2,
    action_horizon: int = 8,
    prediction_horizon: int = 16,
    num_train_timesteps: int = 100,
    num_inference_timesteps: int = 10,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-6,
    num_epochs: int = 100,
    num_train_batches: int = 1000,
    use_ddpm: bool = True,
    ema_power: float = 0.75,
):
    """Create a config object compatible with DiffusionPolicyUNet."""
    
    algo_config = AttrDict({
        "horizon": AttrDict({
            "observation_horizon": observation_horizon,
            "action_horizon": action_horizon,
            "prediction_horizon": prediction_horizon,
        }),
        "ddpm": AttrDict({
            "enabled": use_ddpm,
            "num_train_timesteps": num_train_timesteps,
            "num_inference_timesteps": num_inference_timesteps,
            "beta_schedule": "squaredcos_cap_v2",
            "clip_sample": True,
            "prediction_type": "epsilon",
        }),
        "ddim": AttrDict({
            "enabled": not use_ddpm,
            "num_train_timesteps": num_train_timesteps,
            "num_inference_timesteps": num_inference_timesteps,
            "beta_schedule": "squaredcos_cap_v2",
            "clip_sample": True,
            "set_alpha_to_one": True,
            "steps_offset": 0,
            "prediction_type": "epsilon",
        }),
        "ema": AttrDict({
            "enabled": False,  # Disabled due to diffusers API change
            "power": ema_power,
        }),
        "unet": AttrDict({
            "enabled": True,
        }),
        "transformer": AttrDict({
            "enabled": False,
        }),
        "optim_params": AttrDict({
            "policy": AttrDict({
                "optimizer_type": "adamw",
                "learning_rate": AttrDict({
                    "initial": learning_rate,
                    "decay_factor": 0.1,
                    "epoch_schedule": [],
                    "scheduler_type": "cosine",
                    "warmup_steps": 500,
                    "step_every_batch": True,
                }),
                "regularization": AttrDict({
                    "L2": weight_decay,
                }),
                "num_epochs": num_epochs,
                "num_train_batches": num_train_batches,
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
                    "backbone_kwargs": AttrDict({
                        "pretrained": False,
                        "input_coord_conv": False,
                    }),
                    "pool_class": "SpatialSoftmax",
                    "pool_kwargs": AttrDict({
                        "num_kp": 32,
                        "learnable_temperature": False,
                        "temperature": 1.0,
                        "noise_std": 0.0,
                    }),
                }),
                "obs_randomizer_class": None,
                "obs_randomizer_kwargs": AttrDict({}),
            }),
            "low_dim": AttrDict({
                "core_class": None,
                "core_kwargs": AttrDict({}),
                "obs_randomizer_class": None,
                "obs_randomizer_kwargs": AttrDict({}),
            }),
        }),
    })
    
    global_config = AttrDict({
        "algo_name": "diffusion_policy",
        "all_obs_keys": ["agentview_image", "robot0_eye_in_hand_image", 
                        "robot0_eef_pos", "robot0_eef_ori", "robot0_gripper_qpos"],
    })
    
    return algo_config, obs_config, global_config


# =============================================================================
# Libero Data Download
# =============================================================================
def download_libero_spatial(download_dir: str) -> None:
    """Download only the libero_spatial dataset (~2-3GB)."""
    import urllib.request
    import zipfile
    
    url = "https://utexas.box.com/shared/static/04k94hyizn4huhbv5sz4ev9p2h1p6s7f.zip"
    
    download_path = Path(download_dir)
    download_path.mkdir(parents=True, exist_ok=True)
    
    spatial_dir = download_path / "libero_spatial"
    if spatial_dir.exists() and len(list(spatial_dir.glob("*.hdf5"))) == 10:
        logger.info(f"libero_spatial already downloaded at {spatial_dir}")
        return
    
    zip_path = download_path / "libero_spatial.zip"
    
    logger.info(f"Downloading libero_spatial dataset to {download_path}...")
    logger.info("This is ~2-3GB and may take a few minutes...")
    
    # Download with progress bar
    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 // total_size) if total_size > 0 else 0
        print(f"\rDownloading: {percent}% ({downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB)", end="")
    
    urllib.request.urlretrieve(url, zip_path, reporthook)
    print()  # newline after progress
    
    logger.info("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(download_path)
    
    # Clean up zip file
    zip_path.unlink()
    logger.info(f"libero_spatial dataset ready at {spatial_dir}")


# =============================================================================
# Dataset - Direct HDF5 Loading
# =============================================================================
class LiberoHDF5Dataset(Dataset):
    """Load libero demo data directly from HDF5 files."""
    
    def __init__(
        self,
        hdf5_path: str,
        observation_horizon: int = 2,
        prediction_horizon: int = 16,
        image_size: int = 84,
    ):
        self.observation_horizon = observation_horizon
        self.prediction_horizon = prediction_horizon
        self.image_size = image_size
        self.hdf5_path = hdf5_path
        
        # Load data into memory for faster training
        logger.info(f"Loading {hdf5_path}...")
        with h5py.File(hdf5_path, 'r') as f:
            self.demos = []
            demo_keys = sorted([k for k in f['data'].keys() if k.startswith('demo_')])
            
            for demo_key in demo_keys:
                demo_grp = f['data'][demo_key]
                obs_grp = demo_grp['obs']
                
                demo_data = {
                    'agentview_rgb': obs_grp['agentview_rgb'][()],  # (T, 128, 128, 3)
                    'eye_in_hand_rgb': obs_grp['eye_in_hand_rgb'][()],  # (T, 128, 128, 3)
                    'ee_pos': obs_grp['ee_pos'][()],  # (T, 3)
                    'ee_ori': obs_grp['ee_ori'][()],  # (T, 3) - orientation, not quaternion
                    'gripper_states': obs_grp['gripper_states'][()],  # (T, 2)
                    'actions': demo_grp['actions'][()],  # (T, 7)
                }
                self.demos.append(demo_data)
        
        # Build valid indices (demo_idx, timestep) pairs
        self.valid_indices = []
        for demo_idx, demo in enumerate(self.demos):
            T = len(demo['actions'])
            for t in range(observation_horizon - 1, T - prediction_horizon):
                self.valid_indices.append((demo_idx, t))
        
        logger.info(f"  Loaded {len(self.demos)} demos, {len(self.valid_indices)} valid samples")

    def __len__(self):
        return len(self.valid_indices)
    
    def _process_image(self, img: np.ndarray) -> torch.Tensor:
        """Process image: (H,W,C) uint8 -> (H,W,C) float, resized.
        
        Keep in HWC format - robomimic will convert to CHW internally.
        """
        img = torch.from_numpy(img.copy()).float()  # (H, W, C)
        # Resize: need to go HWC -> CHW -> resize -> CHW -> HWC
        img = img.permute(2, 0, 1)  # (C, H, W)
        img = F.interpolate(img.unsqueeze(0), size=(self.image_size, self.image_size), 
                           mode='bilinear', align_corners=False)[0]
        img = img.permute(1, 2, 0)  # Back to (H, W, C)
        return img  # Keep [0, 255] range and HWC format for robomimic

    def __getitem__(self, idx):
        demo_idx, center_t = self.valid_indices[idx]
        demo = self.demos[demo_idx]
        
        # Collect observations over observation_horizon
        obs_dict = {
            "agentview_image": [],
            "robot0_eye_in_hand_image": [],
            "robot0_eef_pos": [],
            "robot0_eef_ori": [],
            "robot0_gripper_qpos": [],
        }
        
        for i in range(self.observation_horizon):
            t = center_t - (self.observation_horizon - 1 - i)
            
            obs_dict["agentview_image"].append(self._process_image(demo['agentview_rgb'][t]))
            obs_dict["robot0_eye_in_hand_image"].append(self._process_image(demo['eye_in_hand_rgb'][t]))
            obs_dict["robot0_eef_pos"].append(torch.from_numpy(demo['ee_pos'][t].copy()).float())
            obs_dict["robot0_eef_ori"].append(torch.from_numpy(demo['ee_ori'][t].copy()).float())
            obs_dict["robot0_gripper_qpos"].append(torch.from_numpy(demo['gripper_states'][t, :1].copy()).float())  # Take first gripper dim
        
        # Stack observations
        for k in obs_dict:
            obs_dict[k] = torch.stack(obs_dict[k], dim=0)
        
        # Collect actions over prediction_horizon
        actions = []
        for i in range(self.prediction_horizon):
            t = center_t + i
            actions.append(torch.from_numpy(demo['actions'][t]).float())
        actions = torch.stack(actions, dim=0)
        
        # Normalize actions to [-1, 1]
        actions = torch.clamp(actions, -1.0, 1.0)
        
        return {
            "obs": obs_dict,
            "actions": actions,
            "goal_obs": None,
        }


def collate_fn(batch):
    """Custom collate function for robomimic-style batches."""
    obs_keys = batch[0]["obs"].keys()
    return {
        "obs": {k: torch.stack([b["obs"][k] for b in batch], dim=0) for k in obs_keys},
        "actions": torch.stack([b["actions"] for b in batch], dim=0),
        "goal_obs": None,
    }


# =============================================================================
# Training Loop
# =============================================================================
def train_loop(
    # Dataset config
    data_dir: str = "data/libero",  # Where to download/find libero data
    task_suite: str = "libero_spatial",  # Only libero_spatial supported for now
    task_id: int | None = None,  # Specific task ID to train on (0-9 for libero_spatial). If None, trains on all tasks.
    
    # Model config
    observation_horizon: int = 2,
    action_horizon: int = 8,
    prediction_horizon: int = 16,
    image_size: int = 84,
    
    # Training config
    batch_size: int = 64,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-6,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    
    # Wandb config
    use_wandb: bool = True,
    wandb_project: str = "bc-diffusion-policy",
    wandb_entity: str = None,
    wandb_run_name: str = None,
):
    logger.info("=" * 60)
    logger.info("Starting BC training with DiffusionPolicyUNet")
    logger.info(f"Task suite: {task_suite}")
    logger.info(f"Task ID: {task_id if task_id is not None else 'all'}")
    logger.info("=" * 60)
    
    # Create output directory
    output_dir = Path("checkpoints/diffusion_policy_libero")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    if use_wandb:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            config={
                "task_suite": task_suite,
                "task_id": task_id,
                "observation_horizon": observation_horizon,
                "action_horizon": action_horizon,
                "prediction_horizon": prediction_horizon,
                "image_size": image_size,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "device": device,
            },
        )
        logger.info(f"Wandb initialized: {wandb.run.name}")
    
    # Download libero_spatial if needed
    data_path = Path(data_dir)
    download_libero_spatial(str(data_path))
    
    # Load HDF5 files for the task suite
    task_dir = data_path / task_suite
    all_hdf5_files = sorted(task_dir.glob("*.hdf5"))
    
    if not all_hdf5_files:
        raise FileNotFoundError(f"No HDF5 files found in {task_dir}")
    
    # Select specific task or all tasks
    if task_id is not None:
        if task_id < 0 or task_id >= len(all_hdf5_files):
            raise ValueError(f"task_id must be between 0 and {len(all_hdf5_files) - 1}, got {task_id}")
        hdf5_files = [all_hdf5_files[task_id]]
        logger.info(f"Training on single task {task_id}: {hdf5_files[0].name}")
    else:
        hdf5_files = all_hdf5_files
        logger.info(f"Training on all {len(hdf5_files)} tasks")
    
    # Create datasets for each task and concatenate
    datasets = []
    for hdf5_file in hdf5_files:
        ds = LiberoHDF5Dataset(
            hdf5_path=str(hdf5_file),
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            image_size=image_size,
        )
        datasets.append(ds)
    
    dataset = ConcatDataset(datasets)
    logger.info(f"Total samples: {len(dataset)}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    
    num_train_batches = len(dataloader)
    
    # Get observation shapes from a sample batch
    # Note: Images are HWC in dataset but robomimic converts to CHW internally
    # So we need to pass CHW shapes for images
    sample_batch = next(iter(dataloader))
    obs_key_shapes = OrderedDict()
    rgb_keys = {"agentview_image", "robot0_eye_in_hand_image"}
    for k, v in sample_batch["obs"].items():
        shape = tuple(v.shape[2:])  # (B, T, ...) -> (...)
        if k in rgb_keys:
            # Convert HWC (H, W, C) -> CHW (C, H, W) shape
            shape = (shape[2], shape[0], shape[1])
        obs_key_shapes[k] = shape
    
    action_dim = sample_batch["actions"].shape[-1]
    
    logger.info(f"Observation shapes: {obs_key_shapes}")
    logger.info(f"Action dimension: {action_dim}")
    
    # Initialize observation utilities
    ObsUtils.initialize_obs_modality_mapping_from_dict({
        "rgb": ["agentview_image", "robot0_eye_in_hand_image"],
        "low_dim": ["robot0_eef_pos", "robot0_eef_ori", "robot0_gripper_qpos"],
    })
    
    # Create configs
    algo_config, obs_config, global_config = create_diffusion_policy_config(
        observation_horizon=observation_horizon,
        action_horizon=action_horizon,
        prediction_horizon=prediction_horizon,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        num_train_batches=num_train_batches,
    )
    
    # Initialize default encoder kwargs
    ObsUtils.initialize_default_obs_encoder(obs_config.encoder)
    
    # Create model
    logger.info("Creating DiffusionPolicyUNet model...")
    model = DiffusionPolicyUNet(
        algo_config=algo_config,
        obs_config=obs_config,
        global_config=global_config,
        obs_key_shapes=obs_key_shapes,
        ac_dim=action_dim,
        device=torch.device(device),
    )
    
    num_params = sum(p.numel() for p in model.nets.parameters() if p.requires_grad)
    logger.info(f"Model has {num_params:,} trainable parameters")
    
    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs...")
    best_loss = float('inf')
    
    global_step = 0
    for epoch in range(num_epochs):
        model.set_train()
        epoch_losses = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            # Process and move batch to device
            batch = model.process_batch_for_training(batch)
            batch = model.postprocess_batch_for_training(batch, obs_normalization_stats=None)
            
            # Training step
            info = model.train_on_batch(batch, epoch, validate=False)
            model.on_gradient_step()
            
            loss = info["losses"]["l2_loss"].item()
            epoch_losses.append(loss)
            pbar.set_postfix({"loss": f"{loss:.4f}"})
            
            # Log to wandb
            if use_wandb:
                wandb.log({
                    "train/loss": loss,
                    "train/step": global_step,
                }, step=global_step)
            global_step += 1
        
        model.on_epoch_end(epoch)
        
        avg_loss = np.mean(epoch_losses)
        logger.info(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}")
        
        # Log epoch metrics to wandb
        if use_wandb:
            wandb.log({
                "train/epoch_loss": avg_loss,
                "train/best_loss": best_loss,
                "epoch": epoch + 1,
            }, step=global_step)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model": model.serialize(),
                "loss": avg_loss,
            }, output_dir / "best_model.pt")
            logger.info(f"  -> Saved new best model (loss={avg_loss:.4f})")
        
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model": model.serialize(),
                "loss": avg_loss,
            }, output_dir / f"checkpoint_epoch{epoch+1}.pt")
    
    logger.info("Training complete!")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info(f"Checkpoints saved to: {output_dir}")
    
    # Finish wandb run
    if use_wandb:
        wandb.finish()
    

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train BC with DiffusionPolicyUNet on LIBERO")
    parser.add_argument("--task-id", type=int, default=None, 
                        help="Specific task ID to train on (0-9 for libero_spatial). If not specified, trains on all tasks.")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Wandb run name")
    args = parser.parse_args()
    
    train_loop(
        task_id=args.task_id,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_wandb=not args.no_wandb,
        wandb_run_name=args.wandb_run_name,
    )


if __name__ == "__main__":
    main()
