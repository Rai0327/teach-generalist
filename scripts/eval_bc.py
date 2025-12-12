"""
Evaluation script for BC DiffusionPolicyUNet trained on LIBERO.

This script loads a checkpoint from train_bc.py and evaluates it in the LIBERO environment.

Usage:
    python scripts/eval_bc.py --checkpoint checkpoints/diffusion_policy_libero/best_model.pt
    python scripts/eval_bc.py --checkpoint checkpoints/diffusion_policy_libero/best_model.pt --task-id 0
"""
import argparse
import collections
import logging
import math
import pathlib
import sys
from collections import OrderedDict

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from tqdm import tqdm

# Add the project root to path for imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from models.diffusion_policy import DiffusionPolicyUNet
import utils.obs_utils as ObsUtils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # Resolution used to render in environment


# =============================================================================
# Configuration Classes (same as train_bc.py)
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
):
    """Create a config object compatible with DiffusionPolicyUNet."""

    algo_config = AttrDict({
        "horizon": AttrDict({
            "observation_horizon": observation_horizon,
            "action_horizon": action_horizon,
            "prediction_horizon": prediction_horizon,
        }),
        "ddpm": AttrDict({
            "enabled": True,
            "num_train_timesteps": num_train_timesteps,
            "num_inference_timesteps": num_inference_timesteps,
            "beta_schedule": "squaredcos_cap_v2",
            "clip_sample": True,
            "prediction_type": "epsilon",
        }),
        "ddim": AttrDict({
            "enabled": False,
            "num_train_timesteps": num_train_timesteps,
            "num_inference_timesteps": num_inference_timesteps,
            "beta_schedule": "squaredcos_cap_v2",
            "clip_sample": True,
            "set_alpha_to_one": True,
            "steps_offset": 0,
            "prediction_type": "epsilon",
        }),
        "ema": AttrDict({
            "enabled": False,
            "power": 0.75,
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
                    "initial": 1e-4,
                    "decay_factor": 0.1,
                    "epoch_schedule": [],
                    "scheduler_type": "cosine",
                    "warmup_steps": 500,
                    "step_every_batch": True,
                }),
                "regularization": AttrDict({
                    "L2": 1e-6,
                }),
                "num_epochs": 100,
                "num_train_batches": 1000,
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


def _quat2axisangle(quat):
    """
    Convert quaternion to axis-angle representation.
    Copied from robosuite.
    """
    quat = quat.copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def process_image(img: np.ndarray, image_size: int = 84) -> torch.Tensor:
    """Process image: (H,W,C) uint8 -> (C,H,W) float tensor, resized."""
    img = torch.from_numpy(img.copy()).float()  # (H, W, C)
    img = img.permute(2, 0, 1)  # (C, H, W)
    img = F.interpolate(img.unsqueeze(0), size=(image_size, image_size),
                        mode='bilinear', align_corners=False)[0]
    return img  # (C, H, W) in [0, 255] range


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load trained model from checkpoint."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Model config - must match training
    observation_horizon = 2
    action_horizon = 8
    prediction_horizon = 16
    image_size = 84
    action_dim = 7

    # Observation shapes (CHW for images)
    obs_key_shapes = OrderedDict({
        "agentview_image": (3, image_size, image_size),
        "robot0_eye_in_hand_image": (3, image_size, image_size),
        "robot0_eef_pos": (3,),
        "robot0_eef_ori": (3,),
        "robot0_gripper_qpos": (1,),
    })

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
    )

    # Initialize encoder kwargs
    ObsUtils.initialize_default_obs_encoder(obs_config.encoder)

    # Create model
    model = DiffusionPolicyUNet(
        algo_config=algo_config,
        obs_config=obs_config,
        global_config=global_config,
        obs_key_shapes=obs_key_shapes,
        ac_dim=action_dim,
        device=torch.device(device),
    )

    # Load weights
    model.deserialize(checkpoint["model"])
    model.set_eval()
    logger.info(f"Loaded model from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")

    return model, observation_horizon, action_horizon, image_size


def eval_bc(
    checkpoint: str,
    task_suite_name: str = "libero_spatial",
    task_id: int | None = None,
    num_trials_per_task: int = 50,
    num_steps_wait: int = 10,
    replan_steps: int = 8,
    video_out_path: str = "data/libero/bc_videos",
    seed: int = 7,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Evaluate a trained BC diffusion policy on LIBERO tasks."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load model
    model, observation_horizon, action_horizon, image_size = load_model(checkpoint, device)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logger.info(f"Task suite: {task_suite_name}")

    pathlib.Path(video_out_path).mkdir(parents=True, exist_ok=True)

    # Max steps per task suite
    max_steps_map = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    max_steps = max_steps_map.get(task_suite_name, 300)

    # Determine which tasks to evaluate
    if task_id is not None:
        if task_id < 0 or task_id >= num_tasks_in_suite:
            raise ValueError(f"task_id must be between 0 and {num_tasks_in_suite - 1}, got {task_id}")
        task_ids = [task_id]
        logger.info(f"Evaluating single task: {task_id}")
    else:
        task_ids = list(range(num_tasks_in_suite))
        logger.info(f"Evaluating all {num_tasks_in_suite} tasks")

    # Start evaluation
    total_episodes, total_successes = 0, 0
    task_results = {}

    for tid in tqdm(task_ids, desc="Tasks"):
        task = task_suite.get_task(tid)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)

        task_episodes, task_successes = 0, 0

        for episode_idx in tqdm(range(num_trials_per_task), desc=f"Task {tid}", leave=False):
            logger.info(f"\nTask: {task_description}")

            # Reset environment and model
            obs = env.reset()
            model.reset()
            action_queue = collections.deque()

            # Observation history for temporal stacking
            obs_history = collections.deque(maxlen=observation_horizon)

            t = 0
            replay_images = []
            done = False

            logger.info(f"Starting episode {task_episodes + 1}...")

            while t < max_steps + num_steps_wait:
                try:
                    # Wait for objects to stabilize
                    if t < num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Process observations (rotate 180 degrees to match training data)
                    agentview_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    # Process images to tensor format (C, H, W)
                    agentview_tensor = process_image(agentview_img, image_size)
                    wrist_tensor = process_image(wrist_img, image_size)

                    # Get robot state
                    eef_pos = torch.from_numpy(obs["robot0_eef_pos"].copy()).float()
                    eef_ori = torch.from_numpy(_quat2axisangle(obs["robot0_eef_quat"])).float()
                    gripper_qpos = torch.from_numpy(obs["robot0_gripper_qpos"][:1].copy()).float()

                    # Current observation dict
                    current_obs = {
                        "agentview_image": agentview_tensor,
                        "robot0_eye_in_hand_image": wrist_tensor,
                        "robot0_eef_pos": eef_pos,
                        "robot0_eef_ori": eef_ori,
                        "robot0_gripper_qpos": gripper_qpos,
                    }

                    # Add to history
                    obs_history.append(current_obs)

                    # Pad history if needed
                    while len(obs_history) < observation_horizon:
                        obs_history.appendleft(current_obs)

                    # Save image for replay video
                    replay_images.append(agentview_img)

                    if len(action_queue) == 0:
                        # Build observation dict with temporal stacking
                        obs_dict = {}
                        for key in current_obs.keys():
                            # Stack observations: (T, ...) -> add batch dim -> (1, T, ...)
                            stacked = torch.stack([obs_history[i][key] for i in range(observation_horizon)], dim=0)
                            obs_dict[key] = stacked.unsqueeze(0).to(device)

                        # Get action from model
                        with torch.no_grad():
                            action = model.get_action(obs_dict)
                            action = action.squeeze(0).cpu().numpy()

                        action_queue.append(action)

                    # Execute action
                    action = action_queue.popleft()
                    obs, reward, done, info = env.step(action.tolist())

                    if done:
                        task_successes += 1
                        total_successes += 1
                        break

                    t += 1

                except Exception as e:
                    logger.error(f"Caught exception: {e}")
                    import traceback
                    traceback.print_exc()
                    break

            task_episodes += 1
            total_episodes += 1

            # Save replay video
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")[:50]
            video_path = pathlib.Path(video_out_path) / f"task{tid}_ep{episode_idx}_{suffix}.mp4"
            if replay_images:
                imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=10)

            # Log progress
            logger.info(f"Success: {done}")
            logger.info(f"Episodes: {total_episodes}, Successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        task_results[tid] = {
            "task_description": task_description,
            "episodes": task_episodes,
            "successes": task_successes,
            "success_rate": task_successes / task_episodes if task_episodes > 0 else 0,
        }
        logger.info(f"Task {tid} success rate: {task_results[tid]['success_rate']:.2%}")

        env.close()

    # Final results
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    for tid, result in task_results.items():
        logger.info(f"Task {tid}: {result['success_rate']:.2%} ({result['successes']}/{result['episodes']})")
        logger.info(f"  -> {result['task_description']}")
    logger.info("-" * 60)
    logger.info(f"Total: {total_successes}/{total_episodes} ({total_successes / total_episodes * 100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate BC DiffusionPolicyUNet on LIBERO")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (e.g., checkpoints/diffusion_policy_libero/best_model.pt)")
    parser.add_argument("--task-suite-name", type=str, default="libero_spatial",
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
                        help="LIBERO task suite to evaluate on")
    parser.add_argument("--task-id", type=int, default=None,
                        help="Specific task ID to evaluate (0-9 for libero_spatial). If not specified, evaluates all tasks.")
    parser.add_argument("--num-trials-per-task", type=int, default=50,
                        help="Number of evaluation episodes per task")
    parser.add_argument("--replan-steps", type=int, default=8,
                        help="Number of steps before replanning (should match action_horizon)")
    parser.add_argument("--video-out-path", type=str, default="data/libero/bc_videos",
                        help="Directory to save evaluation videos")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run evaluation on")
    args = parser.parse_args()

    eval_bc(
        checkpoint=args.checkpoint,
        task_suite_name=args.task_suite_name,
        task_id=args.task_id,
        num_trials_per_task=args.num_trials_per_task,
        replan_steps=args.replan_steps,
        video_out_path=args.video_out_path,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()

