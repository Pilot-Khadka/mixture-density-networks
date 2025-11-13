import os
import json
import random
import numpy as np
from tqdm import tqdm
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import MT15_V1


HEADLESS = True
TASK_SET = MT15_V1
OUTPUT_ROOT = "rlbench_kinematics_dataset"
NUM_TRIALS_PER_TASK = 50
SEED = 42


def setup_env():
    obs = ObservationConfig()
    obs.set_all(False)
    obs.joint_positions = True
    obs.gripper_open = True
    obs.gripper_pose = True

    env = Environment(
        action_mode=MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=obs,
        headless=HEADLESS,
    )
    env.launch()
    return env


def verify_tasks(env, task_classes):
    available = []
    for task_cls in tqdm(task_classes, desc="Verifying tasks", disable=not HEADLESS):
        try:
            env.get_task(task_cls)
            available.append(task_cls)
        except Exception as e:
            print(f"Skipping {task_cls.__name__}: not available ({e})")
    return available


def collect_for_task(env, task_class, num_trials, out_folder):
    task = env.get_task(task_class)
    os.makedirs(out_folder, exist_ok=True)

    iterator = tqdm(
        range(num_trials), desc=f"{task_class.__name__}", disable=not HEADLESS
    )
    metadata_list = []

    for trial in iterator:
        try:
            descriptions, obs = task.reset()
            demos = task.get_demos(1, live_demos=True)
        except Exception as e:
            iterator.write(f"Error collecting demo for {task_class.__name__}: {e}")
            continue

        if not demos:
            iterator.write(f"Warning: no demo for {task_class.__name__}, trial {trial}")
            continue

        demo = demos[0]
        joint_angles_list, cartesian_coords_list = [], []

        for obs in demo:
            jp = getattr(obs, "joint_positions", None)
            gp = getattr(obs, "gripper_pose", None)
            if jp is None or gp is None:
                continue

            gr = float(obs.gripper_open)
            joint_angles_list.append(np.concatenate([jp, [gr]]))
            cartesian_coords_list.append(gp)

        if not joint_angles_list or not cartesian_coords_list:
            iterator.write(
                f"Warning: demo had no kinematic data, skipping trial {trial}"
            )
            continue

        joint_angles_arr = np.array(joint_angles_list, dtype=np.float32)
        cartesian_coords_arr = np.array(cartesian_coords_list, dtype=np.float32)

        joint_angles_file = f"trial_{trial:03d}_joint_angles.npy"
        cartesian_file = f"trial_{trial:03d}_cartesian.npy"

        np.save(os.path.join(out_folder, joint_angles_file), joint_angles_arr)
        np.save(os.path.join(out_folder, cartesian_file), cartesian_coords_arr)

        metadata_list.append(
            {
                "trial": trial,
                "joint_angles_file": joint_angles_file,
                "cartesian_file": cartesian_file,
                "num_steps": len(joint_angles_list),
                "joint_angles_shape": joint_angles_arr.shape,
                "cartesian_shape": cartesian_coords_arr.shape,
                "descriptions": descriptions if descriptions else [],
            }
        )

    metadata_path = os.path.join(out_folder, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(
            {
                "task_name": task_class.__name__,
                "num_trials": len(metadata_list),
                "trials": metadata_list,
            },
            f,
            indent=2,
        )


def collect_dataset(num_trials=NUM_TRIALS_PER_TASK):
    env = setup_env()

    all_tasks = TASK_SET["train"]
    available_tasks = verify_tasks(env, all_tasks)

    random.seed(SEED)
    random.shuffle(available_tasks)

    split_index = int(0.8 * len(available_tasks))
    train_tasks = available_tasks[:split_index]
    val_tasks = available_tasks[split_index:]

    for subset_name, task_list in [("train", train_tasks), ("validation", val_tasks)]:
        subset_root = os.path.join(OUTPUT_ROOT, subset_name)
        os.makedirs(subset_root, exist_ok=True)

        outer = tqdm(task_list, desc=f"{subset_name} tasks", disable=not HEADLESS)
        for task_cls in outer:
            out_folder = os.path.join(subset_root, task_cls.__name__)
            collect_for_task(env, task_cls, num_trials, out_folder)

    env.shutdown()


if __name__ == "__main__":
    collect_dataset()
