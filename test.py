import os
import numpy as np
from collections import defaultdict
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.tasks.empty_task import EmptyTask

from data.dataloader import RLBenchKinematicsDataset


def main(
    root_dir,
    output_dir="videos",
    fps=30,
):
    dataset = RLBenchKinematicsDataset(root_dir=root_dir, split="train")
    if len(dataset) == 0:
        print("Dataset is empty. Run the data collection script first.")
        return
    env = Environment(
        action_mode=MoveArmThenGripper(JointPosition(), Discrete()),
        headless=False,
        robot_setup="panda",
    )

    env.launch()

    samples_by_task = defaultdict(list)
    for i in range(len(dataset)):
        sample = dataset[i]
        samples_by_task[sample["task"]].append((i, sample))

    print("samples by task:", samples_by_task.keys())

    empty_task = env.get_task(EmptyTask)
    empty_task.reset()

    for task_name, task_samples in samples_by_task.items():
        # task = env.get_task(task_class)
        task = env.get_task(EmptyTask)

        # Replace the randomized scene with the empty one
        task._scene = empty_task._scene
        env._scene = empty_task._scene

        trajectory_data = sample["joint_angles"].numpy()

        if trajectory_data.ndim == 1:
            trajectory_data = trajectory_data.reshape(1, -1)

        if trajectory_data.ndim != 2:
            print(
                f"Warning: Unexpected trajectory shape {trajectory_data.shape} for {task_name}"
            )
            continue

        if trajectory_data.shape[1] < 7:
            print(f"Warning: Not enough joint angles in trajectory for {task_name}")
            continue

        if trajectory_data.shape[1] == 7:
            gripper_states = np.ones((trajectory_data.shape[0], 1))
            trajectory_data = np.hstack([trajectory_data, gripper_states])
        print("task:", task)

        if not env._robot:
            raise RuntimeError("Robot not initialized")

        if not env._scene:
            raise RuntimeError("Scene not initialized")

        for step_idx, step_data in enumerate(trajectory_data):
            joint_positions = step_data[:7].tolist()
            gripper_state = float(step_data[7]) if step_data.shape[0] > 7 else 1.0

            arm = env._robot.arm
            arm.set_joint_target_positions(joint_positions)

            gripper = env._robot.gripper
            gripper.actuate(gripper_state, velocity=0.04)

            for _ in range(10):
                env._scene.step()

            obs = env._scene.get_observation()


if __name__ == "__main__":
    DATASET_ROOT = "rlbench_kinematics_dataset"
    VIDEO_OUTPUT_DIR = "task_videos"

    if os.path.exists(DATASET_ROOT):
        main(
            root_dir=DATASET_ROOT,
            output_dir=VIDEO_OUTPUT_DIR,
            fps=30,
        )
    else:
        print(f"Error: Dataset root directory '{DATASET_ROOT}' not found.")
