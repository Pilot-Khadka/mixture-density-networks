import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


def collate_timesteps(batch):
    joint_angles_list = []
    cartesian_coords_list = []

    for item in batch:
        joint_angles = item["joint_angles"]
        cartesian_coords = item["cartesian_coords"]
        joint_angles_list.append(joint_angles)
        cartesian_coords_list.append(cartesian_coords)

    joint_angles_batch = torch.cat(joint_angles_list, dim=0)
    cartesian_coords_batch = torch.cat(cartesian_coords_list, dim=0)

    return joint_angles_batch, cartesian_coords_batch


def collate_fn(batch):
    joint_angles, cartesian_coords, task_names = zip(*batch)
    return list(joint_angles), list(cartesian_coords), list(task_names)


class RLBenchKinematicsDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []

        if not os.path.exists(root_dir):
            raise FileNotFoundError("Dataset dir not found")

        for task_name in os.listdir(root_dir):
            task_path = os.path.join(root_dir, task_name)

            if not os.path.isdir(task_path):
                continue

            metadata_path = os.path.join(task_path, "metadata.json")

            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                for trial_info in metadata["trials"]:
                    joint_angles_path = os.path.join(
                        task_path, trial_info["joint_angles_file"]
                    )

                    cartesian_path = os.path.join(
                        task_path, trial_info["cartesian_file"]
                    )

                    if os.path.exists(joint_angles_path) and os.path.exists(
                        cartesian_path
                    ):
                        self.samples.append(
                            {
                                "joint_angles_path": joint_angles_path,
                                "cartesian_path": cartesian_path,
                                "task": task_name,
                                "trial": trial_info["trial"],
                                "descriptions": trial_info.get("descriptions", []),
                                "num_steps": trial_info.get("num_steps", 0),
                            }
                        )
            else:
                # fallback when metadata.json doesn't exist: match joint/cartesian pairs by trial number
                files = sorted([f for f in os.listdir(task_path) if f.endswith(".npy")])
                joint_files = [f for f in files if "joint_angles" in f]
                cartesian_files = [f for f in files if "cartesian" in f]
                for joint_file in joint_files:
                    # filename format: trial_XXX_joint_angles.npy
                    trial_num = joint_file.split("_")[1]
                    cartesian_file = f"trial_{trial_num}_cartesian.npy"
                    if cartesian_file in cartesian_files:
                        self.samples.append(
                            {
                                "joint_angles_path": os.path.join(
                                    task_path, joint_file
                                ),
                                "cartesian_path": os.path.join(
                                    task_path, cartesian_file
                                ),
                                "task": task_name,
                                "trial": int(trial_num),
                                "descriptions": [],
                                "num_steps": 0,
                            }
                        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        joint_angles = np.load(sample_info["joint_angles_path"])
        cartesian_coords = np.load(sample_info["cartesian_path"])
        return {
            "joint_angles": torch.from_numpy(joint_angles).float(),
            "cartesian_coords": torch.from_numpy(cartesian_coords).float(),
            "task": sample_info["task"],
            "trial": sample_info["trial"],
            "descriptions": sample_info["descriptions"],
        }


def create_rlbench_dataloader(root_dir, batch_size=8, shuffle=True, num_workers=4):
    dataset = RLBenchKinematicsDataset(root_dir=root_dir)

    def collate_batch(batch):
        joint_angles = [item["joint_angles"] for item in batch]
        cartesian_coords = [item["cartesian_coords"] for item in batch]
        tasks = [item["task"] for item in batch]
        trials = [item["trial"] for item in batch]
        descriptions = [item["descriptions"] for item in batch]

        return {
            "joint_angles": joint_angles,
            "cartesian_coords": cartesian_coords,
            "tasks": tasks,
            "trials": trials,
            "descriptions": descriptions,
        }

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_batch,
    )
    return loader


if __name__ == "__main__":
    dataset_root = "rlbench_kinematics_dataset"
    dataloader = create_rlbench_dataloader(dataset_root, batch_size=4)

    print(f"Dataset size: {len(dataloader.dataset)} samples")
    print(f"Number of batches: {len(dataloader)}\n")

    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        joint_angles = batch["joint_angles"]
        cartesian_coords = batch["cartesian_coords"]
        tasks = batch["tasks"]

        for i in range(len(joint_angles)):
            print(f"  Sample {i}:")
            print(f"    Task: {tasks[i]}")
            print(f"    Joint angles shape: {joint_angles[i].shape}")
            print(f"    Cartesian coords shape: {cartesian_coords[i].shape}")
            print(f"    Joint angles (first timestep): {joint_angles[i][0]}")
            print(f"    Cartesian coords (first timestep): {cartesian_coords[i][0]}")
            print()

        break
