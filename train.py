import os
import yaml
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models.mlp import ForwardKinematicsMLP
from models.mdn import ForwardKinematicsMDN
from data.dataloader import RLBenchKinematicsDataset, collate_timesteps
from data.create_dataset import collect_dataset


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def train_epoch(model, dataloader, optimizer, device, predict_orientation=False):
    model.train()
    total_loss = 0
    total_position_error = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    for joint_angles, cartesian_coords in pbar:
        joint_angles = joint_angles.to(device)
        cartesian_coords = cartesian_coords.to(device)

        if predict_orientation:
            target = cartesian_coords
        else:
            target = cartesian_coords[:, :3]

        optimizer.zero_grad()
        model_output = model(joint_angles)
        loss = model.compute_loss(model_output, target)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            predictions = model.get_prediction(model_output)
            if predict_orientation:
                position_error = torch.sqrt(
                    ((predictions[:, :3] - target[:, :3]) ** 2).sum(dim=1)
                ).mean()
            else:
                position_error = torch.sqrt(
                    ((predictions - target) ** 2).sum(dim=1)
                ).mean()

        total_loss += loss.item()
        total_position_error += position_error.item()
        num_batches += 1

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.6f}",
                "pos_err": f"{position_error.item() * 1000:.2f}mm",
            }
        )

    return total_loss / num_batches, total_position_error / num_batches


def validate(model, dataloader, device, predict_orientation=False):
    model.eval()
    total_loss = 0
    total_position_error = 0
    num_batches = 0

    with torch.no_grad():
        for joint_angles, cartesian_coords in tqdm(dataloader, desc="Validating"):
            joint_angles = joint_angles.to(device)
            cartesian_coords = cartesian_coords.to(device)

            if predict_orientation:
                target = cartesian_coords
            else:
                target = cartesian_coords[:, :3]

            model_output = model(joint_angles)
            loss = model.compute_loss(model_output, target)

            predictions = model.get_prediction(model_output)
            if predict_orientation:
                position_error = torch.sqrt(
                    ((predictions[:, :3] - target[:, :3]) ** 2).sum(dim=1)
                ).mean()
            else:
                position_error = torch.sqrt(
                    ((predictions - target) ** 2).sum(dim=1)
                ).mean()

            total_loss += loss.item()
            total_position_error += position_error.item()
            num_batches += 1

    return total_loss / num_batches, total_position_error / num_batches


def plot_training_curves(history, save_dir, model_name):
    """basic visualization of training"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # loss curves
    axes[0].plot(history["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(history["val_loss"], label="Validation Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_name.upper()} - Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Error curves
    axes[1].plot(
        np.array(history["train_pos_error"]) * 1000, label="Train Error", linewidth=2
    )
    axes[1].plot(
        np.array(history["val_pos_error"]) * 1000, label="Validation Error", linewidth=2
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Position Error (mm)")
    axes[1].set_title(f"{model_name.upper()} - Position Prediction Error")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{model_name}_training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved training curves to {save_path}")
    plt.close()


def save_training_history(history, save_dir, model_name):
    import json

    history_path = os.path.join(save_dir, f"{model_name}_history.json")

    history_serializable = {
        key: [float(val) for val in values] for key, values in history.items()
    }

    with open(history_path, "w") as f:
        json.dump(history_serializable, f, indent=2)

    print(f"Saved training history to {history_path}")


def train_model(
    model,
    model_name,
    train_dataset_root,
    val_dataset_root,
    predict_orientation=False,
    batch_size=512,
    num_epochs=100,
    learning_rate=1e-3,
    checkpoint_dir="checkpoints",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(train_dataset_root):
        collect_dataset()

    if not os.path.exists(val_dataset_root):
        print("Validation dataset not found.")
        return

    print("Loading training dataset...")
    train_dataset = RLBenchKinematicsDataset(root_dir=train_dataset_root)

    print("Loading validation dataset...")
    val_dataset = RLBenchKinematicsDataset(root_dir=val_dataset_root)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Datasets are empty. run data collection first.")
        return

    print(
        f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_timesteps,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_timesteps,
        pin_memory=True,
    )

    model = model.to(device)

    print(f"\nModel: {model_name.upper()}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    history = {
        "train_loss": [],
        "train_pos_error": [],
        "val_loss": [],
        "val_pos_error": [],
    }

    model_checkpoint_dir = os.path.join(checkpoint_dir, model_name)
    os.makedirs(model_checkpoint_dir, exist_ok=True)

    best_val_loss = float("inf")
    best_val_error = float("inf")

    print("\nStarting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss, train_pos_error = train_epoch(
            model, train_loader, optimizer, device, predict_orientation
        )

        val_loss, val_pos_error = validate(
            model, val_loader, device, predict_orientation
        )

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_pos_error"].append(train_pos_error)
        history["val_loss"].append(val_loss)
        history["val_pos_error"].append(val_pos_error)

        print(
            f"  Train Loss: {train_loss:.6f}, Train Error: {train_pos_error * 1000:.2f}mm"
        )
        print(
            f"  Val Loss:   {val_loss:.6f}, Val Error:   {val_pos_error * 1000:.2f}mm"
        )

        if val_pos_error < best_val_error:
            best_val_error = val_pos_error
            best_val_loss = val_loss
            checkpoint_path = os.path.join(model_checkpoint_dir, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_pos_error": val_pos_error,
                    "model_name": model_name,
                    "predict_orientation": predict_orientation,
                },
                checkpoint_path,
            )
            print(f"Saved best model (Val Error: {val_pos_error * 1000:.2f}mm)")

    final_checkpoint_path = os.path.join(model_checkpoint_dir, "final_model.pth")
    torch.save(
        {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_pos_error": val_pos_error,
            "model_name": model_name,
            "predict_orientation": predict_orientation,
        },
        final_checkpoint_path,
    )
    print(f"\nSaved final model to {final_checkpoint_path}")

    plot_training_curves(history, model_checkpoint_dir, model_name)
    save_training_history(history, model_checkpoint_dir, model_name)

    print("\nTraining complete!")
    print(f"Best validation position error: {best_val_error * 1000:.2f}mm")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train forward kinematics model")
    parser.add_argument(
        "--model",
        type=str,
        choices=["mlp", "mdn"],
        required=True,
        help="Model type to train (mlp or mdn)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: config/{model}.yaml)",
    )

    args = parser.parse_args()

    config_path = args.config or f"config/{args.model}.yaml"
    print(f"Loading configuration from {config_path}")
    config = load_config(config_path)

    model_cfg = config["model"]
    training_cfg = config["training"]

    if training_cfg["predict_orientation"]:
        model_cfg["output_dim"] = 6

    if args.model == "mlp":
        model = ForwardKinematicsMLP(**model_cfg)
    elif args.model == "mdn":
        model = ForwardKinematicsMDN(**model_cfg)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    print(f"\nTraining Configuration ({args.model.upper()}):")
    print("\nModel config:")
    for key, value in model_cfg.items():
        print(f"  {key}: {value}")

    print("\nTraining config:")
    for key, value in training_cfg.items():
        print(f"  {key}: {value}")

    trained_model, history = train_model(
        model,
        model_name=args.model,
        train_dataset_root=training_cfg.get(
            "train_dataset_root", "data/rlbench_kinematics_dataset/train/"
        ),
        val_dataset_root=training_cfg.get(
            "val_dataset_root", "data/rlbench_kinematics_dataset/validation/"
        ),
        predict_orientation=training_cfg["predict_orientation"],
        batch_size=training_cfg["batch_size"],
        num_epochs=training_cfg["num_epochs"],
        learning_rate=training_cfg["learning_rate"],
        checkpoint_dir=training_cfg.get("checkpoint_dir", "checkpoints"),
    )
