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


def test_model(model, dataloader, device, predict_orientation=False, visualize=True):
    model.eval()
    all_predictions = []
    all_targets = []
    all_errors = []

    with torch.no_grad():
        for joint_angles, cartesian_coords in tqdm(dataloader, desc="Testing"):
            joint_angles = joint_angles.to(device)
            cartesian_coords = cartesian_coords.to(device)

            if predict_orientation:
                target = cartesian_coords
            else:
                target = cartesian_coords[:, :3]

            model_output = model(joint_angles)
            predictions = model.get_prediction(model_output)

            if predict_orientation:
                position_error = torch.sqrt(
                    ((predictions[:, :3] - target[:, :3]) ** 2).sum(dim=1)
                )
            else:
                position_error = torch.sqrt(((predictions - target) ** 2).sum(dim=1))

            all_predictions.append(predictions.cpu())
            all_targets.append(target.cpu())
            all_errors.append(position_error.cpu())

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_errors = torch.cat(all_errors, dim=0)

    mean_error = all_errors.mean().item()
    std_error = all_errors.std().item()
    median_error = all_errors.median().item()
    max_error = all_errors.max().item()

    print(f"Mean Position Error:   {mean_error * 1000:.2f} mm")
    print(f"Std Position Error:    {std_error * 1000:.2f} mm")
    print(f"Median Position Error: {median_error * 1000:.2f} mm")
    print(f"Max Position Error:    {max_error * 1000:.2f} mm")

    if visualize:
        visualize_test_results(
            all_predictions, all_targets, all_errors, predict_orientation
        )

    return {
        "predictions": all_predictions,
        "targets": all_targets,
        "errors": all_errors,
        "mean_error": mean_error,
        "std_error": std_error,
        "median_error": median_error,
        "max_error": max_error,
    }


def visualize_test_results(
    predictions, targets, errors, predict_orientation, save_path="test_results.png"
):
    predictions_np = predictions.numpy()
    targets_np = targets.numpy()
    errors_np = errors.numpy() * 1000

    if predict_orientation:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        for i, axis_name in enumerate(["X", "Y", "Z"]):
            ax = axes[i // 2, i % 2] if i < 3 else None
            if ax:
                ax.scatter(targets_np[:, i], predictions_np[:, i], alpha=0.3, s=1)
                ax.plot(
                    [targets_np[:, i].min(), targets_np[:, i].max()],
                    [targets_np[:, i].min(), targets_np[:, i].max()],
                    "r--",
                    linewidth=2,
                )
                ax.set_xlabel(f"True {axis_name} (m)")
                ax.set_ylabel(f"Predicted {axis_name} (m)")
                ax.set_title(f"{axis_name}-axis Predictions")
                ax.grid(True, alpha=0.3)

        axes[1, 1].hist(errors_np, bins=50, alpha=0.7, edgecolor="black")
        axes[1, 1].axvline(
            errors_np.mean(),
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {errors_np.mean():.2f}mm",
        )
        axes[1, 1].axvline(
            np.median(errors_np),
            color="g",
            linestyle="--",
            linewidth=2,
            label=f"Median: {np.median(errors_np):.2f}mm",
        )
        axes[1, 1].set_xlabel("Position Error (mm)")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].set_title("Error Distribution")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        for i, axis_name in enumerate(["X", "Y", "Z"]):
            ax = axes[i // 2, i % 2]
            ax.scatter(targets_np[:, i], predictions_np[:, i], alpha=0.3, s=1)
            ax.plot(
                [targets_np[:, i].min(), targets_np[:, i].max()],
                [targets_np[:, i].min(), targets_np[:, i].max()],
                "r--",
                linewidth=2,
            )
            ax.set_xlabel(f"True {axis_name} (m)")
            ax.set_ylabel(f"Predicted {axis_name} (m)")
            ax.set_title(f"{axis_name}-axis Predictions")
            ax.grid(True, alpha=0.3)

        axes[1, 1].hist(errors_np, bins=50, alpha=0.7, edgecolor="black")
        axes[1, 1].axvline(
            errors_np.mean(),
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {errors_np.mean():.2f}mm",
        )
        axes[1, 1].axvline(
            np.median(errors_np),
            color="g",
            linestyle="--",
            linewidth=2,
            label=f"Median: {np.median(errors_np):.2f}mm",
        )
        axes[1, 1].set_xlabel("Position Error (mm)")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].set_title("Error Distribution")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved test visualization to {save_path}")
    plt.close()


def plot_training_curves(history, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(history["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(history["val_loss"], label="Validation Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        np.array(history["train_pos_error"]) * 1000, label="Train Error", linewidth=2
    )
    axes[1].plot(
        np.array(history["val_pos_error"]) * 1000, label="Validation Error", linewidth=2
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Position Error (mm)")
    axes[1].set_title("Position Prediction Error")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved training curves to {save_path}")
    plt.close()


def train_model(
    model,
    dataset_root,
    predict_orientation=False,
    batch_size=512,
    num_epochs=100,
    learning_rate=1e-3,
    checkpoint_dir="checkpoints",
    test_after_training=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(dataset_root):
        print("Dataset is empty. Creating the dataset first....")
        collect_dataset()

    print("Loading dataset...")
    dataset = RLBenchKinematicsDataset(root_dir=dataset_root)

    if len(dataset) == 0:
        print("Dataset is empty. Please run data collection first.")
        return

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    print(f"Train samples: {train_size}, Validation samples: {val_size}")

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

    print("\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

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

    os.makedirs(checkpoint_dir, exist_ok=True)

    best_val_loss = float("inf")

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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_pos_error": val_pos_error,
                    "model_config": model.__dict__
                    if hasattr(model, "__dict__")
                    else {},
                },
                checkpoint_path,
            )
            print(f"  Saved best model to {checkpoint_path}")

    final_checkpoint_path = os.path.join(checkpoint_dir, "final_model.pth")
    torch.save(
        {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_pos_error": val_pos_error,
            "model_config": model.__dict__ if hasattr(model, "__dict__") else {},
        },
        final_checkpoint_path,
    )
    print(f"\nSaved final model to {final_checkpoint_path}")

    plot_training_curves(history, checkpoint_dir)

    if test_after_training:
        print("\nRunning test evaluation...")
        test_results = test_model(
            model,
            val_loader,
            device,
            predict_orientation,
            visualize=True,
        )

    return model, history


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train forward kinematics model")
    parser.add_argument(
        "--model",
        type=str,
        choices=["mlp", "mdn"],
        required=True,
        help="Model type to train (mlp or mdn)",
    )

    args = parser.parse_args()

    config_path = f"config/{args.model}.yaml"
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
    print(f"Model type: {args.model}")

    print("\nModel config:")
    for key, value in model_cfg.items():
        print(f"  {key}: {value}")

    print("\nTraining config:")
    for key, value in training_cfg.items():
        print(f"  {key}: {value}")

    trained_model, history = train_model(
        model,
        dataset_root=training_cfg["dataset_root"],
        predict_orientation=training_cfg["predict_orientation"],
        batch_size=training_cfg["batch_size"],
        num_epochs=training_cfg["num_epochs"],
        learning_rate=training_cfg["learning_rate"],
        checkpoint_dir=training_cfg["checkpoint_dir"],
        test_after_training=training_cfg.get("test_after_training", True),
    )

    print("\nTraining complete...")
    print(
        f"Best validation position error: {min(history['val_pos_error']) * 1000:.2f}mm"
    )
