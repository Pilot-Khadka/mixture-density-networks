import os
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
import seaborn as sns

from models.mlp import ForwardKinematicsMLP
from models.mdn import ForwardKinematicsMDN
from data.dataloader import RLBenchKinematicsDataset, collate_timesteps


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_model(model_class, model_config, checkpoint_path, device):
    model = model_class(**model_config).to(device)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from {checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Val Error: {checkpoint.get('val_pos_error', 0) * 1000:.2f}mm")
    else:
        print(f"Warning: No checkpoint found at {checkpoint_path}")
        print("Using randomly initialized model")

    return model


def get_mdn_predictions_with_uncertainty(model, dataloader, device, n_samples=100):
    """MDN predictions with confidence bands by sampling from the mixture"""
    model.eval()

    all_means = []
    all_stds = []
    all_samples = []
    all_targets = []

    with torch.no_grad():
        for joint_angles, cartesian_coords in tqdm(dataloader, desc="MDN Uncertainty"):
            joint_angles = joint_angles.to(device)
            cartesian_coords = cartesian_coords.to(device)
            target = cartesian_coords[:, :3]  # Only position

            pi, mu, sigma = model(joint_angles)

            batch_samples = []
            for _ in range(n_samples):
                component_idx = torch.multinomial(pi, 1).squeeze()

                batch_indices = torch.arange(len(joint_angles))
                selected_mu = mu[batch_indices, component_idx]
                selected_sigma = sigma[batch_indices, component_idx]

                sample = torch.normal(selected_mu, selected_sigma)
                batch_samples.append(sample)

            batch_samples = torch.stack(batch_samples, dim=1)  # [batch, n_samples, dim]

            sample_mean = batch_samples.mean(dim=1)
            sample_std = batch_samples.std(dim=1)

            all_means.append(sample_mean.cpu())
            all_stds.append(sample_std.cpu())
            all_samples.append(batch_samples.cpu())
            all_targets.append(target.cpu())

    return {
        "means": torch.cat(all_means, dim=0),
        "stds": torch.cat(all_stds, dim=0),
        "samples": torch.cat(all_samples, dim=0),
        "targets": torch.cat(all_targets, dim=0),
    }


def evaluate_model(model, dataloader, device, model_name="Model"):
    model.eval()

    all_predictions = []
    all_targets = []
    all_errors = []

    with torch.no_grad():
        for joint_angles, cartesian_coords in tqdm(
            dataloader, desc=f"Evaluating {model_name}"
        ):
            joint_angles = joint_angles.to(device)
            cartesian_coords = cartesian_coords.to(device)
            target = cartesian_coords[:, :3]  # Only position

            model_output = model(joint_angles)
            predictions = model.get_prediction(model_output)

            position_error = torch.sqrt(((predictions - target) ** 2).sum(dim=1))

            all_predictions.append(predictions.cpu())
            all_targets.append(target.cpu())
            all_errors.append(position_error.cpu())

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_errors = torch.cat(all_errors, dim=0)

    metrics = {
        "mean_error": all_errors.mean().item(),
        "std_error": all_errors.std().item(),
        "median_error": all_errors.median().item(),
        "max_error": all_errors.max().item(),
        "percentile_95": torch.quantile(all_errors, 0.95).item(),
        "percentile_99": torch.quantile(all_errors, 0.99).item(),
    }

    return {
        "predictions": all_predictions,
        "targets": all_targets,
        "errors": all_errors,
        "metrics": metrics,
    }


def compare_models_visualization(
    mlp_results, mdn_results, mdn_uncertainty, save_dir="results"
):
    """Create detailed comparison visualizations"""
    os.makedirs(save_dir, exist_ok=True)

    sns.set_style("whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    mlp_errors_mm = mlp_results["errors"].numpy() * 1000
    mdn_errors_mm = mdn_results["errors"].numpy() * 1000

    axes[0].hist(
        mlp_errors_mm, bins=50, alpha=0.6, label="MLP", color="blue", density=True
    )
    axes[0].hist(
        mdn_errors_mm, bins=50, alpha=0.6, label="MDN", color="red", density=True
    )
    axes[0].axvline(mlp_errors_mm.mean(), color="blue", linestyle="--", linewidth=2)
    axes[0].axvline(mdn_errors_mm.mean(), color="red", linestyle="--", linewidth=2)
    axes[0].set_xlabel("Position Error (mm)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Error Distribution Comparison")
    axes[0].legend()

    axes[1].boxplot([mlp_errors_mm, mdn_errors_mm], labels=["MLP", "MDN"])
    axes[1].set_ylabel("Position Error (mm)")
    axes[1].set_title("Error Distribution Box Plot")
    axes[1].grid(True, alpha=0.3)

    sorted_mlp = np.sort(mlp_errors_mm)
    sorted_mdn = np.sort(mdn_errors_mm)
    cumulative = np.arange(1, len(sorted_mlp) + 1) / len(sorted_mlp)

    axes[2].plot(sorted_mlp, cumulative, label="MLP", linewidth=2)
    axes[2].plot(sorted_mdn, cumulative, label="MDN", linewidth=2)
    axes[2].set_xlabel("Position Error (mm)")
    axes[2].set_ylabel("Cumulative Probability")
    axes[2].set_title("Cumulative Error Distribution")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "error_comparison.png"), dpi=150)
    plt.close()

    # 2. Prediction Accuracy per Axis
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for i, axis_name in enumerate(["X", "Y", "Z"]):
        # MLP predictions
        ax_mlp = axes[0, i]
        ax_mlp.scatter(
            mlp_results["targets"][:, i],
            mlp_results["predictions"][:, i],
            alpha=0.3,
            s=1,
            color="blue",
        )
        ax_mlp.plot(
            [mlp_results["targets"][:, i].min(), mlp_results["targets"][:, i].max()],
            [mlp_results["targets"][:, i].min(), mlp_results["targets"][:, i].max()],
            "r--",
            linewidth=2,
        )
        ax_mlp.set_xlabel(f"True {axis_name} (m)")
        ax_mlp.set_ylabel(f"Predicted {axis_name} (m)")
        ax_mlp.set_title(f"MLP - {axis_name} Axis")
        ax_mlp.grid(True, alpha=0.3)

        # MDN predictions
        ax_mdn = axes[1, i]
        ax_mdn.scatter(
            mdn_results["targets"][:, i],
            mdn_results["predictions"][:, i],
            alpha=0.3,
            s=1,
            color="red",
        )
        ax_mdn.plot(
            [mdn_results["targets"][:, i].min(), mdn_results["targets"][:, i].max()],
            [mdn_results["targets"][:, i].min(), mdn_results["targets"][:, i].max()],
            "r--",
            linewidth=2,
        )
        ax_mdn.set_xlabel(f"True {axis_name} (m)")
        ax_mdn.set_ylabel(f"Predicted {axis_name} (m)")
        ax_mdn.set_title(f"MDN - {axis_name} Axis")
        ax_mdn.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "axis_predictions.png"), dpi=150)
    plt.close()

    # 3. MDN Uncertainty Visualization with Confidence Bands
    if mdn_uncertainty is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        n_samples = min(1000, len(mdn_uncertainty["means"]))
        indices = np.random.choice(
            len(mdn_uncertainty["means"]), n_samples, replace=False
        )

        for i, axis_name in enumerate(["X", "Y", "Z"]):
            ax = axes[i]

            targets = mdn_uncertainty["targets"][indices, i]
            means = mdn_uncertainty["means"][indices, i]
            stds = mdn_uncertainty["stds"][indices, i]

            sort_idx = np.argsort(targets)
            targets = targets[sort_idx]
            means = means[sort_idx]
            stds = stds[sort_idx]

            ax.scatter(targets, means, alpha=0.5, s=1, color="red", label="MDN Mean")
            ax.fill_between(
                targets,
                means - 2 * stds,  # 95% confidence interval
                means + 2 * stds,
                alpha=0.2,
                color="red",
                label="95% Confidence",
            )
            ax.fill_between(
                targets,
                means - stds,  # 68% confidence interval
                means + stds,
                alpha=0.3,
                color="red",
                label="68% Confidence",
            )

            ax.plot(
                [targets.min(), targets.max()],
                [targets.min(), targets.max()],
                "k--",
                linewidth=1,
                alpha=0.5,
            )

            ax.set_xlabel(f"True {axis_name} (m)")
            ax.set_ylabel(f"Predicted {axis_name} (m)")
            ax.set_title(f"MDN Predictions with Uncertainty - {axis_name} Axis")
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "mdn_uncertainty_bands.png"), dpi=150)
        plt.close()

        # 4. Uncertainty vs Error Analysis
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        actual_errors = (
            torch.sqrt(
                ((mdn_uncertainty["means"] - mdn_uncertainty["targets"]) ** 2).sum(
                    dim=1
                )
            ).numpy()
            * 1000
        )

        avg_uncertainty = mdn_uncertainty["stds"].mean(dim=1).numpy() * 1000

        axes[0].scatter(avg_uncertainty, actual_errors, alpha=0.3, s=1)
        axes[0].set_xlabel("Predicted Uncertainty (mm)")
        axes[0].set_ylabel("Actual Error (mm)")
        axes[0].set_title("MDN: Uncertainty vs Actual Error")
        axes[0].grid(True, alpha=0.3)

        # diagonal line for reference
        max_val = max(avg_uncertainty.max(), actual_errors.max())
        axes[0].plot([0, max_val], [0, max_val], "r--", alpha=0.5)

        axes[1].hist(avg_uncertainty, bins=50, alpha=0.7, color="red")
        axes[1].set_xlabel("Predicted Uncertainty (mm)")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Distribution of MDN Uncertainty")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "uncertainty_analysis.png"), dpi=150)
        plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("tight")
    ax.axis("off")

    metrics_names = [
        "Mean Error (mm)",
        "Std Error (mm)",
        "Median Error (mm)",
        "Max Error (mm)",
        "95th Percentile (mm)",
        "99th Percentile (mm)",
    ]

    mlp_values = [
        f"{mlp_results['metrics']['mean_error'] * 1000:.2f}",
        f"{mlp_results['metrics']['std_error'] * 1000:.2f}",
        f"{mlp_results['metrics']['median_error'] * 1000:.2f}",
        f"{mlp_results['metrics']['max_error'] * 1000:.2f}",
        f"{mlp_results['metrics']['percentile_95'] * 1000:.2f}",
        f"{mlp_results['metrics']['percentile_99'] * 1000:.2f}",
    ]

    mdn_values = [
        f"{mdn_results['metrics']['mean_error'] * 1000:.2f}",
        f"{mdn_results['metrics']['std_error'] * 1000:.2f}",
        f"{mdn_results['metrics']['median_error'] * 1000:.2f}",
        f"{mdn_results['metrics']['max_error'] * 1000:.2f}",
        f"{mdn_results['metrics']['percentile_95'] * 1000:.2f}",
        f"{mdn_results['metrics']['percentile_99'] * 1000:.2f}",
    ]

    table_data = list(zip(metrics_names, mlp_values, mdn_values))

    table = ax.table(
        cellText=table_data,
        colLabels=["Metric", "MLP", "MDN"],
        cellLoc="center",
        loc="center",
        colWidths=[0.4, 0.3, 0.3],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)

    for i in range(3):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    for i in range(1, len(metrics_names) + 1):
        mlp_val = float(mlp_values[i - 1])
        mdn_val = float(mdn_values[i - 1])

        if mlp_val < mdn_val:
            table[(i, 1)].set_facecolor("#90EE90")  # Light green for better
            table[(i, 2)].set_facecolor("#FFB6C1")  # Light red for worse
        else:
            table[(i, 1)].set_facecolor("#FFB6C1")
            table[(i, 2)].set_facecolor("#90EE90")

    plt.title(
        "Performance Comparison: MLP vs MDN", fontsize=14, fontweight="bold", pad=20
    )
    plt.savefig(
        os.path.join(save_dir, "performance_table.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()

    print(f"\nVisualization saved to {save_dir}/")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_dataset_root = "rlbench_kinematics_dataset/validation/"

    if not os.path.exists(test_dataset_root):
        print(f"Error: Test dataset not found at {test_dataset_root}")
        return

    print("\nLoading test dataset...")
    test_dataset = RLBenchKinematicsDataset(root_dir=test_dataset_root)
    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_timesteps,
        pin_memory=True,
    )
    print(f"Test samples: {len(test_dataset)}")

    mlp_config = load_config("config/mlp.yaml")
    mdn_config = load_config("config/mdn.yaml")

    if mlp_config["training"].get("predict_orientation", False):
        mlp_config["model"]["output_dim"] = 6
    if mdn_config["training"].get("predict_orientation", False):
        mdn_config["model"]["output_dim"] = 6

    print("\nLoading MLP model...")
    mlp_model = load_model(
        ForwardKinematicsMLP,
        mlp_config["model"],
        "checkpoints/mlp/best_model.pth",
        device,
    )

    print("\nLoading MDN model...")
    mdn_model = load_model(
        ForwardKinematicsMDN,
        mdn_config["model"],
        "checkpoints/mdn/best_model.pth",
        device,
    )

    print("\n" + "=" * 50)
    print("Evaluating Models on Test Set")
    print("=" * 50)

    print("\nEvaluating MLP...")
    mlp_results = evaluate_model(mlp_model, test_loader, device, "MLP")

    print("\nMLP Performance:")
    print(f"  Mean Error:   {mlp_results['metrics']['mean_error'] * 1000:.2f} mm")
    print(f"  Std Error:    {mlp_results['metrics']['std_error'] * 1000:.2f} mm")
    print(f"  Median Error: {mlp_results['metrics']['median_error'] * 1000:.2f} mm")
    print(f"  Max Error:    {mlp_results['metrics']['max_error'] * 1000:.2f} mm")
    print(f"  95th %ile:    {mlp_results['metrics']['percentile_95'] * 1000:.2f} mm")
    print(f"  99th %ile:    {mlp_results['metrics']['percentile_99'] * 1000:.2f} mm")

    print("\nEvaluating MDN...")
    mdn_results = evaluate_model(mdn_model, test_loader, device, "MDN")

    print("\nMDN Performance:")
    print(f"  Mean Error:   {mdn_results['metrics']['mean_error'] * 1000:.2f} mm")
    print(f"  Std Error:    {mdn_results['metrics']['std_error'] * 1000:.2f} mm")
    print(f"  Median Error: {mdn_results['metrics']['median_error'] * 1000:.2f} mm")
    print(f"  Max Error:    {mdn_results['metrics']['max_error'] * 1000:.2f} mm")
    print(f"  95th %ile:    {mdn_results['metrics']['percentile_95'] * 1000:.2f} mm")
    print(f"  99th %ile:    {mdn_results['metrics']['percentile_99'] * 1000:.2f} mm")

    print("\nCalculating MDN uncertainty estimates...")
    mdn_uncertainty = get_mdn_predictions_with_uncertainty(
        mdn_model, test_loader, device, n_samples=100
    )

    print("\n" + "=" * 50)
    print("Model Comparison")
    print("=" * 50)

    mlp_mean = mlp_results["metrics"]["mean_error"] * 1000
    mdn_mean = mdn_results["metrics"]["mean_error"] * 1000
    improvement = ((mlp_mean - mdn_mean) / mlp_mean) * 100

    if mdn_mean < mlp_mean:
        print(f"MDN performs better by {improvement:.1f}%")
    else:
        print(f"MLP performs better by {-improvement:.1f}%")

    print("\nCreating comparison visualizations...")
    compare_models_visualization(mlp_results, mdn_results, mdn_uncertainty)

    print("\nTesting complete!")


if __name__ == "__main__":
    main()
