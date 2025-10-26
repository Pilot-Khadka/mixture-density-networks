import torch
import torch.nn as nn
import torch.nn.functional as F


class ForwardKinematicsMLP(nn.Module):
    def __init__(
        self,
        input_dim=8,
        hidden_dims=[256, 512, 512, 256],
        output_dim=3,
        dropout=0.1,
    ):
        super(ForwardKinematicsMLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def compute_loss(self, model_output, target):
        return F.mse_loss(model_output, target)

    def get_prediction(self, model_output):
        return model_output

    def predict(self, x):
        return self.forward(x)


if __name__ == "__main__":
    input_dim = 8
    output_dim = 3
    batch_size = 32

    print("Testing MLP...")
    mlp = ForwardKinematicsMLP(input_dim=input_dim, output_dim=output_dim)

    joint_angles = torch.randn(batch_size, input_dim)
    target_positions = torch.randn(batch_size, output_dim)

    mlp_output = mlp(joint_angles)
    mlp_loss = mlp.compute_loss(mlp_output, target_positions)
    mlp_pred = mlp.get_prediction(mlp_output)

    print(f"MLP output shape: {mlp_output.shape}")
    print(f"MLP loss (MSE): {mlp_loss.item():.4f}")
    print(f"MLP prediction shape: {mlp_pred.shape}")
