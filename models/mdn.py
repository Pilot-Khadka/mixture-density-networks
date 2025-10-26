import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ForwardKinematicsMDN(nn.Module):
    def __init__(
        self,
        input_dim=8,
        hidden_dims=[256, 512, 512, 256],
        output_dim=3,
        n_gaussians=5,
        dropout=0.1,
    ):
        super(ForwardKinematicsMDN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_gaussians = n_gaussians

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.pi_head = nn.Linear(prev_dim, n_gaussians)
        self.mu_head = nn.Linear(prev_dim, n_gaussians * output_dim)
        self.sigma_head = nn.Linear(prev_dim, n_gaussians * output_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        features = self.backbone(x)

        pi = F.softmax(self.pi_head(features), dim=1)

        mu = self.mu_head(features)
        mu = mu.view(batch_size, self.n_gaussians, self.output_dim)

        sigma = torch.exp(self.sigma_head(features))
        sigma = sigma.view(batch_size, self.n_gaussians, self.output_dim)

        return pi, mu, sigma

    def compute_loss(self, model_output, target):
        pi, mu, sigma = model_output
        batch_size, n_gaussians, output_dim = mu.shape

        target = target.unsqueeze(1).expand_as(mu)
        exponent = -0.5 * torch.sum(((target - mu) / sigma) ** 2, dim=2)
        normalizer = -output_dim * math.log(math.sqrt(2 * math.pi)) - torch.sum(
            torch.log(sigma), dim=2
        )

        log_prob = exponent + normalizer
        log_pi = torch.log(pi + 1e-8)
        log_sum = torch.logsumexp(log_pi + log_prob, dim=1)

        loss = -torch.mean(log_sum)
        return loss

    def get_prediction(self, model_output):
        pi, mu, sigma = model_output
        max_pi_idx = torch.argmax(pi, dim=1)
        batch_size = pi.shape[0]
        return mu[torch.arange(batch_size), max_pi_idx]

    def predict(self, x):
        model_output = self.forward(x)
        return self.get_prediction(model_output)

    def sample(self, x, n_samples=1):
        pi, mu, sigma = self.forward(x)
        batch_size = x.shape[0]
        samples = []

        for _ in range(n_samples):
            component_idx = torch.multinomial(pi, num_samples=1).squeeze()
            selected_mu = mu[torch.arange(batch_size), component_idx]
            selected_sigma = sigma[torch.arange(batch_size), component_idx]

            eps = torch.randn_like(selected_mu)
            sample = selected_mu + selected_sigma * eps
            samples.append(sample)

        return torch.stack(samples, dim=1)


if __name__ == "__main__":
    input_dim = 8
    output_dim = 3
    batch_size = 32

    mdn = ForwardKinematicsMDN(
        input_dim=input_dim, output_dim=output_dim, n_gaussians=5
    )

    joint_angles = torch.randn(batch_size, input_dim)
    target_positions = torch.randn(batch_size, output_dim)

    model_output = mdn(joint_angles)
    loss = mdn.compute_loss(model_output, target_positions)

    pi, mu, sigma = model_output
    print(f"\nMDN pi shape: {pi.shape}")
    print(f"MDN mu shape: {mu.shape}")
    print(f"MDN sigma shape: {sigma.shape}")
    print(f"MDN loss (NLL): {loss.item():.4f}")

    prediction = mdn.predict(joint_angles)
    print(f"\nMDN prediction shape: {prediction.shape}")

    samples = mdn.sample(joint_angles, n_samples=10)
    print(f"MDN samples shape: {samples.shape}")
