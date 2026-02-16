"""
Minimal Rectified Flow: 2D Gaussian -> 2D Spiral
Copy-paste into Google Colab and run directly.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ── 1. Generate spiral data ─────────────────────────────────────────────────

def make_spiral(n, noise=0.3):
    """Generate 2D spiral points."""
    t = torch.linspace(0, 3 * np.pi, n)
    r = t / (3 * np.pi) * 4
    x = r * torch.cos(t) + torch.randn(n) * noise * 0.1
    y = r * torch.sin(t) + torch.randn(n) * noise * 0.1
    return torch.stack([x, y], dim=1)  # (n, 2)

# ── 2. Velocity network v(x, t) ─────────────────────────────────────────────

class VelocityNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),  # input: (x, y, t)
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),  # output: velocity (vx, vy)
        )

    def forward(self, x, t):
        # x: (B, 2), t: (B, 1)
        return self.net(torch.cat([x, t], dim=1))

# ── 3. Training ─────────────────────────────────────────────────────────────

def train(n_iters=5000, batch_size=512, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VelocityNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for step in range(n_iters):
        # x0 ~ N(0, I)  (source)
        x0 = torch.randn(batch_size, 2, device=device)
        # x1 ~ spiral   (target)
        x1 = make_spiral(batch_size).to(device)
        # t ~ U(0, 1)
        t = torch.rand(batch_size, 1, device=device)

        # Linear interpolation: x_t = (1-t)*x0 + t*x1
        x_t = (1 - t) * x0 + t * x1

        # Target velocity: dx/dt = x1 - x0
        target_v = x1 - x0

        # Predict velocity
        pred_v = model(x_t, t)

        # MSE loss
        loss = ((pred_v - target_v) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (step + 1) % 1000 == 0:
            print(f"Step {step+1}/{n_iters}  loss={loss.item():.4f}")

    return model, losses

# ── 4. Inference (ODE integration) ──────────────────────────────────────────

@torch.no_grad()
def sample(model, n=1000, steps=100):
    device = next(model.parameters()).device
    # Start from Gaussian noise
    x = torch.randn(n, 2, device=device)
    dt = 1.0 / steps

    trajectory = [x.cpu().clone()]
    for i in range(steps):
        t = torch.full((n, 1), i * dt, device=device)
        v = model(x, t)
        x = x + v * dt
        trajectory.append(x.cpu().clone())

    return trajectory

# ── 5. Visualization ────────────────────────────────────────────────────────

def visualize(trajectory, losses):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # (a) Source: Gaussian noise (t=0)
    src = trajectory[0].numpy()
    axes[0].scatter(src[:, 0], src[:, 1], s=2, alpha=0.5, c="steelblue")
    axes[0].set_title("Source (Gaussian, t=0)")
    axes[0].set_xlim(-5, 5); axes[0].set_ylim(-5, 5)
    axes[0].set_aspect("equal")

    # (b) Midpoint (t=0.5)
    mid = trajectory[len(trajectory) // 2].numpy()
    axes[1].scatter(mid[:, 0], mid[:, 1], s=2, alpha=0.5, c="orange")
    axes[1].set_title("Midpoint (t=0.5)")
    axes[1].set_xlim(-5, 5); axes[1].set_ylim(-5, 5)
    axes[1].set_aspect("equal")

    # (c) Generated: final output (t=1)
    gen = trajectory[-1].numpy()
    axes[2].scatter(gen[:, 0], gen[:, 1], s=2, alpha=0.5, c="crimson")
    axes[2].set_title("Generated (t=1)")
    axes[2].set_xlim(-5, 5); axes[2].set_ylim(-5, 5)
    axes[2].set_aspect("equal")

    # (d) Training loss
    axes[3].plot(losses, color="black", linewidth=0.5)
    axes[3].set_title("Training Loss")
    axes[3].set_xlabel("Step")
    axes[3].set_ylabel("MSE")
    axes[3].set_yscale("log")

    plt.tight_layout()
    plt.savefig("rectified_flow_spiral.png", dpi=150)
    plt.show()
    print("Saved to rectified_flow_spiral.png")

    # Bonus: trajectory animation (selected particles)
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    n_traces = 50
    steps_idx = list(range(0, len(trajectory), max(1, len(trajectory) // 20)))
    colors = plt.cm.viridis(np.linspace(0, 1, len(steps_idx)))

    for j, si in enumerate(steps_idx):
        pts = trajectory[si].numpy()[:n_traces]
        ax2.scatter(pts[:, 0], pts[:, 1], s=5, color=colors[j], alpha=0.7)

    # Draw flow lines for a few particles
    for i in range(n_traces):
        xs = [trajectory[si][i, 0].item() for si in steps_idx]
        ys = [trajectory[si][i, 1].item() for si in steps_idx]
        ax2.plot(xs, ys, linewidth=0.3, color="gray", alpha=0.5)

    ax2.set_title("Flow Trajectories (t=0 → t=1)")
    ax2.set_xlim(-5, 5); ax2.set_ylim(-5, 5)
    ax2.set_aspect("equal")
    plt.tight_layout()
    plt.savefig("rectified_flow_trajectories.png", dpi=150)
    plt.show()

# ── 6. Run ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Training Rectified Flow: Gaussian → Spiral")
    model, losses = train(n_iters=5000, batch_size=512)

    print("Sampling...")
    trajectory = sample(model, n=2000, steps=100)

    print("Visualizing...")
    visualize(trajectory, losses)
