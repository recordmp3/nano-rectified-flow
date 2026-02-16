"""
Minimal DDPM / DDIM: 2D Gaussian -> 2D Spiral
Same network as rectified_flow_2d_spiral.py.
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

# ── 2. Noise-prediction network eps(x_t, t) ─────────────────────────────────
#    Same architecture as VelocityNet — predicts noise instead of velocity.

class EpsNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),   # input: (x, y, t)
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),   # output: predicted noise (eps_x, eps_y)
        )

    def forward(self, x, t):
        # x: (B, 2), t: (B, 1)
        return self.net(torch.cat([x, t], dim=1))

# ── 3. Noise schedule ───────────────────────────────────────────────────────

def linear_beta_schedule(T, beta_min=1e-4, beta_max=0.02):
    """Linear schedule for betas."""
    return torch.linspace(beta_min, beta_max, T)


def make_schedule(T, beta_min=1e-4, beta_max=0.02):
    """Pre-compute all DDPM quantities."""
    betas = linear_beta_schedule(T, beta_min, beta_max)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)           # \bar{alpha}_t
    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bar": alpha_bar,
        "sqrt_ab": sqrt_alpha_bar,
        "sqrt_1m_ab": sqrt_one_minus_alpha_bar,
    }

# ── 4. DDPM Training ────────────────────────────────────────────────────────

def train(T=1000, n_iters=5000, batch_size=512, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EpsNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    sched = {k: v.to(device) for k, v in make_schedule(T).items()}

    losses = []
    for step in range(n_iters):
        # x_0 ~ spiral (data)
        x0 = make_spiral(batch_size).to(device)

        # Sample random timestep for each sample
        ts = torch.randint(0, T, (batch_size,), device=device)  # (B,)

        # Sample noise
        eps = torch.randn_like(x0)

        # Forward diffusion: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
        sqrt_ab = sched["sqrt_ab"][ts].unsqueeze(1)       # (B, 1)
        sqrt_1m_ab = sched["sqrt_1m_ab"][ts].unsqueeze(1)  # (B, 1)
        x_t = sqrt_ab * x0 + sqrt_1m_ab * eps

        # Normalise t to [0, 1] so the network sees the same input format
        t_norm = ts.float().unsqueeze(1) / T  # (B, 1)

        # Predict noise
        pred_eps = model(x_t, t_norm)

        # MSE loss
        loss = ((pred_eps - eps) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (step + 1) % 1000 == 0:
            print(f"Step {step+1}/{n_iters}  loss={loss.item():.4f}")

    return model, losses, sched

# ── 5a. DDPM Sampling (stochastic, T steps) ─────────────────────────────────

@torch.no_grad()
def sample_ddpm(model, sched, T=1000, n=1000):
    device = next(model.parameters()).device
    x = torch.randn(n, 2, device=device)

    trajectory = [x.cpu().clone()]
    for i in reversed(range(T)):
        t_norm = torch.full((n, 1), i / T, device=device)
        pred_eps = model(x, t_norm)

        alpha = sched["alphas"][i]
        alpha_bar = sched["alpha_bar"][i]
        beta = sched["betas"][i]

        # DDPM reverse step: x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps) + sigma_t * z
        coef = beta / torch.sqrt(1.0 - alpha_bar)
        x = (x - coef * pred_eps) / torch.sqrt(alpha)

        if i > 0:
            sigma = torch.sqrt(beta)
            x = x + sigma * torch.randn_like(x)

        # Save a few snapshots (every 100 steps + the last one)
        if i % 100 == 0 or i == 0:
            trajectory.append(x.cpu().clone())

    return trajectory

# ── 5b. DDIM Sampling (deterministic, fewer steps) ──────────────────────────

@torch.no_grad()
def sample_ddim(model, sched, T=1000, n=1000, steps=50, eta=0.0):
    """
    DDIM sampler.  eta=0 → fully deterministic, eta=1 → equivalent to DDPM.
    `steps` controls how many denoising steps to actually run (sub-sequence of [0..T-1]).
    """
    device = next(model.parameters()).device
    alpha_bar = sched["alpha_bar"]

    # Build sub-sequence of timesteps (evenly spaced)
    seq = torch.linspace(T - 1, 0, steps + 1).long()  # e.g. [999, 979, …, 0]

    x = torch.randn(n, 2, device=device)
    trajectory = [x.cpu().clone()]

    for i in range(steps):
        t_cur = seq[i].item()
        t_prev = seq[i + 1].item()

        t_norm = torch.full((n, 1), t_cur / T, device=device)
        pred_eps = model(x, t_norm)

        ab_cur = alpha_bar[t_cur]
        ab_prev = alpha_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0)

        # Predict x_0
        x0_pred = (x - torch.sqrt(1 - ab_cur) * pred_eps) / torch.sqrt(ab_cur)

        # Direction pointing to x_t
        sigma = eta * torch.sqrt((1 - ab_prev) / (1 - ab_cur) * (1 - ab_cur / ab_prev))
        dir_xt = torch.sqrt(1 - ab_prev - sigma ** 2) * pred_eps

        x = torch.sqrt(ab_prev) * x0_pred + dir_xt
        if sigma > 0:
            x = x + sigma * torch.randn_like(x)

        trajectory.append(x.cpu().clone())

    return trajectory

# ── 6. Visualization ────────────────────────────────────────────────────────

def visualize(traj_ddpm, traj_ddim, losses):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # (a) Source noise
    src = traj_ddpm[0].numpy()
    axes[0].scatter(src[:, 0], src[:, 1], s=2, alpha=0.5, c="steelblue")
    axes[0].set_title("Source (Gaussian)")
    axes[0].set_xlim(-5, 5); axes[0].set_ylim(-5, 5)
    axes[0].set_aspect("equal")

    # (b) DDPM result
    gen_ddpm = traj_ddpm[-1].numpy()
    axes[1].scatter(gen_ddpm[:, 0], gen_ddpm[:, 1], s=2, alpha=0.5, c="crimson")
    axes[1].set_title("DDPM (T=1000 steps)")
    axes[1].set_xlim(-5, 5); axes[1].set_ylim(-5, 5)
    axes[1].set_aspect("equal")

    # (c) DDIM result
    gen_ddim = traj_ddim[-1].numpy()
    axes[2].scatter(gen_ddim[:, 0], gen_ddim[:, 1], s=2, alpha=0.5, c="darkorange")
    axes[2].set_title("DDIM (50 steps, eta=0)")
    axes[2].set_xlim(-5, 5); axes[2].set_ylim(-5, 5)
    axes[2].set_aspect("equal")

    # (d) Training loss
    axes[3].plot(losses, color="black", linewidth=0.5)
    axes[3].set_title("Training Loss")
    axes[3].set_xlabel("Step"); axes[3].set_ylabel("MSE")
    axes[3].set_yscale("log")

    plt.tight_layout()
    plt.savefig("ddpm_ddim_spiral.png", dpi=150)
    plt.show()
    print("Saved to ddpm_ddim_spiral.png")

# ── 7. Run ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    T = 1000

    print("Training DDPM: learning to denoise Spiral")
    model, losses, sched = train(T=T, n_iters=5000, batch_size=512)

    print("Sampling with DDPM (1000 steps)...")
    traj_ddpm = sample_ddpm(model, sched, T=T, n=2000)

    print("Sampling with DDIM (50 steps, deterministic)...")
    traj_ddim = sample_ddim(model, sched, T=T, n=2000, steps=50, eta=0.0)

    print("Visualizing...")
    visualize(traj_ddpm, traj_ddim, losses)
