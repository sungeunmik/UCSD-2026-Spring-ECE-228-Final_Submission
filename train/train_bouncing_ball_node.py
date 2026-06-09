from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

from dataset.bouncing_ball import BouncingBall, sample_bouncing_ball_dataset
from models.event_neural_ode import EventNeuralODERegressor
from models.neural_ode_baseline import DirectMLPPredictor, NeuralODERegressor, Standardizer
from models.rnn import RNNBaseline


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # dataset sampling uses np.random; seed it for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BouncingBallDataset:
    def __init__(
        self,
        num_samples: int,
        T_final: float,
        gravity: float,
        restitution: float,
        x0_range: tuple[float, float],
        v0_range: tuple[float, float],
        t0_range: tuple[float, float] | None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        x_raw, y_raw = sample_bouncing_ball_dataset(
            num_samples=num_samples,
            BouncingBall=BouncingBall,
            T_final=T_final,
            gravity=gravity,
            restitution=restitution,
            x0_range=x0_range,
            v0_range=v0_range,
            t0_range=t0_range,
            dtype=dtype,
        )
        t0 = x_raw[:, 0:1]
        tT = x_raw[:, 1:2] # tT is additionally included
        z0 = x_raw[:, 2:4]
        dt = T_final - t0
        self.dt = dt
        self.z0 = z0
        self.target = y_raw
        self.raw_input = x_raw
        self.T_final = T_final
        self.gravity = gravity
        self.restitution = restitution

    def tensor_dataset(self) -> TensorDataset:
        return TensorDataset(self.dt, self.z0, self.target)


def build_loaders(
    num_samples: int,
    batch_size: int,
    T_final: float,
    T_final_test: float,
    gravity: float,
    restitution: float,
    x0_range: tuple[float, float],
    v0_range: tuple[float, float],
    t0_range: tuple[float, float] | None,
    train_ratio: float,
    val_ratio: float,
    seed: int,
):
    num_trainval = int((train_ratio + val_ratio) * num_samples)
    num_test = num_samples - num_trainval

    trainval_full = BouncingBallDataset(
        num_samples=num_trainval,
        T_final=T_final,
        gravity=gravity,
        restitution=restitution,
        x0_range=x0_range,
        v0_range=v0_range,
        t0_range=t0_range,
    )
    test_full = BouncingBallDataset(
        num_samples=num_test,
        T_final=T_final_test,
        gravity=gravity,
        restitution=restitution,
        x0_range=x0_range,
        v0_range=v0_range,
        t0_range=t0_range,
    )

    trainval_dataset = trainval_full.tensor_dataset()
    test_dataset = test_full.tensor_dataset()

    train_len = int(train_ratio / (train_ratio + val_ratio) * num_trainval)
    val_len = num_trainval - train_len
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(trainval_dataset, [train_len, val_len], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    dt_train = train_set.dataset.tensors[0][train_set.indices]
    z0_train = train_set.dataset.tensors[1][train_set.indices]
    y_train = train_set.dataset.tensors[2][train_set.indices]

    stats = {
        "dt": Standardizer.from_tensor(dt_train),
        "z0": Standardizer.from_tensor(z0_train),
        "y": Standardizer.from_tensor(y_train),
    }
    return trainval_full, train_loader, val_loader, test_loader, stats


def make_model(
    model_name: str, hidden_dim: int, num_layers: int, restitution: float,
    step_size: float = 0.05, activation: str = "relu", stats: dict | None = None,
) -> nn.Module:
    hidden_dims = [hidden_dim] * num_layers
    activation_module = _activation_module(activation)
    if model_name == "mlp":
        return DirectMLPPredictor(hidden_dims=hidden_dims, activation=activation_module, dropout=0.0)
    if model_name == "rnn":
        return RNNBaseline()
    if model_name == "node":
        # Paper-style Neural ODE baseline: dx/dt = v, dv/dt = MLP(x, v), no events.
        return NeuralODERegressor(
            stats=stats,
            state_dim=2,
            hidden_dims=hidden_dims,
            activation=_activation_module(activation),
            step_size=step_size,
        )
    if model_name == "event_node":
        # Same MLP acceleration drift as `node`, plus x=0 contact and learnable
        # restitution reset. The ONLY difference from `node` is event handling.
        return EventNeuralODERegressor(
            stats=stats,
            state_dim=2,
            hidden_dims=hidden_dims,
            activation=_activation_module(activation),
            restitution=restitution,
            step_size=step_size,
        )
    raise ValueError(f"Unknown model_name: {model_name}")


def _activation_module(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


def run_epoch(model, loader, optimizer, device, stats, training: bool):
    """Run one epoch."""
    mse = nn.MSELoss()
    if training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    for dt, z0, target in loader:
        dt = dt.to(device)
        z0 = z0.to(device)
        target = target.to(device)

        # All models share one I/O contract: normalised dt / z0 in, normalised
        # target-frame state out. node and event_node decode internally and
        # integrate in physical units (node without events, event_node with the
        # x=0 bounce); mlp / rnn map the normalised inputs directly.
        z0_norm = stats["z0"].encode(z0)
        target_norm = stats["y"].encode(target)
        dt_model = stats["dt"].encode(dt)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            pred_norm = model(dt_model, z0_norm)
            loss = mse(pred_norm, target_norm)
            if training:
                loss.backward()

        if training:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += loss.item() * dt.shape[0]

    return running_loss / len(loader.dataset)

def evaluate_metrics(model, loader, device, stats):
    model.eval()
    mse = nn.MSELoss(reduction="sum")
    total_mse = 0.0
    total_mae = 0.0
    total_count = 0

    with torch.no_grad():
        for dt, z0, target in loader:
            dt = dt.to(device)
            z0 = z0.to(device)
            target = target.to(device)
            pred = predict_physical(model, dt, z0, stats)
            total_mse += mse(pred, target).item()
            total_mae += torch.abs(pred - target).sum().item()
            total_count += target.numel()

    return {
        "mse": total_mse / total_count,
        "rmse": (total_mse / total_count) ** 0.5,
        "mae": total_mae / total_count,
    }


@torch.no_grad()
def predict_physical(model, dt, z0, stats):
    # Inference only: no_grad prevents the per-sample ODE solve from building an
    # autograd graph. Without it, callers that loop (e.g. plotting a trajectory
    # point by point) would retain one event-ODE graph per call and blow up RAM.
    device = stats["dt"].mean.device
    dtype = stats["dt"].mean.dtype

    dt = dt.to(device=device, dtype=dtype)
    z0 = z0.to(device=device, dtype=dtype)

    z0_norm = stats["z0"].encode(z0)
    pred_norm = model(stats["dt"].encode(dt), z0_norm)
    pred = stats["y"].decode(pred_norm)
    pred[:, 0] = torch.clamp(pred[:, 0], min=0.0)
    return pred


def _build_optimizer(model, lr):
    """Adam over all learnable parameters; NODE drift is now an acceleration MLP."""
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)


def _phys_str(model) -> str:
    parts = []
    for attr in ("gravity", "restitution"):
        if hasattr(model, attr):
            try:
                parts.append(f"{attr[0]}={float(getattr(model, attr).detach()):.3f}")
            except Exception:
                pass
    return ("  " + " ".join(parts)) if parts else ""


def train_model(model, train_loader, val_loader, device, stats, lr, epochs):
    optimizer = _build_optimizer(model, lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 3, 1), gamma=0.5)

    history = {"train_loss": [], "val_loss": []}
    best_state = None
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, stats, training=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, stats, training=False)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"epoch={epoch:04d} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}{_phys_str(model)}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def plot_losses(history: dict, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("normalized MSE")
    ax.set_title("Training Curves")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def plot_example_trajectory(
    model, ball, stats, save_path: Path,
    examples: list[tuple[float, float]], T_final: float,
):
    device = stats["dt"].mean.device
    dtype = stats["dt"].mean.dtype

    n = len(examples)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 6), sharex="col")
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, (x0, v0) in enumerate(examples):
        dt_grid = torch.linspace(0.0, T_final, 250, device=device, dtype=dtype)
        z0 = torch.tensor([[x0, v0]], dtype=dtype, device=device)

        # Predict the whole time grid in one batched call (same z0 repeated):
        # every model maps (dt, z0) -> z(dt), so the trajectory is just the
        # prediction at each horizon. Batching avoids a slow per-point loop.
        dt_col = dt_grid.view(-1, 1)
        z0_rep = z0.expand(dt_col.shape[0], -1)
        pred_traj = predict_physical(model, dt_col, z0_rep, stats)

        pred_x = pred_traj[:, 0].detach().cpu().numpy()
        pred_v = pred_traj[:, 1].detach().cpu().numpy()
        t_plot = dt_grid.detach().cpu().numpy()

        t_true, x_true, v_true = ball.trajectory(t0=0.0, x0=x0, v0=v0, T_final=T_final, num_points=250)

        axes[0, col].plot(t_true, x_true, label="truth", linewidth=2)
        axes[0, col].plot(t_plot, pred_x, label="prediction", linewidth=2, linestyle="--")
        axes[0, col].set_title(f"x0={x0}, v0={v0}")
        axes[0, col].set_ylabel("position")
        axes[0, col].legend()
        axes[0, col].grid(True, alpha=0.3)

        axes[1, col].plot(t_true, v_true, label="truth", linewidth=2)
        axes[1, col].plot(t_plot, pred_v, label="prediction", linewidth=2, linestyle="--")
        axes[1, col].set_xlabel("time")
        axes[1, col].set_ylabel("velocity")
        axes[1, col].legend()
        axes[1, col].grid(True, alpha=0.3)

    fig.suptitle(f"Example Trajectories (T_final={T_final})")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    
def save_checkpoint(model, stats, history, metrics, args, save_path: Path):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "history": history,
        "metrics": metrics,
        "args": vars(args),
        "dt_mean": stats["dt"].mean,
        "dt_std": stats["dt"].std,
        "z0_mean": stats["z0"].mean,
        "z0_std": stats["z0"].std,
        "y_mean": stats["y"].mean,
        "y_std": stats["y"].std,
    }
    torch.save(checkpoint, save_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a plain Neural ODE baseline on bouncing-ball data.")
    parser.add_argument("--model", choices=["mlp", "node", "rnn", "event_node"], default="node")
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--activation", choices=["relu", "tanh", "gelu"], default="relu")
    parser.add_argument("--step-size", type=float, default=0.05,
                        help="Fixed RK4 integration step (physical time) for node / "
                             "event_node. Smaller = more accurate at long horizons "
                             "(more bounces) but slower. Step count per batch is set "
                             "from the longest horizon so every sample is resolved "
                             "at <= this step.")
    parser.add_argument("--seed", type=int, default=228)
    parser.add_argument("--gravity", type=float, default=9.8,
                        help="True gravity used to generate the data.")
    parser.add_argument("--restitution", type=float, default=0.9)
    parser.add_argument("--T-final", type=float, default=5.0,
                        help="T_final for train/val data")
    parser.add_argument("--T-final-test", type=float, default=10.0,
                        help="T_final for test data (generalization horizon)")
    parser.add_argument("--x0-min", type=float, default=0.5)
    parser.add_argument("--x0-max", type=float, default=10.0)
    parser.add_argument("--v0-min", type=float, default=-5.0)
    parser.add_argument("--v0-max", type=float, default=5.0)
    parser.add_argument("--t0-min", type=float, default=0.0)
    parser.add_argument("--t0-max", type=float, default=4.0)
    parser.add_argument("--outdir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.outdir is None:
        args.outdir = f"runs/bouncing_ball_{args.model}"
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    full, train_loader, val_loader, test_loader, stats = build_loaders(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        T_final=args.T_final,
        T_final_test=args.T_final_test,
        gravity=args.gravity,
        restitution=args.restitution,
        x0_range=(args.x0_min, args.x0_max),
        v0_range=(args.v0_min, args.v0_max),
        t0_range=(args.t0_min, args.t0_max),
        train_ratio=0.7,
        val_ratio=0.15,
        seed=args.seed,
    )
    for key in stats:
        stats[key] = stats[key].to(device)

    # node and event_node integrate in physical units, so they take the
    # standardisers (stats) at construction to decode dt / z0 and encode the target.
    model = make_model(
        model_name=args.model,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        restitution=args.restitution,
        step_size=args.step_size,
        activation=args.activation,
        stats=stats,
    ).to(device)

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        stats=stats,
        lr=args.lr,
        epochs=args.epochs,
    )

    metrics = {
        "train": evaluate_metrics(model, train_loader, device, stats),
        "val": evaluate_metrics(model, val_loader, device, stats),
        "test": evaluate_metrics(model, test_loader, device, stats),
    }
    print(json.dumps(metrics, indent=2))

    plot_losses(history, outdir / f"{args.model}_loss_curve.png")
    ball = BouncingBall(gravity=args.gravity, restitution=args.restitution)
    plot_example_trajectory(
        model=model,
        ball=ball,
        stats=stats,
        save_path=outdir / f"{args.model}_example_trajectory.png",
        examples=[(8.0, 0.0), (3.0, 2.0), (5.0, -3.0)],
        T_final=args.T_final_test,
    )
    save_checkpoint(model, stats, history, metrics, args, outdir / f"{args.model}_checkpoint.pt")

    with open(outdir / f"{args.model}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved outputs to: {outdir}")


if __name__ == "__main__":
    main()
