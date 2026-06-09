from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import List, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def parse_examples(text: str) -> List[Tuple[float, float]]:
    items = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        x0_str, v0_str = chunk.split(",")
        items.append((float(x0_str), float(v0_str)))
    if not items:
        raise ValueError("No valid examples were provided.")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Render trajectory comparisons from multiple model checkpoints.")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="One or more checkpoint paths.")
    parser.add_argument("--labels", nargs="*", default=None, help="Optional labels for checkpoints.")
    parser.add_argument("--save-path", required=True, help="Output image path.")
    parser.add_argument("--repo-root", default=".", help="Repository root that contains dataset/ and models/.")
    parser.add_argument("--examples", default="8.0,0.0;3.0,2.0;5.0,-3.0", help='Semicolon-separated "x0,v0" pairs.')
    parser.add_argument("--T-final", type=float, default=None, help="Override plotting horizon.")
    parser.add_argument("--title", default=None, help="Optional figure title.")
    parser.add_argument("--dpi", type=int, default=180, help="Saved figure DPI.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo_root))

    from dataset.bouncing_ball import BouncingBall
    from models.event_neural_ode import EventNeuralODERegressor
    from models.neural_ode_baseline import DirectMLPPredictor, NeuralODERegressor, Standardizer
    from models.rnn import RNNBaseline

    def activation_module(name: str) -> nn.Module:
        if name == "relu":
            return nn.ReLU()
        if name == "tanh":
            return nn.Tanh()
        if name == "gelu":
            return nn.GELU()
        raise ValueError(f"Unsupported activation: {name}")

    def build_model(ckpt_args: dict, stats: dict, device: torch.device) -> nn.Module:
        model_name = ckpt_args["model"]
        hidden_dim = ckpt_args.get("hidden_dim", 128)
        num_layers = ckpt_args.get("num_layers", 2)
        step_size = ckpt_args.get("step_size", 0.05)
        restitution = ckpt_args.get("restitution", 0.9)
        activation = activation_module(ckpt_args.get("activation", "relu"))
        hidden_dims = [hidden_dim] * num_layers

        if model_name == "mlp":
            model = DirectMLPPredictor(
                hidden_dims=hidden_dims,
                activation=activation,
                dropout=0.0,
            )
        elif model_name == "node":
            model = NeuralODERegressor(
                stats=stats,
                state_dim=2,
                hidden_dims=hidden_dims,
                activation=activation,
                step_size=step_size,
            )
        elif model_name == "event_node":
            model = EventNeuralODERegressor(
                stats=stats,
                state_dim=2,
                hidden_dims=hidden_dims,
                activation=activation,
                restitution=restitution,
                step_size=step_size,
            )
        elif model_name == "rnn":
            model = RNNBaseline(hidden_size=hidden_dim)
        else:
            raise ValueError(f"Unsupported model type: {model_name}")

        return model.to(device)

    def load_stats(ckpt: dict, device: torch.device):
        stats = {
            "dt": Standardizer(ckpt["dt_mean"].to(device), ckpt["dt_std"].to(device)),
            "z0": Standardizer(ckpt["z0_mean"].to(device), ckpt["z0_std"].to(device)),
            "y": Standardizer(ckpt["y_mean"].to(device), ckpt["y_std"].to(device)),
        }
        if "state_mean" in ckpt and "state_std" in ckpt:
            stats["state"] = Standardizer(ckpt["state_mean"].to(device), ckpt["state_std"].to(device))
        return stats

    @torch.no_grad()
    def predict_physical(model: nn.Module, dt: torch.Tensor, z0: torch.Tensor, stats: dict) -> torch.Tensor:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        dt = dt.to(device=device, dtype=dtype)
        z0 = z0.to(device=device, dtype=dtype)

        dt_model = stats["dt"].encode(dt)
        z0_model = stats["z0"].encode(z0)
        pred_norm = model(dt_model, z0_model)
        pred = stats["y"].decode(pred_norm)

        pred[:, 0] = torch.clamp(pred[:, 0], min=0.0)
        return pred

    @torch.no_grad()
    def predict_trajectory(model: nn.Module, stats: dict, x0: float, v0: float, T_final: float, num_points: int = 300):
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        dt_grid = torch.linspace(0.0, T_final, num_points, device=device, dtype=dtype)
        z0 = torch.tensor([[x0, v0]], device=device, dtype=dtype)

        if isinstance(model, (NeuralODERegressor, EventNeuralODERegressor)):
            z0_model = stats["z0"].encode(z0)
            dt_grid_model = stats["dt"].encode(dt_grid.unsqueeze(1)).squeeze(1)
            pred_norm_traj = model.trajectory(dt_grid_model, z0_model)
            pred_traj = stats["y"].decode(pred_norm_traj)
            if pred_traj.shape[0] == num_points + 1:
                pred_traj = pred_traj[1:]
        else:
            dt_col = dt_grid.view(-1, 1)
            z0_rep = z0.expand(dt_col.shape[0], -1)
            pred_traj = predict_physical(model, dt_col, z0_rep, stats)

        pred_traj[:, 0] = torch.clamp(pred_traj[:, 0], min=0.0)
        return dt_grid.detach().cpu().numpy(), pred_traj[:, 0].detach().cpu().numpy(), pred_traj[:, 1].detach().cpu().numpy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_paths = [Path(p).resolve() for p in args.checkpoints]
    if args.labels is not None and len(args.labels) not in (0, len(ckpt_paths)):
        raise ValueError("The number of --labels must match the number of --checkpoints.")
    labels = args.labels if args.labels else [p.stem.replace("_checkpoint", "") for p in ckpt_paths]

    def remap_legacy_keys(state_dict: dict) -> dict:
        # Older node / event_node checkpoints named the drift submodule
        # `func.acceleration_net`; the refactored model renamed it to
        # `func.acceleration`. The weights are identical, so just remap the key.
        return {
            k.replace("func.acceleration_net.", "func.acceleration."): v
            for k, v in state_dict.items()
        }

    models_and_stats = []
    ckpt_args_list = []
    for ckpt_path, label in zip(ckpt_paths, labels):
        ckpt = torch.load(ckpt_path, map_location=device)
        ckpt_args = ckpt["args"]
        stats = load_stats(ckpt, device)
        model = build_model(ckpt_args, stats, device)
        model.load_state_dict(remap_legacy_keys(ckpt["model_state_dict"]))
        model.eval()
        models_and_stats.append((label, model, stats))
        ckpt_args_list.append(ckpt_args)

    first_args = ckpt_args_list[0]
    gravity = float(first_args.get("gravity", 9.8))
    restitution = float(first_args.get("restitution", 0.9))
    default_T = float(first_args.get("T_final_test", first_args.get("T_final", 5.0)))
    T_final = float(args.T_final) if args.T_final is not None else default_T

    examples = parse_examples(args.examples)
    ball = BouncingBall(gravity=gravity, restitution=restitution)

    n = len(examples)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 6), sharex="col")
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, (x0, v0) in enumerate(examples):
        t_true, x_true, v_true = ball.trajectory(
            t0=0.0,
            x0=x0,
            v0=v0,
            T_final=T_final,
            num_points=300,
        )

        axes[0, col].plot(t_true, x_true, label="truth", linewidth=2.5, color="black")
        axes[1, col].plot(t_true, v_true, label="truth", linewidth=2.5, color="black")

        for label, model, stats in models_and_stats:
            t_plot, pred_x, pred_v = predict_trajectory(model, stats, x0, v0, T_final, num_points=300)
            axes[0, col].plot(t_plot, pred_x, label=label, linewidth=1.8, linestyle="--")
            axes[1, col].plot(t_plot, pred_v, label=label, linewidth=1.8, linestyle="--")

        axes[0, col].set_title(f"x0={x0}, v0={v0}")
        axes[0, col].set_ylabel("position")
        axes[0, col].grid(True, alpha=0.3)
        axes[0, col].legend()

        axes[1, col].set_xlabel("time")
        axes[1, col].set_ylabel("velocity")
        axes[1, col].grid(True, alpha=0.3)
        axes[1, col].legend()

    fig.suptitle(args.title or f"Trajectory comparison (T_final={T_final:g})")
    fig.tight_layout()
    save_path = Path(args.save_path).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=args.dpi)
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
