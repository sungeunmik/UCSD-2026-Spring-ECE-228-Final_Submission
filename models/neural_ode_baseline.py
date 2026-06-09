from __future__ import annotations

import math

import torch
import torch.nn as nn

#simple integral function - RK4
def rk4_step(drift, z: torch.Tensor, t_step: torch.Tensor) -> torch.Tensor:
    """based on the current z = [x, v], drift returns [dx/dt, dv/dt] which is the velocity and acceleration, and we take one RK4 step of size t_step."""
    k1 = drift(z)
    k2 = drift(z + 0.5 * t_step * k1)
    k3 = drift(z + 0.5 * t_step * k2)
    k4 = drift(z + t_step * k3)
    return z + (t_step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: nn.Module | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if activation is None:
            activation = nn.Tanh()

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation.__class__())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AccelerationMLPODEFunc(nn.Module):
    """dx/dt = v, dv/dt = MLP(x, v)."""

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
        if activation is None:
            activation = nn.ReLU()
        self.acceleration = MLP(
            input_dim=2,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation=activation,
            dropout=0.0,
        )
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 1:
            z = z.unsqueeze(0)
        v = z[..., 1:2]  # z = [x, v], v just recievs 
        acceleration = self.acceleration(z) # dv/dt is learned by MLP target is gravity 9.8
        return torch.cat([v, acceleration], dim=-1)


class DirectMLPPredictor(nn.Module):
    """
    Baseline 1: directly predict z(T) from [dt, x0, v0].
    """

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        activation: nn.Module | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]
        self.net = MLP(
            input_dim=3,
            hidden_dims=hidden_dims,
            output_dim=2,
            activation=activation or nn.Tanh(),
            dropout=dropout,
        )

    def forward(self, dt: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
        dt = _ensure_column_vector(dt, z0)
        features = torch.cat([dt, z0], dim=1)
        return self.net(features)


class NeuralODERegressor(nn.Module):
    """
    Input is dt, x0, v0; output is x(T), v(T) with no event handling.
    """

    def __init__(
        self,
        stats: dict,
        state_dim: int = 2,
        hidden_dims: list[int] | None = None,
        activation: nn.Module | None = None,
        step_size: float = 0.05,
    ) -> None:
        super().__init__()
        if state_dim != 2:
            raise ValueError("NeuralODERegressor expects state_dim=2 for [x, v].")
        self.func = AccelerationMLPODEFunc(hidden_dims=hidden_dims, activation=activation)
        self.step_size = step_size
        # Standardisers used to decode dt / z0 into physical units and encode the
        # target. Injected at construction so they are always present.
        self.dt_std = stats["dt"]
        self.z0_std = stats["z0"]
        self.y_std = stats["y"]


    def _integrate(self, dt_phys: torch.Tensor, z0_phys: torch.Tensor) -> torch.Tensor:
        """MLP acceleration integration of [0, dt] for a whole batch (no events).""" 
        n = max(16, math.ceil(float(dt_phys.max()) / max(self.step_size, 1e-6))) # at least 16 steps
        t_step = dt_phys / n
        drift = lambda zz: self.func(zz)
        z = z0_phys
        for _ in range(n):
            z = rk4_step(drift, z, t_step)
        return z

    def forward(self, dt_norm: torch.Tensor, z0_norm: torch.Tensor) -> torch.Tensor:
        dt_phys = self.dt_std.decode(dt_norm)
        z0_phys = self.z0_std.decode(z0_norm)
        z_final = self._integrate(dt_phys, z0_phys)
        return self.y_std.encode(z_final)

    @torch.no_grad()
    def trajectory(self, dt_grid_norm: torch.Tensor, z0_norm: torch.Tensor) -> torch.Tensor:
        """Normalised (T, 2) trajectory from one initial state (plot/eval only)."""
        if z0_norm.shape[0] != 1:
            raise ValueError("trajectory expects exactly one initial state")
        z0_phys = self.z0_std.decode(z0_norm)
        dt_grid_phys = self.dt_std.decode(dt_grid_norm.unsqueeze(1))
        z0_rep = z0_phys.expand(dt_grid_phys.shape[0], -1).contiguous()
        z = self._integrate(dt_grid_phys, z0_rep)
        return self.y_std.encode(z)


class Standardizer(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        std = torch.where(std < 1e-8, torch.ones_like(std), std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    @classmethod
    def from_tensor(cls, x: torch.Tensor) -> "Standardizer":
        return cls(x.mean(dim=0), x.std(dim=0, unbiased=False))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


def _ensure_column_vector(dt: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(dt):
        dt = torch.tensor(dt, dtype=z0.dtype, device=z0.device)
    dt = dt.to(device=z0.device, dtype=z0.dtype)
    if dt.ndim == 0:
        dt = dt.view(1, 1).expand(z0.shape[0], 1)
    elif dt.ndim == 1:
        if dt.shape[0] != z0.shape[0]:
            raise ValueError("1D dt tensor must have length batch_size")
        dt = dt.unsqueeze(1)
    elif dt.ndim == 2 and dt.shape[1] == 1:
        if dt.shape[0] != z0.shape[0]:
            raise ValueError("2D dt tensor must have shape (batch_size, 1)")
    else:
        raise ValueError("dt must be scalar, (batch,), or (batch, 1)")
    return dt
