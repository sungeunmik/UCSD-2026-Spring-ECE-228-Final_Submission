from __future__ import annotations
import math

import torch
import torch.nn as nn

from models.neural_ode_baseline import AccelerationMLPODEFunc, rk4_step


class EventNeuralODERegressor(nn.Module):
    def __init__(
        self,
        stats: dict,
        state_dim: int = 2,
        hidden_dims: list[int] | None = None,
        activation: nn.Module | None = None,
        restitution: float = 0.9, # restituition  - 0.0 means no bounce, 1.0 means perfect bounce with no loss of energy
        step_size: float = 0.05,
    ) -> None:
        super().__init__()
        if state_dim != 2:
            raise ValueError("EventNeuralODERegressor expects state_dim=2 for [x, v].")

        self.func = AccelerationMLPODEFunc(hidden_dims=hidden_dims, activation=activation)

        r = min(max(float(restitution), 1e-4), 1.0 - 1e-4)
        self.restitution_logit = nn.Parameter(torch.logit(torch.as_tensor(r))) # restitution is also trainable 

        self.step_size = step_size

        self.dt_std = stats["dt"]
        self.z0_std = stats["z0"]
        self.y_std = stats["y"]


    @property
    def restitution(self) -> torch.Tensor:
        return torch.sigmoid(self.restitution_logit)

    def _drift(self, z: torch.Tensor) -> torch.Tensor:
        return self.func(z)

    def _event(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return (x0 >= 0) & (x1 < 0.0) 

    def _integrate(self, dt_phys: torch.Tensor, z0_phys: torch.Tensor) -> torch.Tensor:
        n = max(16, math.ceil(float(dt_phys.max()) / max(self.step_size, 1e-6))) # at least 16 steps
        t_step = dt_phys / n
        e = self.restitution
        z = z0_phys
        for _ in range(n):
            z_new = rk4_step(self._drift, z, t_step) #same as normal neural ODE
            x0, x1 = z[:, 0:1], z_new[:, 0:1]
            crossed = self._event(x0, x1) # check if current step crosses the event (x=0)

            #check which part the event happens
            alpha = (x0 / (x0 - x1).clamp_min(1e-9)).clamp(0.0, 1.0)
            z_cross = rk4_step(self._drift, z, alpha * t_step)
            v_after = e * z_cross[:, 1:2].abs() # preserve the speed but reverse the direction and apply restitution
            z_bounce = torch.cat([torch.zeros_like(z_cross[:, 0:1]), v_after], dim=1)
            z_after = rk4_step(self._drift, z_bounce, (1.0 - alpha) * t_step)
            z = torch.where(crossed, z_after, z_new) # if event is crossed, use the bounce result, otherwise use the normal integration result
        return z

    def forward(self, dt_norm: torch.Tensor, z0_norm: torch.Tensor) -> torch.Tensor:
        dt_phys = self.dt_std.decode(dt_norm)
        z0_phys = self.z0_std.decode(z0_norm)
        z_final = self._integrate(dt_phys, z0_phys)
        return self.y_std.encode(z_final)

    @torch.no_grad()
    def trajectory(self, dt_grid_norm: torch.Tensor, z0_norm: torch.Tensor) -> torch.Tensor:

        if z0_norm.shape[0] != 1:
            raise ValueError("trajectory expects exactly one initial state")
        z0_phys = self.z0_std.decode(z0_norm)
        dt_grid_phys = self.dt_std.decode(dt_grid_norm.unsqueeze(1))
        z0_rep = z0_phys.expand(dt_grid_phys.shape[0], -1).contiguous()
        z = self._integrate(dt_grid_phys, z0_rep)
        return self.y_std.encode(z)
