from __future__ import annotations

import torch
import torch.nn as nn


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


class RNNBaseline(nn.Module):
    """
    Step k:  [delta_t, x_k, v_k]  -->  LSTM(h_k, c_k)  -->  [x_{k+1}, v_{k+1}]
    """

    def __init__(self, hidden_size: int = 128, num_steps: int = 20) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.lstm = nn.LSTM(
            input_size=4,   # [dt_norm, step_frac, x, v]
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )


    def forward(self, dt: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
        dt = _ensure_column_vector(dt, z0)          # (B,1) total horizon
        t_step = dt / self.num_steps                    # (B,1) per-step Δt
        z = z0
        h, c = None, None
        for k in range(self.num_steps):
            t_frac = torch.full_like(dt, k / self.num_steps)        
            x = torch.cat([t_step, t_frac, z], dim=1).unsqueeze(1)     
            out, (h, c) = self.lstm(x, (h, c) if h is not None else None)
            dz = self.decoder(out[:, 0, :])                          
            z = z + t_step * dz                                         
        return z
