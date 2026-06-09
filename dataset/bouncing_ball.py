import numpy as np
import torch
import torch.nn as nn


class BouncingBall(nn.Module):
    def __init__(self, gravity: float = 9.8, restitution: float = 1.0):
        super().__init__()
        if gravity <= 0.0:
            raise ValueError("gravity must be positive")
        if not 0.0 <= restitution <= 1.0:
            raise ValueError("restitution must be in [0, 1]")

        self.gravity = float(gravity)
        self.restitution = float(restitution)

    def forward(self, t0: float, x0: float, v0: float, T_final: float) -> tuple[float, float]:
        return self.state_at(t0, x0, v0, T_final)

    def _segment_state(self, x_start: float, v_start: float, delta_t: float) -> tuple[float, float]:
        # used when the ball is in the air at T_final
        x_t = x_start + v_start * delta_t - 0.5 * self.gravity * (delta_t ** 2)
        v_t = v_start - self.gravity * delta_t
        return x_t, v_t

    def _time_to_ground(self, x_start: float, v_start: float) -> float:
        discriminant = max(v_start ** 2 + 2.0 * self.gravity * x_start, 0.0)
        return (v_start + np.sqrt(discriminant)) / self.gravity

    def state_at(self, 
        t0: float, x0: float, v0: float, T_final: float,
        max_bounces: int = 100000,
        x_tol: float = 1e-12, v_tol: float = 1e-12, t_tol: float = 1e-12
    ) -> tuple[float, float]:
        t0 = float(t0)
        x0 = float(x0)
        v0 = float(v0)
        T_final = float(T_final)

        if T_final < t0:
            raise ValueError("T_final must be greater than or equal to t0")
        if x0 < 0.0:
            raise ValueError("x0 must be non-negative")

        remaining = T_final - t0
        x_curr = x0
        v_curr = v0

        if remaining <= t_tol:
            return x_curr, v_curr

        for _ in range(max_bounces):
            if remaining <= t_tol:
                return x_curr, v_curr

            tau_hit = self._time_to_ground(x_curr, v_curr)

            if tau_hit <= t_tol:
                # if the ball is at the ground and not moving, return 0.0, 0.0
                if x_curr <= x_tol and abs(v_curr) <= v_tol:
                    return 0.0, 0.0
                # if the ball is at the ground and moving downward, reflect the velocity
                if x_curr <= x_tol and v_curr < 0.0:
                    v_curr = self.restitution * abs(v_curr)
                    x_curr = 0.0
                    continue

            if remaining <= tau_hit:
                # if remaining time is less than the time to hit the ground, update the state
                x_t, v_t = self._segment_state(x_curr, v_curr, remaining)
                x_t = 0.0 if abs(x_t) < x_tol else x_t
                return max(x_t, 0.0), v_t

            impact_speed = np.sqrt(max(v_curr ** 2 + 2.0 * self.gravity * x_curr, 0.0))
            remaining -= tau_hit
            x_curr = 0.0
            v_curr = self.restitution * impact_speed

            if v_curr <= v_tol:
                return 0.0, 0.0

            if self.restitution < 1.0:
                # if restitution is less than 1.0, the ball will eventually settle at the ground
                # first bounce time = 2 * v0 / g
                # next bounce time = 2 * v0 / g * e
                # settling time = 2 * v0 / g * (1 + e + e^2 + ...) = 2 * v0 / g * (1 / (1 - e))
                # if remaining time is greater than the settling time, directly return 0.0, 0.0
                settling_time = (2.0 * v_curr) / (self.gravity * (1.0 - self.restitution))
                if remaining >= settling_time - t_tol:
                    return 0.0, 0.0

        raise RuntimeError("Exceeded the maximum number of bounce updates")

    def trajectory(
        self,
        t0: float,
        x0: float,
        v0: float,
        T_final: float,
        num_points: int = 400,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if num_points < 2:
            raise ValueError("num_points must be at least 2")

        t_values = np.linspace(float(t0), float(T_final), num_points)
        x_values = np.empty_like(t_values)
        v_values = np.empty_like(t_values)

        for idx, t_value in enumerate(t_values):
            x_values[idx], v_values[idx] = self.state_at(t0, x0, v0, float(t_value))

        return t_values, x_values, v_values


def sample_bouncing_ball_dataset(
    num_samples: int,
    BouncingBall,
    T_final: float,
    *,
    gravity: float = 9.8,
    restitution: float = 1.0,
    t0_range: tuple[float, float] = (0.0, 3.0),
    x0_range: tuple[float, float] = (0.5, 10.0),
    v0_range: tuple[float, float] = (-5.0, 5.0),
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    t0_low, t0_high = map(float, t0_range)
    x0_low, x0_high = map(float, x0_range)
    v0_low, v0_high = map(float, v0_range)

    if not 0.0 <= t0_low <= t0_high < float(T_final):
        raise ValueError("t0_range must satisfy 0 <= low <= high < T_final")
    if not 0.0 <= x0_low <= x0_high:
        raise ValueError("x0_range must satisfy 0 <= low <= high")

    ball = BouncingBall(gravity=gravity, restitution=restitution)

    x_data = np.empty((num_samples, 4), dtype=np.float64)
    y_data = np.empty((num_samples, 2), dtype=np.float64)

    for idx in range(num_samples):
        t0 = np.random.uniform(t0_low, t0_high)
        tT = T_final
        x0 = np.random.uniform(x0_low, x0_high)
        v0 = np.random.uniform(v0_low, v0_high)
        xT, vT = ball.state_at(t0, x0, v0, tT)

        x_data[idx] = (t0, tT, x0, v0)
        y_data[idx] = (xT, vT)

    return torch.tensor(x_data, dtype=dtype), torch.tensor(y_data, dtype=dtype)


def main():
    num_samples = 8
    T_final = 5.0

    x_data, y_data = sample_bouncing_ball_dataset(
        num_samples=num_samples,
        BouncingBall=BouncingBall,
        T_final=T_final,
        gravity=9.8,
        restitution=0.9,
    )

    print("x_data shape:", tuple(x_data.shape))
    print("y_data shape:", tuple(y_data.shape))
    print("first input sample:", x_data[0])
    print("first target sample:", y_data[0])


if __name__ == "__main__":
    main()
