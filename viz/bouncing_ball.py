from pathlib import Path
import sys

import matplotlib.pyplot as plt

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.bouncing_ball import BouncingBall


def plot_bouncing_ball_trajectory(
    ball: BouncingBall,
    t0: float,
    x0: float,
    v0: float,
    T_final: float,
    num_points: int = 400,
    save_path: str | None = None,
):
    t_values, x_values, v_values = ball.trajectory(
        t0=t0,
        x0=x0,
        v0=v0,
        T_final=T_final,
        num_points=num_points,
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_values, x_values, label="x(t)", linewidth=2.0)
    ax.plot(t_values, v_values, label="v(t)", linewidth=2.0, linestyle="--")
    ax.set_xlabel("t")
    ax.set_ylabel("state value")
    ax.set_title("Bouncing Ball Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, ax


def main():
    ball = BouncingBall(gravity=9.8, restitution=0.8)
    plot_bouncing_ball_trajectory(ball, t0=0.0, x0=10.0, v0=0.0, T_final=12.0)
    # plt.show()
    plt.savefig("figures/bouncing_ball_trajectory.png")


if __name__ == "__main__":
    main()
