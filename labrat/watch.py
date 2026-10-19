"""Watch a trained policy live in a MuJoCo window.

Usage: uv run python -m labrat.watch runs/<run_id> [--episodes 10]
"""

import argparse
import time
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import yaml
from stable_baselines3 import SAC

gym.register_envs(gymnasium_robotics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a trained policy.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    cfg = yaml.safe_load((args.run_dir / "config.yaml").read_text())
    env = gym.make(cfg["env_id"], render_mode="human")
    model = SAC.load(args.run_dir / "model.zip", env=env)
    step_seconds = 1 / env.metadata["render_fps"]

    for episode in range(args.episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            time.sleep(step_seconds)  # play at real-time speed
        print(f"episode {episode + 1}: {'success' if info['is_success'] else 'fail'}")
    env.close()


if __name__ == "__main__":
    main()
