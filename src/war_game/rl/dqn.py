"""PyTorch DQN training harness with TensorBoard metrics."""

from __future__ import annotations

import argparse
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from .env import EWSearchEnv


class QNetwork(nn.Module):
    def __init__(self, observations: int, actions: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observations, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, actions),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


def train(
    episodes: int = 500, output: str = "runs/ew_dqn", seed: int = 7, stochastic: bool = False
) -> Path:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = EWSearchEnv(seed=seed, stochastic=stochastic)
    policy = QNetwork(env.observation_size, env.action_size).to(device)
    target = QNetwork(env.observation_size, env.action_size).to(device)
    target.load_state_dict(policy.state_dict())
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-4)
    replay: deque = deque(maxlen=100_000)
    writer = SummaryWriter(output)
    global_step = 0
    for episode in range(episodes):
        state, _ = env.reset(seed + episode)
        total = 0.0
        for _ in range(env.max_steps):
            epsilon = max(0.05, 1.0 - global_step / 30_000)
            with torch.no_grad():
                action = (
                    random.randrange(env.action_size)
                    if random.random() < epsilon
                    else int(policy(torch.tensor(state, device=device).unsqueeze(0)).argmax())
                )
            next_state, reward, done, truncated, _ = env.step(action)
            replay.append((state, action, reward, next_state, done))
            state, total, global_step = next_state, total + reward, global_step + 1
            if len(replay) >= 256:
                batch = random.sample(replay, 128)
                states, actions, rewards, next_states, dones = zip(*batch)
                states_t = torch.tensor(np.array(states), device=device)
                actions_t = torch.tensor(actions, device=device).unsqueeze(1)
                rewards_t = torch.tensor(rewards, device=device, dtype=torch.float32)
                next_t = torch.tensor(np.array(next_states), device=device)
                dones_t = torch.tensor(dones, device=device, dtype=torch.float32)
                q = policy(states_t).gather(1, actions_t).squeeze(1)
                with torch.no_grad():
                    expected = rewards_t + 0.98 * (1 - dones_t) * target(next_t).max(1).values
                loss = nn.functional.smooth_l1_loss(q, expected)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 10)
                optimizer.step()
                writer.add_scalar("train/loss", loss.item(), global_step)
            if global_step % 500 == 0:
                target.load_state_dict(policy.state_dict())
            if done or truncated:
                break
        writer.add_scalar("episode/reward", total, episode)
        writer.add_scalar("episode/epsilon", epsilon, episode)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / "ew_dqn.pt"
    torch.save(
        {
            "model": policy.state_dict(),
            "observations": env.observation_size,
            "actions": env.action_size,
            "device": str(device),
        },
        model_path,
    )
    writer.close()
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--output", default="runs/ew_dqn")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stochastic", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    print(train(args.episodes, args.output, args.seed, args.stochastic))


if __name__ == "__main__":
    main()
