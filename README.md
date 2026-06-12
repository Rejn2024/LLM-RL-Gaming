# LLM + RL Hex War Game

A modular, real-time 2D hex-cell simulation baseline for the architecture:

```text
Local Ollama strategic planner -> validated task decomposition -> RL/search tactical controllers
                                      ^                           |
                                      +---- simulation feedback <-+
```

The engine supports multiple **blue** friendly, **red** hostile, and **white** observer/civilian units; terrain-dependent movement and defence; fog of war; kinetic and electronic-warfare (EW) actions; and interchangeable human, bot, RL, or LLM controllers. White units are deliberately neutral and do not affect victory determination.

> This is a research simulation, not a model of real-world weapon effects. Its abstract values and mechanics should not be interpreted operationally.

## Windows 11 / RTX 5080 setup

Install Python 3.11+ and a current NVIDIA driver. In PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
# Install a CUDA-enabled PyTorch build appropriate for the current driver from pytorch.org first.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[visual,dev]"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

PyTorch automatically selects CUDA in the DQN trainer and otherwise falls back to CPU. The network and replay batch are intentionally moderate; increase layer widths, replay size, parallel environments, or batch size after profiling the 5080.

## Run

```powershell
war-game                       # pygame visualizer
war-game --headless --steps 20 # deterministic headless smoke run
war-game --stochastic --seed 42 # sample combat/EW outcomes reproducibly
python -m war_game.rl.dqn --episodes 500 --output runs/ew_dqn --stochastic
tensorboard --logdir runs
pytest
```

The visualizer advances bots in real time. The engine uses deterministic expected outcomes by default and can run much faster than real time for training/search. Pass `--stochastic` at startup to sample attack-damage variance and EW success; `--seed` makes those samples reproducible. Programmatic callers can make the same choice with `GameEngine(..., stochastic=True)` or `EWSearchEnv(..., stochastic=True)`.

## Ollama strategic planning

Install Ollama and fetch a local instruction model, for example `ollama pull qwen2.5:7b`. The adapter uses Ollama's local `/api/generate` endpoint without additional dependencies:

```python
from war_game.engine import GameEngine
from war_game.llm import OllamaPlanner, actions_from_plan, observation
from war_game.model import Faction
from war_game.scenario import demo_scenario

game_map, units = demo_scenario()
engine = GameEngine(game_map, units)
planner = OllamaPlanner(model="qwen2.5:7b")
plan = planner.plan(observation(engine, Faction.BLUE))
for action in actions_from_plan(plan, engine, Faction.BLUE):
    engine.submit(action)
engine.step()
```

Only faction-visible information is supplied to the planner, and every LLM-produced action is treated as untrusted input and validated before submission. For continual learning, persist observations, plans, validated actions, event streams, and rewards as trajectories, then evaluate candidate planner/controller checkpoints against fixed seeds before promotion.

## RL/search harness

`EWSearchEnv` provides a compact Gym-like interface without requiring Gymnasium. It teaches a blue EW unit to search for a red target, decide when to jam, and optionally attack. Actions are six neighbor moves, EW, kinetic attack, and wait. `dqn.py` supplies:

- a PyTorch DQN policy/target network;
- replay buffer, epsilon exploration, target synchronization, and gradient clipping;
- automatic CUDA/CPU selection;
- TensorBoard reward, epsilon, and loss metrics;
- checkpoint output for later tactical-controller integration.

For broader tactical search, clone the engine at a decision boundary, roll out candidate `Action` sequences with heuristic/opponent controllers, and score outcomes. The `Controller` protocol is the extension seam for learned policies, tree search, remote opponents, local opponent LLMs, and white-force behavior.

## Project layout

```text
src/war_game/model.py       domain types, factions, terrain, actions
src/war_game/map.py         axial hex geometry and A* pathfinding
src/war_game/engine.py      deterministic/stochastic simulation and fog of war
src/war_game/controllers.py pluggable controller protocol and baseline bot
src/war_game/visual.py      optional pygame rendering
src/war_game/llm.py         Ollama observations, prompts, and action validation
src/war_game/rl/            EW/search environment and PyTorch DQN trainer
tests/                      headless simulation, LLM boundary, and RL contract tests
```

## Extension roadmap

1. Add scenario files and explicit objectives/civilian-protection rewards.
2. Add line-of-sight, communications, logistics, and richer unit capabilities.
3. Wrap `EWSearchEnv` for Gymnasium/Stable-Baselines or implement parallel vector environments.
4. Add tactical tree search whose leaf evaluator is learned from simulation outcomes.
5. Record planner-task-action trajectories and train/evaluate local LLM adapters offline.
6. Add self-play leagues with red and white `Controller` implementations and immutable evaluation sets.
