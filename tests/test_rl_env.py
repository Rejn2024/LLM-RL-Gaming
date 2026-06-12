import numpy as np

from war_game.rl.env import EWSearchEnv


def test_environment_contract():
    env = EWSearchEnv(max_steps=2, seed=3)
    observation, info = env.reset(3)
    assert observation.shape == (env.observation_size,)
    assert observation.dtype == np.float32 and info == {}
    _, reward, terminated, truncated, info = env.step(8)
    assert isinstance(reward, float) and not terminated and not truncated and "distance" in info
    _, _, _, truncated, _ = env.step(8)
    assert truncated
