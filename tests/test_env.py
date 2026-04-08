"""Basic tests for ICUSepsisMIMICEnv."""
import numpy as np
import pytest
import sys
sys.path.insert(0, '..')
from icu_sepsis_mimic import ICUSepsisMIMICEnv, load_dataset

def test_env_init():
    env = ICUSepsisMIMICEnv()
    assert env.K == 750
    assert env.N_ACTIONS == 50
    assert env.n_admissible_pairs > 0

def test_reset():
    env = ICUSepsisMIMICEnv()
    state, info = env.reset(seed=42)
    assert 0 <= state < env.K
    assert isinstance(info, dict)

def test_step():
    env = ICUSepsisMIMICEnv()
    state, _ = env.reset(seed=42)
    next_state, reward, terminated, truncated, info = env.step(0)
    assert 0 <= next_state <= env.N_STATES
    assert reward in [0.0, 1.0]
    assert isinstance(terminated, bool)
    assert isinstance(info, dict)

def test_episode():
    env = ICUSepsisMIMICEnv()
    state, _ = env.reset(seed=42)
    total_reward = 0.0
    steps = 0
    for _ in range(200):
        action = env.random_policy(state)
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        steps += 1
        if done:
            break
    assert total_reward in [0.0, 1.0]
    assert steps > 0

def test_policy_return():
    env = ICUSepsisMIMICEnv()
    ret, std = env.compute_policy_return(env.random_policy, n_episodes=100)
    assert 0.5 <= ret <= 1.0

def test_load_dataset():
    for split in ['real', 'expert', 'random']:
        dataset = load_dataset(split)
        assert len(dataset) > 0
        traj = dataset[0]
        assert 'observations' in traj
        assert 'actions' in traj
        assert 'rewards' in traj
        assert 'next_observations' in traj
        assert 'terminals' in traj

if __name__ == '__main__':
    test_env_init()
    test_reset()
    test_step()
    test_episode()
    test_policy_return()
    test_load_dataset()
    print("All tests passed!")
