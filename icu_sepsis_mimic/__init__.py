"""
ICU-Sepsis-MIMIC-IV: A tabular MDP benchmark for offline RL in sepsis treatment.

Built from MIMIC-IV (v3.1) using Sepsis-3 criteria.

Quick start:
    from icu_sepsis_mimic import ICUSepsisMIMICEnv, load_dataset

    # Online RL
    env = ICUSepsisMIMICEnv()
    state, _ = env.reset()
    next_state, reward, done, _, info = env.step(action=0)

    # Offline RL
    dataset = load_dataset('real')  # or 'expert', 'random'
    for trajectory in dataset:
        states = trajectory['observations']
        actions = trajectory['actions']
        rewards = trajectory['rewards']
"""

from icu_sepsis_mimic.env import ICUSepsisMIMICEnv
from icu_sepsis_mimic.datasets import load_dataset, get_dataset_stats, list_datasets

__version__ = '0.1.0'
__author__ = 'Chenhui Wang, Vincent B Liu, Deqian Kong, Edouardo Honig, Ying Nian Wu'

__all__ = [
    'ICUSepsisMIMICEnv',
    'load_dataset',
    'get_dataset_stats',
    'list_datasets',
]
