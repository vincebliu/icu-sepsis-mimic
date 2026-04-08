"""
Offline dataset loader for ICU-Sepsis-MIMIC-IV.

Provides three dataset splits:
- Real:   Actual MIMIC-IV clinician trajectories (10,790 patients)
- Expert: Trajectories from optimal policy via value iteration
- Random: Trajectories from uniform random policy

Each dataset is a list of trajectories, where each trajectory is a list of
(state, action, reward, next_state, done) tuples.
"""

import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'datasets')

SPLITS = ['real', 'expert', 'random']

SPLIT_STATS = {
    'real':   {'n_trajectories': 10790, 'survival_rate': 0.783},
    'expert': {'n_trajectories': 10000, 'survival_rate': 0.883},
    'random': {'n_trajectories': 10000, 'survival_rate': 0.702},
}

def load_dataset(split, data_dir=None, as_trajectories=True):
    """
    Load an offline dataset split.

    Args:
        split: One of 'real', 'expert', 'random'
        data_dir: Path to datasets directory (uses package default if None)
        as_trajectories: If True, returns list of trajectory dicts.
                        If False, returns raw DataFrame.

    Returns:
        If as_trajectories=True:
            List of dicts, each with keys:
            - 'observations': np.array of state ids
            - 'actions': np.array of action ids
            - 'rewards': np.array of rewards
            - 'next_observations': np.array of next state ids
            - 'terminals': np.array of done flags
        If as_trajectories=False:
            pandas DataFrame
    """
    assert split in SPLITS, f"Split must be one of {SPLITS}, got '{split}'"

    data_dir = data_dir or DATA_DIR
    path = os.path.join(data_dir, f'{split}.parquet')

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            f"Please ensure datasets are downloaded."
        )

    df = pd.read_parquet(path)

    if not as_trajectories:
        return df

    # Convert to list of trajectory dicts
    trajectories = []
    for traj_id, group in df.groupby('trajectory_id'):
        group = group.sort_values('step').reset_index(drop=True)
        trajectories.append({
            'trajectory_id': traj_id,
            'observations': group['state'].values.astype(np.int32),
            'actions': group['action'].values.astype(np.int32),
            'rewards': group['reward'].values.astype(np.float32),
            'next_observations': group['next_state'].values.astype(np.int32),
            'terminals': group['done'].values.astype(bool),
            'vaso_levels': group['vaso_level'].values.astype(np.int32),
            'fluid_levels': group['fluid_level'].values.astype(np.int32),
            'vent_levels': group['vent'].values.astype(np.int32),
        })

    return trajectories


def get_dataset_stats(split, data_dir=None):
    """Return summary statistics for a dataset split."""
    df = load_dataset(split, data_dir=data_dir, as_trajectories=False)
    trajs = df.groupby('trajectory_id')

    return {
        'split': split,
        'n_trajectories': trajs.ngroups,
        'n_transitions': len(df),
        'survival_rate': df.groupby('trajectory_id')['reward'].sum().gt(0).mean(),
        'avg_trajectory_length': trajs.size().mean(),
        'min_trajectory_length': trajs.size().min(),
        'max_trajectory_length': trajs.size().max(),
    }


def list_datasets():
    """Print available dataset splits and their statistics."""
    print("ICU-Sepsis-MIMIC-IV Offline Datasets")
    print("="*50)
    for split in SPLITS:
        stats = SPLIT_STATS[split]
        print(f"\n{split.upper()}")
        print(f"  Trajectories:  {stats['n_trajectories']:,}")
        print(f"  Survival rate: {stats['survival_rate']:.3f}")
