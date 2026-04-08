"""
ICU-Sepsis-MIMIC-IV: A tabular MDP benchmark for sepsis treatment optimization.

Built from MIMIC-IV using Sepsis-3 criteria (SOFA >= 4).
Extends ICU-Sepsis (Choudhary et al. 2024) with:
  - MIMIC-IV instead of MIMIC-III
  - Sepsis-3 cohort criteria
  - 50 actions (5 vasopressor x 5 fluid x 2 ventilation) vs 25
  - Larger policy differentiation headroom
"""

import numpy as np
import pandas as pd
import pickle
import os
from collections import defaultdict
import gymnasium as gym
from gymnasium import spaces

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

class ICUSepsisMIMICEnv(gym.Env):
    """
    ICU-Sepsis-MIMIC-IV Gymnasium Environment.

    State space:  750 discrete states + 2 terminal states (survive/die)
    Action space: 50 discrete actions
                  action_id = vaso_level * 10 + fluid_level * 2 + vent_level
                  vaso_level: 0-4 (none to very high norepinephrine equivalent)
                  fluid_level: 0-4 (none to very high IV fluids)
                  vent_level: 0-1 (not ventilated / ventilated)
    Reward:       +1 for survival at episode end, 0 otherwise
    Discount:     gamma = 1.0

    Follows ICU-Sepsis (Choudhary et al. 2024) design conventions:
    - Inadmissible actions map to uniform over admissible actions
    - Binary terminal reward based on 90-day mortality
    - 4-hour time blocks
    """

    metadata = {'render_modes': []}

    # MDP parameters
    K = 750           # number of non-terminal states
    N_ACTIONS = 50    # total actions
    TAU = 20          # transition threshold

    # Action metadata
    VASO_LABELS = ['None', 'Low', 'Medium', 'High', 'Very High']
    FLUID_LABELS = ['None', 'Low', 'Medium', 'High', 'Very High']
    VENT_LABELS = ['Not Ventilated', 'Ventilated']

    def __init__(self, data_dir=None, tau=20):
        super().__init__()

        self.data_dir = data_dir or DATA_DIR
        self.tau = tau
        self.S_SURVIVE = self.K
        self.S_DIE = self.K + 1
        self.N_STATES = self.K + 2

        self.action_space = spaces.Discrete(self.N_ACTIONS)
        self.observation_space = spaces.Discrete(self.N_STATES)

        self._build_action_map()
        self._load_mdp()

        self.current_state = None
        self.done = False

    def _build_action_map(self):
        """Build mapping from action_id to clinical intervention levels."""
        self.action_map = {}
        self.action_descriptions = {}
        for vaso in range(5):
            for fluid in range(5):
                for vent in range(2):
                    aid = vaso * 10 + fluid * 2 + vent
                    self.action_map[aid] = {
                        'vaso_level': vaso,
                        'fluid_level': fluid,
                        'vent_level': vent
                    }
                    self.action_descriptions[aid] = (
                        f"Vaso:{self.VASO_LABELS[vaso]} | "
                        f"Fluid:{self.FLUID_LABELS[fluid]} | "
                        f"Vent:{self.VENT_LABELS[vent]}"
                    )

    def _load_mdp(self):
        """Load transition matrix, initial state distribution, expert policy."""
        transitions = pd.read_parquet(
            os.path.join(self.data_dir, 'transitions.parquet')
        )
        trajectories = pd.read_parquet(
            os.path.join(self.data_dir, 'trajectories.parquet')
        )

        # Build transition counts
        trans_counts = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        for _, row in transitions.iterrows():
            s = int(row['state_id'])
            a = int(row['action_id'])
            ns = int(row['next_state_id'])
            c = int(row['count'])
            if s not in [self.S_SURVIVE, self.S_DIE]:
                trans_counts[s][a][ns] += c

        # Normalize — only keep admissible actions (count >= tau)
        self.trans_prob = {}
        for s in trans_counts:
            self.trans_prob[s] = {}
            for a in trans_counts[s]:
                total = sum(trans_counts[s][a].values())
                if total >= self.tau:
                    self.trans_prob[s][a] = {
                        ns: c / total
                        for ns, c in trans_counts[s][a].items()
                    }
            self.trans_prob[s] = {
                a: v for a, v in self.trans_prob[s].items() if v
            }
        self.trans_prob = {
            s: v for s, v in self.trans_prob.items() if v
        }

        # Admissible actions per state
        self.admissible_actions = {
            s: list(actions.keys())
            for s, actions in self.trans_prob.items()
        }

        # Initial state distribution
        init_states = trajectories.groupby('stay_id').first()['state_id'].values
        self.init_state_dist = np.zeros(self.K)
        for s in init_states:
            if 0 <= s < self.K:
                self.init_state_dist[s] += 1
        self.init_state_dist /= self.init_state_dist.sum()

        # Expert policy from empirical data
        expert_counts = defaultdict(lambda: defaultdict(int))
        for _, row in trajectories.iterrows():
            s = int(row['state_id'])
            a = int(row['action_id'])
            if 0 <= s < self.K:
                expert_counts[s][a] += 1

        self.expert_policy = {}
        for s in expert_counts:
            total = sum(expert_counts[s].values())
            self.expert_policy[s] = {
                a: c / total for a, c in expert_counts[s].items()
            }

        # MDP summary stats
        self.n_admissible_pairs = sum(
            len(v) for v in self.admissible_actions.values()
        )

    def _get_transition(self, state, action):
        """
        Get transition distribution for (state, action).
        Inadmissible actions map to uniform over admissible actions
        following ICU-Sepsis design convention.
        """
        if state not in self.trans_prob or not self.trans_prob[state]:
            return {self.S_DIE: 1.0}

        if action in self.trans_prob[state]:
            return self.trans_prob[state][action]

        # Inadmissible: uniform mixture over admissible transitions
        adm = list(self.trans_prob[state].values())
        merged = defaultdict(float)
        for t in adm:
            for ns, p in t.items():
                merged[ns] += p / len(adm)
        return dict(merged)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self.current_state = int(
            np.random.choice(self.K, p=self.init_state_dist)
        )
        self.done = False
        return self.current_state, {}

    def step(self, action):
        assert not self.done, "Episode done. Call reset()."
        assert 0 <= action < self.N_ACTIONS, f"Invalid action {action}"

        trans = self._get_transition(self.current_state, action)
        states = list(trans.keys())
        probs = list(trans.values())
        next_state = int(np.random.choice(states, p=probs))

        reward = 1.0 if next_state == self.S_SURVIVE else 0.0
        terminated = next_state in [self.S_SURVIVE, self.S_DIE]

        info = {
            'action_details': self.action_map[action],
            'action_description': self.action_descriptions[action],
            'admissible': action in self.trans_prob.get(
                self.current_state, {}
            ),
            'prev_state': self.current_state
        }

        self.current_state = next_state
        self.done = terminated

        return next_state, reward, terminated, False, info

    def get_action_description(self, action_id):
        """Human readable description of an action."""
        return self.action_descriptions.get(action_id, "Unknown action")

    def get_admissible_actions(self, state):
        """Return list of admissible actions for a state."""
        return self.admissible_actions.get(state, list(range(self.N_ACTIONS)))

    def compute_policy_return(self, policy_fn, n_episodes=1000, seed=42):
        """Monte Carlo evaluation of a policy function."""
        np.random.seed(seed)
        returns = []
        for _ in range(n_episodes):
            state, _ = self.reset()
            total_reward = 0.0
            for _ in range(200):
                action = policy_fn(state)
                state, reward, done, _, _ = self.step(action)
                total_reward += reward
                if done:
                    break
            returns.append(total_reward)
        return float(np.mean(returns)), float(np.std(returns))

    def random_policy(self, state):
        """Uniform random over all actions."""
        return np.random.randint(0, self.N_ACTIONS)

    def expert_policy_fn(self, state):
        """Sample from empirical clinician action distribution."""
        if state in self.expert_policy:
            actions = list(self.expert_policy[state].keys())
            probs = list(self.expert_policy[state].values())
            return int(np.random.choice(actions, p=probs))
        return np.random.randint(0, self.N_ACTIONS)

    def render(self):
        if self.current_state == self.S_SURVIVE:
            print("Episode ended: SURVIVED")
        elif self.current_state == self.S_DIE:
            print("Episode ended: DIED")
        else:
            print(f"State: {self.current_state}")

    def close(self):
        pass

    def __repr__(self):
        return (
            f"ICUSepsisMIMICEnv("
            f"K={self.K}, "
            f"N_ACTIONS={self.N_ACTIONS}, "
            f"tau={self.tau}, "
            f"n_admissible_pairs={self.n_admissible_pairs})"
        )
