"""
ICU-Sepsis-MIMIC-IV Augmented Gymnasium Environment

The augmented MDP conditions on patient phenotype (s0) at episode start.
Each phenotype has its own transition dynamics, initial state distribution,
and clinician policy, while sharing the same 750-state physiological space.

Usage:
    env = ICUSepsisAugmentedEnv(data_dir='...', phenotype=0)  # fixed phenotype
    env = ICUSepsisAugmentedEnv(data_dir='...', phenotype=None)  # sample phenotype
"""

import numpy as np
import pickle
import os
import gymnasium as gym
from gymnasium import spaces


class ICUSepsisAugmentedEnv(gym.Env):
    """
    Augmented ICU-Sepsis-MIMIC-IV Gymnasium Environment.

    State space:  750 physiological states + 2 terminal (survive/die)
    Action space: 50 actions (5 vaso x 5 fluid x 2 vent)
    Phenotype:    0 = Compensated Sepsis (mortality ~14%)
                  1 = Decompensated Sepsis (mortality ~40%)

    If phenotype=None, phenotype is sampled at reset() proportional
    to cohort size (71% phenotype 0, 29% phenotype 1).
    """

    K = 750
    N_ACTIONS = 50
    N_PHENOTYPES = 2
    PHENOTYPE_NAMES = {
        0: 'Compensated Sepsis',
        1: 'Decompensated Sepsis',
    }
    PHENOTYPE_SIZES = {0: 7652, 1: 3138}

    VASO_LABELS  = ['None', 'Low', 'Medium', 'High', 'Very High']
    FLUID_LABELS = ['None', 'Low', 'Medium', 'High', 'Very High']
    VENT_LABELS  = ['Not Ventilated', 'Ventilated']

    def __init__(self, data_dir=None, phenotype=None):
        super().__init__()

        self.data_dir = data_dir or os.path.join(
            os.path.dirname(__file__), '..', 'data', 'augmented'
        )
        self.S_SURVIVE = self.K
        self.S_DIE     = self.K + 1
        self.N_STATES  = self.K + 2

        self.action_space      = spaces.Discrete(self.N_ACTIONS)
        self.observation_space = spaces.Discrete(self.N_STATES)

        self._phenotype_fixed = phenotype
        self._load_mdp()
        self._build_action_map()

        self.current_state    = None
        self.current_phenotype = None
        self.done             = False

    def _load_mdp(self):
        d = self.data_dir
        with open(f'{d}/trans_probs.pkl',       'rb') as f:
            self.trans_probs = pickle.load(f)
        with open(f'{d}/admissibles.pkl',        'rb') as f:
            self.admissibles = pickle.load(f)
        with open(f'{d}/value_fns.pkl',          'rb') as f:
            self.value_fns   = pickle.load(f)
        with open(f'{d}/policies.pkl',           'rb') as f:
            self.opt_policies = pickle.load(f)
        with open(f'{d}/init_dists.pkl',         'rb') as f:
            self.init_dists  = pickle.load(f)
        with open(f'{d}/clinician_policies.pkl', 'rb') as f:
            self.clin_policies = pickle.load(f)

        # Phenotype sampling weights
        total = sum(self.PHENOTYPE_SIZES.values())
        self._phenotype_probs = np.array([
            self.PHENOTYPE_SIZES[ph] / total
            for ph in range(self.N_PHENOTYPES)
        ])

    def _build_action_map(self):
        self.action_map = {}
        self.action_descriptions = {}
        for vaso in range(5):
            for fluid in range(5):
                for vent in range(2):
                    aid = vaso * 10 + fluid * 2 + vent
                    self.action_map[aid] = {
                        'vaso_level': vaso,
                        'fluid_level': fluid,
                        'vent_level': vent,
                    }
                    self.action_descriptions[aid] = (
                        f"Vaso:{self.VASO_LABELS[vaso]} | "
                        f"Fluid:{self.FLUID_LABELS[fluid]} | "
                        f"Vent:{self.VENT_LABELS[vent]}"
                    )

    def _get_transition(self, state, action, phenotype):
        tp = self.trans_probs[phenotype]
        adm = self.admissibles[phenotype]
        if state not in tp or not tp[state]:
            return {self.S_DIE: 1.0}
        if action in tp[state]:
            return tp[state][action]
        # Inadmissible: uniform over admissible
        acts = adm.get(state, [])
        if not acts:
            return {self.S_DIE: 1.0}
        from collections import defaultdict
        merged = defaultdict(float)
        for a in acts:
            if a in tp[state]:
                for ns, p in tp[state][a].items():
                    merged[ns] += p / len(acts)
        return dict(merged)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # Sample or fix phenotype
        if self._phenotype_fixed is not None:
            self.current_phenotype = self._phenotype_fixed
        else:
            self.current_phenotype = int(np.random.choice(
                self.N_PHENOTYPES, p=self._phenotype_probs
            ))

        # Sample initial state from phenotype distribution
        self.current_state = int(np.random.choice(
            self.K, p=self.init_dists[self.current_phenotype]
        ))
        self.done = False

        info = {'phenotype': self.current_phenotype,
                'phenotype_name': self.PHENOTYPE_NAMES[self.current_phenotype]}
        return self.current_state, info

    def step(self, action):
        assert not self.done, "Episode done. Call reset()."
        assert 0 <= action < self.N_ACTIONS

        trans = self._get_transition(
            self.current_state, action, self.current_phenotype
        )
        states = list(trans.keys())
        probs  = list(trans.values())
        next_state = int(np.random.choice(states, p=probs))

        reward     = 1.0 if next_state == self.S_SURVIVE else 0.0
        terminated = next_state in [self.S_SURVIVE, self.S_DIE]

        info = {
            'phenotype': self.current_phenotype,
            'phenotype_name': self.PHENOTYPE_NAMES[self.current_phenotype],
            'action_description': self.action_descriptions[action],
            'admissible': action in self.trans_probs[self.current_phenotype].get(
                self.current_state, {}
            ),
        }

        self.current_state = next_state
        self.done = terminated
        return next_state, reward, terminated, False, info

    def random_policy(self, state):
        acts = self.admissibles[self.current_phenotype].get(
            state, list(range(self.N_ACTIONS))
        )
        return int(np.random.choice(acts))

    def clinician_policy(self, state):
        cp = self.clin_policies[self.current_phenotype]
        if state in cp:
            return cp[state]
        return self.random_policy(state)

    def optimal_policy(self, state):
        op = self.opt_policies[self.current_phenotype]
        if state in op:
            return op[state]
        return self.random_policy(state)

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

    def get_admissible_actions(self, state):
        return self.admissibles[self.current_phenotype].get(
            state, list(range(self.N_ACTIONS))
        )

    def get_action_description(self, action_id):
        return self.action_descriptions.get(action_id, "Unknown")

    def __repr__(self):
        return (
            f"ICUSepsisAugmentedEnv("
            f"K={self.K}, N_ACTIONS={self.N_ACTIONS}, "
            f"N_PHENOTYPES={self.N_PHENOTYPES}, "
            f"phenotype={self._phenotype_fixed})"
        )
