# ICU-Sepsis-MIMIC-IV

A tabular MDP benchmark for sepsis treatment optimization built from MIMIC-IV.

## Overview

ICU-Sepsis-MIMIC-IV extends ICU-Sepsis (Choudhary et al. 2024) with:
- MIMIC-IV (v3.1) instead of MIMIC-III
- Sepsis-3 cohort criteria (SOFA >= 4)
- 50 actions (5 vasopressor x 5 fluid x 2 ventilation levels) vs 25
- Larger policy differentiation headroom (0.188 vs 0.100)

## MDP Properties

| Property | ICU-Sepsis | Ours |
|----------|-----------|------|
| Patients | ~17,000 | 10,790 |
| States | 716 | 750 |
| Actions | 25 | 50 |
| Cohort | MIMIC-III | MIMIC-IV |
| Criteria | Komorowski | Sepsis-3 |
| Expert return | 0.780 | 0.783 |
| Optimal return | 0.880 | 0.971 |
| Headroom | 0.100 | 0.188 |

## Quick Start

```python
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
```

## Dataset Splits

| Split | Trajectories | Survival Rate |
|-------|-------------|---------------|
| Random | 10,000 | 0.702 |
| Expert | 10,000 | 0.883 |
| Real | 10,790 | 0.783 |

## Installation

```bash
git clone https://github.com/vincebliu/icu-sepsis-mimic.git
cd icu-sepsis-mimic
pip install -e .
```

## Requirements

- Python >= 3.8
- gymnasium >= 0.26.0
- numpy, pandas, pyarrow, scikit-learn

## Citation

```bibtex
@article{liu2026icusepsismimic,
  title={ICU-Sepsis-MIMIC-IV: A Benchmark MDP for Sepsis Treatment Optimization},
  author={Liu, Vincent B and Wang, Chenhui and Kong, Deqian and Honig, Edouardo and Wu, Ying Nian},
  year={2026}
}
```
