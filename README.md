# ICU-Sepsis-MIMIC-IV

A phenotype-stratified tabular MDP benchmark for sepsis treatment optimization, built from MIMIC-IV v3.1.

## Overview

ICU-Sepsis-MIMIC-IV extends ICU-Sepsis (Choudhary et al. 2024) with:
- **MIMIC-IV (v3.1)** instead of MIMIC-III
- **Sepsis-3 cohort criteria** (SOFA ≥ 4)
- **50 actions** (5 vasopressor × 5 fluid × 2 ventilation) vs 25
- **Phenotype-stratified evaluation** revealing two clinically distinct patient subgroups

## Environments

### 1. Pooled MDP (ICUSepsisMIMICEnv)
Standard tabular MDP across all 10,790 sepsis patients.

```python
from icu_sepsis_mimic import ICUSepsisMIMICEnv, load_dataset

env = ICUSepsisMIMICEnv()
state, _ = env.reset()
next_state, reward, done, _, info = env.step(action=0)

dataset = load_dataset('real')  # 'expert', 'random'
```

### 2. Phenotype-Stratified MDP (ICUSepsisAugmentedEnv)
Separate transition dynamics per phenotype, identified by K-means clustering on admission features.

```python
from icu_sepsis_mimic.augmented_env import ICUSepsisAugmentedEnv

# Compensated Sepsis (n=7,652, mortality 14.3%)
env = ICUSepsisAugmentedEnv(phenotype=0)

# Decompensated Sepsis (n=3,138, mortality 39.5%)
env = ICUSepsisAugmentedEnv(phenotype=1)

# Sample phenotype proportionally
env = ICUSepsisAugmentedEnv(phenotype=None)

state, info = env.reset()
print(info['phenotype_name'])  # 'Compensated Sepsis' or 'Decompensated Sepsis'
```

## MDP Properties

| Property | ICU-Sepsis | Ours (Pooled) | Ph0 Compensated | Ph1 Decompensated |
|----------|-----------|---------------|-----------------|-------------------|
| Patients | ~17,000 | 10,790 | 7,652 | 3,138 |
| Mortality | ~20% | 22.5% | 14.3% | 39.5% |
| States | 716 | 750 | 750 | 750 |
| Actions | 25 | 50 | 50 | 50 |
| Optimal (VI) | 0.880 | 0.971 | 0.934 | 0.578 |
| Clinician | 0.780 | 0.783 | 0.843 | 0.453 |
| Random | — | 0.702 | 0.689 | 0.374 |

## Benchmark Results

Results on Real dataset (5 seeds, mean ± SD):

| Method | Ph0 Compensated | Ph1 Decompensated |
|--------|----------------|-------------------|
| Random | 0.689 | 0.374 |
| Sarsa | 0.814 ± 0.008 | 0.402 ± 0.012 |
| BC | 0.856 ± 0.003 | 0.467 ± 0.005 |
| IQL | 0.862 ± 0.002 | 0.472 ± 0.002 |
| TT | 0.874 ± 0.001 | 0.502 ± 0.009 |
| Clinician | 0.843 | 0.453 |
| Optimal (VI) | 0.934 | 0.578 |

**Key finding:** All methods exceed clinician performance for Compensated patients. For Decompensated patients (39.5% mortality), no method substantially improves over clinician performance — this remains an open challenge.

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

## Data Access

This benchmark uses MIMIC-IV. Access requires completion of CITI training and a PhysioNet data use agreement. See https://physionet.org/content/mimiciv/.

## License
MIT