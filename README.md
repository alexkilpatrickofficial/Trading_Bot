# Crypto Trading GA with Hybrid PPO

A **Genetic Algorithm (GA)** project that evolves **chromosomes** (architectures) for a **Hybrid PPO** trading model in a cryptocurrency environment. This system combines **reinforcement learning** (via [Stable-Baselines3 PPO](https://github.com/DLR-RM/stable-baselines3)) with an **auxiliary price prediction** head to improve feature extraction and trading performance.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preprocessing](#data-preprocessing)
  - [Running the GA + PPO Training](#running-the-ga--ppo-training)
- [Project Structure](#project-structure)
- [Technical Highlights](#technical-highlights)
- [Roadmap / Future Work](#roadmap--future-work)
- [License](#license)
- [Author / Contact](#author--contact)

## Overview
This repository implements a **hybrid approach** for cryptocurrency trading using:

- **Genetic Algorithms (GA)** to evolve model architectures and hyperparameters (chromosomes).
- **Stable-Baselines3 PPO** for on-policy reinforcement learning.
- An **auxiliary price-prediction head** that provides an extra learning signal to the feature extractor, even when trading performance is suboptimal.

## Features

### Genetic Algorithm
- Evolves different neural architectures and hyperparameters.
- Automatic architecture validation via `validate_chromosome`.
- Modular block design (CNN, LSTM, TCN, MLP, etc.) combined in a flexible pipeline.

### Hybrid PPO
- Utilizes Stable-Baselines3 PPO for on-policy learning.
- Integrates an auxiliary MSE loss for price prediction alongside the main profit-based objective.
- Custom policy (`HybridPolicy`) dynamically handles action shape mismatches.

### Callbacks
- **TensorBoardCallback** for logging training metrics.
- **CheckpointCallback** for model saving at regular intervals.
- **EarlyStoppingCallback** to halt training if multiple bad episodes occur.

### Custom Environment
- `MultiStockTradingEnv` with a multi-discrete action space (`[3, 10]` → buy/sell/hold plus trade size).
- Robust logging and shape-handling to ensure proper action conversion.

### Data Preprocessing
- Processes raw crypto CSV data via `main_preprocessing.py`.
- Saves preprocessed data to HDF5.
- Trains and saves a scaler for feature normalization.

## Installation

### Clone the Repository
```bash
git clone https://github.com/alexkilpatrickofficial/Trading_Bot.git
cd Trading_Bot
```

### Create/Activate a Virtual Environment
```bash
python -m venv venv
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### GPU Check (Optional)
Ensure that PyTorch (with CUDA support) is installed properly if you want GPU acceleration.

## Usage

### Data Preprocessing
Before running the GA optimization, process your raw dataset:

```bash
python main_preprocessing.py
```
This script:
- Loads raw CSV data (e.g., `Gemini_BTCUSD_1h.csv`).
- Saves dataset location for future runs.
- Preprocesses data, normalizes features, and saves output to HDF5.
- Trains and saves a scaler for feature normalization.

### Running the GA + PPO Training
Ensure data preprocessing is completed, then start the GA optimization:

```bash
python main.py
```
This script:
- Loads preprocessed data.
- Initializes or resumes GA population.
- Evolves chromosomes using `evaluate(...)`.
- Trains PPO models with `HybridPolicy`.

### Monitor Training
Launch TensorBoard to track training progress:
```bash
tensorboard --logdir=logs
```
Navigate to [http://localhost:6006](http://localhost:6006) to view metrics.

## Project Structure

```
Trading_Bot/
├── main.py                   # GA optimization & PPO training orchestration
├── evaluation.py             # Chromosome evaluation & model training
├── envs/
│   ├── multi_stock_env.py    # Custom trading environment
├── hybrid_policy.py          # Custom HybridPolicy for PPO
├── models.py                 # Modular architecture builder (CNN, LSTM, etc.)
├── ga_utils.py               # Genetic Algorithm utilities
├── callbacks.py              # Custom callbacks for logging, checkpointing
├── data_preprocessing.py     # Data processing utilities
├── main_preprocessing.py     # Preprocesses and normalizes dataset
├── requirements.txt          # Dependency list
```

## Technical Highlights

### Multi-Discrete Action Space
Supports actions defined as `[3, 10]` (buy/sell/hold plus discrete trade size). Custom shape-handling ensures proper action conversion.

### Auxiliary Loss
The `HybridPolicy` merges PPO loss with an auxiliary MSE loss for price prediction, ensuring feature learning even when trading decisions are suboptimal.

### Evolutionary Architecture Search
The GA evolves modular architectures (CNN, LSTM, TCN, MLP) and hyperparameters, validated via simulation in `simulate_modular_blocks`.

## Roadmap / Future Work

- **Custom Rollout Buffer**: Handle multi-discrete action shape issues.
- **Data Pipeline Enhancements**: Support multi-asset trading.
- **Enhanced Reward Shaping**: Improve reward signals beyond profit-based metrics.
- **Scaling Up**: Implement multi-agent setups for parallel training.

## License
This project is licensed under the GRU Public License.

## Author / Contact
**Alex Kilpatrick** (@alexkilpatrickofficial)  
Email: [alexkilpatrick@proton.me](mailto:alexkilpatrick@proton.me)

