# Crypto Trading GA with Hybrid PPO

A **Genetic Algorithm (GA)** project that evolves **chromosomes** (architectures) for a **Hybrid PPO** trading model in a cryptocurrency environment. This system combines **reinforcement learning** (via [Stable-Baselines3 PPO](https://github.com/DLR-RM/stable-baselines3)) with an **auxiliary price prediction** head to improve feature extraction and trading performance.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
  - [Genetic Algorithm](#genetic-algorithm)
  - [Hybrid PPO](#hybrid-ppo)
  - [Callbacks](#callbacks)
  - [Custom Environment](#custom-environment)
  - [Data Preprocessing](#data-preprocessing)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preprocessing](#data-preprocessing-1)
  - [Running the GA + PPO Training](#running-the-ga--ppo-training)
  - [Monitoring Training](#monitoring-training)
- [Project Structure](#project-structure)
- [Technical Highlights](#technical-highlights)
- [Roadmap / Future Work](#roadmap--future-work)
- [License](#license)
- [Author / Contact](#author--contact)

---

## Overview

This repository implements a **hybrid approach** for cryptocurrency trading using:

- **Genetic Algorithms (GA)** to evolve model architectures and hyperparameters (chromosomes).
- **Stable-Baselines3 PPO** for on-policy reinforcement learning.
- An **auxiliary price-prediction head** that provides an extra learning signal to the feature extractor, even when trading performance is suboptimal.

In our system, the GA evolves various modular architectures (using components such as CNNs, LSTMs, TCNs, and MLPs) which are then used by a custom Hybrid PPO policy. The policy not only focuses on profit-based objectives but also on accurate price prediction, ensuring robust feature learning even when trading decisions incur losses.

---

## Features

### Genetic Algorithm
- **Architecture Evolution**: Evolve neural architectures and hyperparameters encoded as chromosomes.
- **Validation**: Automatic validation of chromosomes via `validate_chromosome` and `modular_shape_is_valid` functions.
- **Modular Design**: Supports building architectures from multiple blocks (CNN, LSTM, TCN, MLP, etc.) as specified in the chromosome.

### Hybrid PPO
- **On-Policy Learning**: Uses Stable-Baselines3 PPO for training the trading agent.
- **Auxiliary Loss**: Incorporates an auxiliary MSE loss for price prediction, allowing the feature extractor to keep learning even when trading rewards are negative.
- **Custom Policy (`HybridPolicy`)**: Manages action shape consistency and logs separate metrics for profit-based and price prediction performance.

### Callbacks
- **TensorBoardCallback**: Logs basic environment metrics (reward, profit, balance) to TensorBoard.
- **CheckpointCallback**: Saves model checkpoints at regular intervals.
- **EarlyStoppingCallback**: Monitors episodes for abnormal terminations and stops training if a threshold is exceeded.
- **TrainingPerformanceCallback**: Logs detailed training metrics such as policy loss, value loss, total loss, explained variance, and auxiliary (price prediction) loss for separate visualizations.

### Custom Environment
- **Simplified Action Space**: Uses a `Discrete(3)` action space: Hold, Buy, or Sell.
- **Robust Observation Handling**: Processes raw data into a flattened observation vector with technical indicators, account metrics, and extra features.
- **Logging and Debugging**: Detailed logging ensures that actions, observations, and rewards are processed as expected.

### Data Preprocessing
- **Raw Data Handling**: Preprocesses raw crypto CSV data (e.g., `Gemini_BTCUSD_1h.csv`).
- **Technical Indicator Calculation**: Computes required technical indicators and UT Bot candle signals.
- **Normalization and Scaling**: Applies scaling (using `RobustScaler` by default) to continuous features and saves the scaler for future runs.
- **Data Caching**: Caches preprocessed data in HDF5 and pickle formats for efficient reloading.

---

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

### (Optional) GPU Check
Ensure that PyTorch with CUDA support is installed properly if you plan to use GPU acceleration.

---

## Usage

### Data Preprocessing
Before running the GA optimization, process your raw dataset:
```bash
python main_preprocessing.py
```
This script:
- Loads raw CSV data (e.g., `Gemini_BTCUSD_1h.csv`).
- Saves the dataset location for future runs.
- Preprocesses data (normalizes, calculates indicators, generates UT Bot signals).
- Saves the processed data to HDF5.
- Trains and saves a scaler for feature normalization.

### Running the GA + PPO Training
Ensure data preprocessing is complete, then start the GA optimization:
```bash
python main.py
```
This script:
- Loads preprocessed data.
- Initializes or resumes a GA population.
- Evaluates each chromosome by creating a PPO model with the custom `HybridPolicy`.
- Evolves chromosomes based on performance metrics.

### Monitoring Training
Launch TensorBoard to track training progress and detailed performance metrics:
```bash
tensorboard --logdir=logs
```
Then visit [http://localhost:6006](http://localhost:6006) in your browser.

---

## Project Structure
```
Trading_Bot/
├── main.py                   # GA optimization & PPO training orchestration
├── evaluation.py             # Chromosome evaluation & model training
├── envs/
│   └── multi_stock_env.py    # Custom trading environment (Discrete action space)
├── hybrid_policy.py          # Custom HybridPolicy for PPO with auxiliary price prediction
├── models.py                 # Modular architecture builder (DynamicHybridFeatureExtractor, ModularBlock)
├── ga_utils.py               # Genetic Algorithm utilities (crossover, mutation, selection, etc.)
├── callbacks.py              # Custom callbacks for logging, checkpointing, early stopping, training performance
├── data_preprocessing.py     # Data preprocessing utilities (loading, indicator calculation, scaling)
├── main_preprocessing.py     # Script to preprocess and normalize dataset
├── requirements.txt          # Dependency list
├── config.py                 # Configuration: feature columns, file paths, constants, etc.
```

---

## Technical Highlights
- **Hybrid Neural Network**: Combining feature extraction for trading and price prediction within one policy.
- **TensorBoard Logging**: Detailed metrics for both trading performance and price-prediction loss.
- **Discrete Action Space**: Eliminates dimensional mismatches and simplifies policy training.
- **Genetic Evolution of Architecture**: Allows the system to discover effective architectures automatically.

---

## Roadmap / Future Work
- **Multi-Discrete Action Support**: Potentially reintroduce multi-discrete actions (e.g., size of Buy/Sell).
- **Full Exchange Emulation**: Expand environment to handle partial fills, slippage, and more realistic trading conditions.
- **Advanced Indicators**: Incorporate additional indicators and external signals (e.g., sentiment analysis).
- **Distributed Training**: Parallelize GA population evaluation for faster experimentation.

---

## License
This project is licensed under the **GRU Public License**.

---

## Author / Contact
- **Alex Kilpatrick**  
  - GitHub: [@alexkilpatrickofficial](https://github.com/alexkilpatrickofficial)  
  - Email: [alexkilpatrick@proton.me](mailto:alexkilpatrick@proton.me)

Feel free to open an issue on GitHub if you encounter any problems or have suggestions. Enjoy experimenting with this Hybrid PPO + GA approach for crypto trading!
