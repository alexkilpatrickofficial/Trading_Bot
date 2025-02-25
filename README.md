Crypto Trading GA with Hybrid PPO
A Genetic Algorithm (GA) project that evolves chromosomes (architectures) for a Hybrid PPO trading model in a cryptocurrency environment. This system combines reinforcement learning (via Stable-Baselines3 PPO) with auxiliary price prediction for improved feature extraction and trading performance.

Table of Contents
Overview
Features
Installation
Usage
Project Structure
Technical Highlights
Roadmap / Future Work
License
Overview
This repository implements a hybrid approach for trading in cryptocurrency markets using:

Genetic Algorithms (GA) to evolve model architectures (defined as chromosomes).
Stable-Baselines3 PPO for on-policy reinforcement learning.
An auxiliary price-prediction head that helps the policy learn better feature representations and maintain learning signals even when trading performance is temporarily poor.
Why a Hybrid Approach?
Traditional RL trading models can struggle if the reward is highly volatile or negative. Adding an auxiliary objective (e.g., predicting price movements) provides additional feedback signals to the model’s feature extractor, improving sample efficiency and potentially leading to more robust policies.

Features
Genetic Algorithm:
Evolves different neural architectures and hyperparameters (chromosomes).
Automatic architecture validation (validate_chromosome).
Modular blocks (CNN, LSTM, TCN, MLP, etc.) combined in a flexible pipeline.
Hybrid PPO:
Uses Stable-Baselines3 for on-policy RL.
Auxiliary MSE Loss for price prediction, combined with the main PPO objective.
Callbacks:
TensorBoard logging for rewards and performance metrics.
Model checkpoints at regular intervals.
Early stopping conditions if consecutive episodes terminate badly.
Custom Environment:
MultiStockTradingEnv with multi-discrete action space ([3, 10] → buy/sell/hold + discrete trade size).
Logging and shape-handling code that ensures correct shaping for (1, 2) vs (2,) actions.
Installation
Clone the Repository:

bash
Copy
Edit
git clone https://github.com/alexkilpatrickofficial/crypto-ga-ppo.git
cd crypto-ga-ppo
Create/Activate a Virtual Environment (optional but recommended):

bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # Linux/Mac
# or
venv\Scripts\activate      # Windows
Install Requirements:

bash
Copy
Edit
pip install -r requirements.txt
Check GPU (optional):

Ensure PyTorch or CUDA is installed properly for GPU acceleration if desired.
Usage
Data Setup:

Place your raw CSV data (e.g., Gemini_BTCUSD_1h.csv) in the default path or update references in main.py.
Run GA + PPO:

bash
Copy
Edit
python main.py
The script:
Preprocesses data (caches it in HDF5, trains a scaler).
Loads or initializes a genetic algorithm population (population).
Evolves them, calling evaluate(...) for each chromosome.
Each evaluate(...) function sets up a single-environment PPO with a HybridPolicy.
TensorBoard:

If launched, you can navigate to http://localhost:6006 to monitor training logs.
Model Checkpoints:

Saved at intervals by the CheckpointCallback. Look under models/ or the path you configured.
Project Structure
A brief explanation of the key modules:

main.py
Orchestrates the entire GA flow: loads data, initializes or resumes the GA state, and evolves chromosomes across generations.

evaluation.py
Contains the evaluate(individual, …) function which:

Validates a chromosome.
Creates a PPO model (HybridPolicy).
Trains it (with optional callbacks).
Evaluates final performance and returns a fitness tuple.
envs/multi_stock_env.py
The custom environment derived from Gymnasium. Implements multi-discrete action logic for buy/sell/hold plus discrete trade fraction, includes reward shaping and logging.

hybrid_policy.py
Custom policy inheriting from Stable-Baselines3’s ActorCriticPolicy:

Adds an auxiliary MSE loss for price prediction.
Merges the PPO loss with the auxiliary head’s MSE.
models.py

DynamicHybridFeatureExtractor: Builds modular blocks (CNN, LSTM, TCN, MLP, etc.) from chromosome specs.
ModularBlock: Single block representing one architecture component.
ga_utils.py
DEAP-based genetic algorithm utilities (crossover, mutation, selection, etc.), plus logging/tensorboard directory helpers.

callbacks.py
Custom callbacks for:

TensorBoard logging.
Checkpointing the PPO model.
Early stopping based on consecutive bad episodes.
requirements.txt
Lists necessary Python dependencies (Stable-Baselines3, PyTorch, DEAP, etc.).

Technical Highlights
Multi-Discrete Action: We handle (3,10) actions (buy/sell/hold + 10 discrete trade sizes). Additional shape-handling code ensures (1,2) or (2,) mismatch is resolved, though SB3’s rollout buffer might need attention if you see shape errors.
Auxiliary Loss: HybridPolicy merges the PPO objective with a price-prediction MSE objective, improving feature extraction consistency.
Evolutionary Architecture Search: The GA mutates and crosses over block types (CNN, LSTM, etc.) for dynamic configurations, validated through simulation in simulate_modular_blocks.
Roadmap / Future Work
Custom Rollout Buffer for Multi-Discrete:
If SB3 shape errors persist, implement a specialized buffer or flatten your action space to a single discrete dimension.

Data Pipeline Enhancements:
Extend data_preprocessing.py to handle additional markets or multi-asset training.

Additional Reward Shaping:
Explore more nuanced reward signals beyond profit and price-prediction accuracy.

Multi-Agent or Multi-Env:
Scale up to multiple parallel environments for faster training.

License
(Include your preferred license statement here, e.g., MIT License or Apache License. If no license is specified, remove or adjust this section.)

Author / Contact
Your Name (@alexkilpatrickofficial)
Email: alexkilpatrick@proton.me
Feel free to open an issue on GitHub if you encounter any problems or have suggestions. Enjoy experimenting with the Hybrid PPO + GA approach for crypto trading!
