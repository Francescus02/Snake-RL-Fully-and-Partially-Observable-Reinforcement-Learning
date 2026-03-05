# Snake RL: Fully and Partially Observable Reinforcement Learning

This project implements Reinforcement Learning (RL) agents to play the game of Snake in two distinct scenarios: **Fully Observable** (the agent sees the entire 7×7 board) and **Partially Observable** (the agent sees only a 3×3 field of view around its head). Both implementations use Proximal Policy Optimization (PPO) and Actor-Critic architectures.

## Prerequisites

* **Python Version**: This code was developed and tested using **Python 3.12**. Using this specific version is recommended for compatibility with the specific library versions used (TensorFlow, NumPy, etc.).
* **Libraries**:
    * `tensorflow`
    * `numpy`
    * `matplotlib`
    * `tqdm` (for progress bars)

## Project Structure

The project is divided into two main directories, each containing its own environment, model, training, and evaluation scripts.

### 1. Fully Observable Folder
In this setting, the agent has complete knowledge of the board state.
* `environments_fully_observable.py`: Contains the logic for the 7×7 Snake board where the entire grid is passed as input to the agent.
* `snake_model.py`: Implements a CNN-based Actor-Critic network. It uses three convolutional layers to process the spatial features of the board.
* `train.py`: The PPO training script. It includes 4-fold rotation data augmentation to improve sample efficiency.
* `baseline.py`: A rule-based tactical agent used as a performance benchmark. It uses Breadth-First Search (BFS) for pathfinding and a conditional tail-biting strategy to avoid getting trapped.
* `evaluate.py`: Script to run evaluation simulations, load saved weights, and generate comparative performance plots (`comparison_results.png` and `training_progress.png`).

### 2. Partially Observable Folder
In this setting, the agent only sees a 3×3 window (mask) centered on its head.
* `environments_partially_observable.py`: Implements the environment logic where the state is restricted to a local field of view.
* `snake_model.py`: Implements a Recurrent Actor-Critic network. It combines a CNN for local spatial features with a **GRU (Gated Recurrent Unit)** layer to provide temporal memory, allowing the agent to "remember" the board state outside its current view.
* `train_exploration.py`: Advanced training script utilizing:
    * **Curriculum Learning**: Transitions the agent from a 5×5 board to a 7×7 board as performance improves.
    * **Reward Shaping**: Uses Manhattan distance to the fruit to provide denser feedback.
    * **Truncated BPTT**: Backpropagation through time for the recurrent GRU layer.
* `baseline_exploration.py`: A heuristic agent for the PO setting. It uses a "HUNT" mode (when fruit is visible) and a "PATROL" mode (systematic exploration).
* `evaluate_exploration.py`: Evaluation script specifically designed to handle the recurrent hidden states of the PO agents and generate progress visualizations (`comparison_results_po.png` and `training_progress_po.png`).

## How to Run the Code

All scripts should be executed from within their respective folders.

### Training an Agent
To start the training process from scratch:
* **Fully Observable**:
    ```bash
    cd "Fully Observable"
    python train.py
    ```
* **Partially Observable**:
    ```bash
    cd "Partially Observable"
    python train_exploration.py
    ```

### Evaluating and Generating Plots
To evaluate the trained models (saved as `.weights.h5` files) against the baseline:
* **Fully Observable**:
    ```bash
    cd "Fully Observable"
    python evaluate.py
    ```
* **Partially Observable**:
    ```bash
    cd "Partially Observable"
    python evaluate_exploration.py
    ```

### Running the Baselines
To run a standalone simulation of the rule-based heuristic agents:
* **Fully Observable**: `python baseline.py`
* **Partially Observable**: `python baseline_exploration.py`

## Files Summary
* `snake_weights_*.weights.h5`: Saved neural network weights at different training stages (best, final, and periodic snapshots).
* `training_history.json`: Logs of rewards and losses used for plotting training progress.