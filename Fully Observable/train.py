"""
train.py — PPO Training for Fully Observable Snake

This module implements Proximal Policy Optimization (PPO) for training a Snake
agent on a 7×7 board with full observability. The agent learns through parallel
environment interaction and policy gradient optimization.

Training Algorithm: PPO
-----------------------
PPO balances sample efficiency with training stability by:
- Reusing collected experience multiple times (K_EPOCHS epochs)
- Clipping probability ratios to prevent destructive policy updates
- Maintaining separate value function with clipped updates

Key Features:
- Parallel environments (500 boards) for efficient data collection
- 4-fold rotation augmentation for data diversity
- Generalized Advantage Estimation (GAE) for variance reduction
- Linear scheduling of learning rate and entropy coefficient
- Mini-batch updates for stable gradient estimates
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import json
from tqdm import tqdm
from environments_fully_observable import OriginalSnakeEnvironment
from snake_model import ActorCriticModel


# ═══════════════════════════════════════════════════════════════════════
#  Hyperparameters
# ═══════════════════════════════════════════════════════════════════════

# Environment configuration
NUM_BOARDS      = 500          # Number of parallel game boards
BOARD_SIZE      = 7            # Board dimensions (7×7 with wall borders)
USE_RESIDUAL    = False        # Enable residual connections in CNN

# Training schedule
TOTAL_UPDATES   = 5000         # Total number of policy updates
N_STEPS         = 20           # Rollout length per update
K_EPOCHS        = 4            # Number of epochs to reuse each batch
MINI_BATCH_SIZE = 500          # Samples per mini-batch

# PPO clipping parameter
CLIP_RANGE      = 0.2          # Clamp probability ratio to [0.8, 1.2]

# Optimization
LR_START        = 2e-4         # Initial learning rate
LR_END          = 5e-5         # Final learning rate (linear decay)
GAMMA           = 0.99         # Discount factor for future rewards
GAE_LAMBDA      = 0.95         # GAE lambda parameter
MAX_GRAD_NORM   = 0.5          # Gradient clipping threshold

# Entropy regularization
ENT_START       = 0.02         # Initial entropy coefficient (exploration)
ENT_END         = 0.005        # Final entropy coefficient (exploitation)

# Data augmentation
USE_AUGMENTATION = True        # Enable 4-fold rotation augmentation
AUGMENTATION_TYPE = "rotation_only"  # Rotation-based augmentation

# Optional reward shaping
USE_REWARD_SHAPING = False     # Add Manhattan distance reward component
SHAPE_ALPHA = 0.005            # Weight for distance-based shaping

# Logging and checkpointing
LOG_INTERVAL    = 50           # Console logging frequency
EVAL_INTERVAL   = 500          # Weight saving frequency


# ═══════════════════════════════════════════════════════════════════════
#  Rotation Augmentation
# ═══════════════════════════════════════════════════════════════════════

def rotate_board_90_with_walls(board):
    """
    Rotate entire board 90° clockwise including walls.
    
    Walls remain at board edges after rotation, preserving environment structure.
    Handles both single boards (H, W, C) and batched boards (1, H, W, C).
    
    Args:
        board: Board state array
        
    Returns:
        Rotated board with same shape as input
    """
    if board.ndim == 4:
        # Batched: (1, H, W, C) - rotate height and width axes
        return np.rot90(board, k=-1, axes=(1, 2))  # k=-1 for clockwise
    else:
        # Single: (H, W, C) - rotate spatial axes
        return np.rot90(board, k=-1, axes=(0, 1))


# Action transformations under rotation
# Snake actions: UP=0, RIGHT=1, DOWN=2, LEFT=3
ACTION_ROT90 = np.array([3, 0, 1, 2])    # 90° clockwise rotation mapping
ACTION_ROT180 = np.array([2, 3, 0, 1])   # 180° rotation mapping
ACTION_ROT270 = np.array([1, 2, 3, 0])   # 270° clockwise rotation mapping


def apply_augmentation(obs, aug_type):
    """
    Apply rotation augmentation to observation.
    
    Args:
        obs: Observation array (H, W, C) or (1, H, W, C)
        aug_type: Rotation type (0=none, 1=90°, 2=180°, 3=270°)
        
    Returns:
        Augmented observation
    """
    if aug_type == 0:
        return obs.copy()
    
    aug_obs = obs.copy()
    for _ in range(aug_type):
        aug_obs = rotate_board_90_with_walls(aug_obs)
    
    return aug_obs


def transform_actions(actions, aug_type):
    """
    Transform action labels to match augmented observation frame.
    
    Args:
        actions: Action indices (N,)
        aug_type: Rotation type applied to observations
        
    Returns:
        Transformed action indices
    """
    if aug_type == 0:
        return actions
    elif aug_type == 1:
        return ACTION_ROT90[actions]
    elif aug_type == 2:
        return ACTION_ROT180[actions]
    else:  # aug_type == 3
        return ACTION_ROT270[actions]


def inverse_transform_actions(actions, aug_type):
    """
    Transform actions from augmented frame back to environment frame.
    
    Applies inverse rotation to convert actions sampled in rotated frame
    back to original board orientation for environment execution.
    
    Args:
        actions: Actions in augmented frame (N,)
        aug_type: Rotation type that was applied
        
    Returns:
        Actions in environment frame (N,)
    """
    if aug_type == 0:
        return actions
    elif aug_type == 1:
        return ACTION_ROT270[actions]  # Inverse of 90° is 270°
    elif aug_type == 2:
        return ACTION_ROT180[actions]  # 180° is self-inverse
    else:  # aug_type == 3
        return ACTION_ROT90[actions]   # Inverse of 270° is 90°


# ═══════════════════════════════════════════════════════════════════════
#  Optional Reward Shaping
# ═══════════════════════════════════════════════════════════════════════

def compute_distances(boards):
    """
    Calculate Manhattan distance from snake head to fruit.
    
    Used for optional reward shaping to provide dense gradient signal.
    
    Args:
        boards: Raw board arrays (N, H, W) with cell values
        
    Returns:
        distances: Manhattan distances (N,)
    """
    n = boards.shape[0]
    heads = np.zeros((n, 2), dtype=float)
    fruits = np.zeros((n, 2), dtype=float)
    
    HEAD_VAL = 4   # Cell value for snake head
    FRUIT_VAL = 2  # Cell value for fruit
    
    for i in range(n):
        h = np.argwhere(boards[i] == HEAD_VAL)
        f = np.argwhere(boards[i] == FRUIT_VAL)
        if len(h) > 0:
            heads[i] = h[0]
        if len(f) > 0:
            fruits[i] = f[0]
    
    return np.abs(heads - fruits).sum(axis=1)


# ═══════════════════════════════════════════════════════════════════════
#  Utility Functions
# ═══════════════════════════════════════════════════════════════════════

def linear_schedule(start: float, end: float, progress: float) -> float:
    """
    Linearly interpolate between start and end values.
    
    Args:
        start: Initial value
        end: Final value
        progress: Progress in [0, 1]
        
    Returns:
        Interpolated value
    """
    return start + (end - start) * progress


def explained_variance(v_pred: np.ndarray, v_true: np.ndarray) -> float:
    """
    Calculate fraction of variance in true values explained by predictions.
    
    Returns value near 1.0 when predictions are well-calibrated,
    near 0 when they provide no useful information.
    
    Args:
        v_pred: Predicted values
        v_true: True target values
        
    Returns:
        Explained variance ratio
    """
    return float(1.0 - np.var(v_true - v_pred) / (np.var(v_true) + 1e-8))


# ═══════════════════════════════════════════════════════════════════════
#  PPO Update Step
# ═══════════════════════════════════════════════════════════════════════

@tf.function
def ppo_train_step(model, optimizer,
                   states, actions, returns, advantages,
                   old_log_probs, old_values,
                   clip_range, ent_coef):
    """
    Perform one PPO gradient update on a mini-batch.
    
    Computes clipped surrogate objective for policy, clipped value loss for
    critic, and entropy bonus for exploration. Updates model parameters via
    gradient descent.
    
    Args:
        model: ActorCriticModel instance
        optimizer: TensorFlow optimizer
        states: Batch of observations (B, H, W, C)
        actions: Action indices (B,)
        returns: Target returns (B,)
        advantages: Normalized advantage estimates (B,)
        old_log_probs: Log probabilities from rollout (B,)
        old_values: Value estimates from rollout (B,)
        clip_range: PPO clipping epsilon
        ent_coef: Entropy regularization coefficient
        
    Returns:
        total_loss: Combined loss value
        policy_loss: Policy gradient loss
        value_loss: Value function loss
        entropy: Policy entropy
    """
    with tf.GradientTape() as tape:
        logits, values = model(states, training=True)
        values = tf.squeeze(values)
        
        # Policy loss: PPO clipped surrogate objective
        probs = tf.nn.softmax(logits)
        indices = tf.stack(
            [tf.range(tf.shape(actions)[0]), tf.cast(actions, tf.int32)], axis=1
        )
        selected_probs = tf.gather_nd(probs, indices)
        new_log_probs = tf.math.log(selected_probs + 1e-10)
        
        ratio = tf.exp(new_log_probs - old_log_probs)  # π_new / π_old
        surr1 = ratio * advantages
        surr2 = tf.clip_by_value(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
        policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
        
        # Value loss: Clipped value function objective
        v_clipped = old_values + tf.clip_by_value(
            values - old_values, -clip_range, clip_range
        )
        value_loss = 0.5 * tf.reduce_mean(
            tf.maximum(tf.square(values - returns),
                      tf.square(v_clipped - returns))
        )
        
        # Entropy bonus: Encourages exploration
        entropy = -tf.reduce_mean(
            tf.reduce_sum(probs * tf.math.log(probs + 1e-10), axis=1)
        )
        
        total_loss = policy_loss + value_loss - ent_coef * entropy
    
    # Compute and apply gradients with clipping
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return total_loss, policy_loss, value_loss, entropy


# ═══════════════════════════════════════════════════════════════════════
#  Main Training Loop
# ═══════════════════════════════════════════════════════════════════════

def train():
    """
    Execute the full PPO training procedure.
    
    Trains an actor-critic agent on parallel Snake environments using PPO
    algorithm with rotation augmentation and scheduled hyperparameters.
    """
    
    tf.random.set_seed(0)
    np.random.seed(0)

    print("=" * 70)
    print(" PPO Training — Fully Observable Snake")
    print("=" * 70)
    print(f" Augmentation: {'ON (' + AUGMENTATION_TYPE + ')' if USE_AUGMENTATION else 'OFF'}")
    print(f" Reward Shaping: {'ON' if USE_REWARD_SHAPING else 'OFF'}")
    print(f" Board Size: {BOARD_SIZE}×{BOARD_SIZE} (playable: {BOARD_SIZE-2}×{BOARD_SIZE-2})")
    print("=" * 70)

    # Initialize environment and model
    env = OriginalSnakeEnvironment(NUM_BOARDS, BOARD_SIZE)
    model = ActorCriticModel(num_actions=4, use_residual=USE_RESIDUAL)
    
    # Build model with dummy forward pass
    dummy_state = env.to_state()
    model(tf.convert_to_tensor(dummy_state[:1], dtype=tf.float32))
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LR_START,
        clipnorm=MAX_GRAD_NORM
    )

    # Training history for visualization
    history = {
        "step": [], "avg_reward": [], "explained_var": [],
        "entropy": [], "policy_loss": [], "value_loss": []
    }
    best_avg_reward = -float('inf')

    current_state_np = env.to_state()

    # Determine augmentation types
    n_aug_types = 4 if USE_AUGMENTATION else 1  # 0,1,2,3 rotations or identity only

    print(f"\nStarting training... (Updates: {TOTAL_UPDATES})\n")
    
    for update in tqdm(range(TOTAL_UPDATES)):

        # Schedule hyperparameters linearly over training
        progress = update / max(TOTAL_UPDATES - 1, 1)
        current_lr = linear_schedule(LR_START, LR_END, progress)
        current_ent = linear_schedule(ENT_START, ENT_END, progress)
        optimizer.learning_rate.assign(current_lr)

        # ═════════════════════════════════════════════════════════════
        # 1. ROLLOUT COLLECTION
        # ═════════════════════════════════════════════════════════════
        
        # Assign random augmentation type per board (fixed for entire rollout)
        aug_types = np.random.randint(0, n_aug_types, size=NUM_BOARDS)
        
        # Storage for rollout data
        mb_states = []
        mb_actions = []
        mb_rewards = []
        mb_values = []
        mb_log_probs = []

        for t in range(N_STEPS):
            # Apply augmentation if enabled
            if USE_AUGMENTATION:
                aug_states = np.stack([
                    apply_augmentation(current_state_np[i:i+1], aug_types[i])[0]
                    for i in range(NUM_BOARDS)
                ])
            else:
                aug_states = current_state_np.copy()
            
            # Model inference in augmented frame
            state_tensor = tf.convert_to_tensor(aug_states, dtype=tf.float32)
            logits, val = model(state_tensor, training=False)
            
            # Sample actions from policy distribution
            action = tf.random.categorical(logits, 1)
            action_np = tf.squeeze(action, axis=1).numpy()
            
            # Compute log probabilities for sampled actions
            probs = tf.nn.softmax(logits)
            indices = tf.stack(
                [tf.range(NUM_BOARDS), tf.cast(action_np, tf.int32)], axis=1
            )
            log_probs_np = tf.math.log(
                tf.gather_nd(probs, indices) + 1e-10
            ).numpy()
            
            # Transform actions to environment frame
            if USE_AUGMENTATION:
                env_actions = np.array([
                    inverse_transform_actions(action_np[i:i+1], aug_types[i])[0]
                    for i in range(NUM_BOARDS)
                ])
            else:
                env_actions = action_np
            
            # Optional reward shaping: record distances before step
            if USE_REWARD_SHAPING:
                prev_dists = compute_distances(env.boards)
            
            # Environment step
            rewards_tensor = env.move(env_actions.reshape(NUM_BOARDS, 1))
            rewards_np = tf.squeeze(rewards_tensor).numpy()
            
            # Apply distance-based reward shaping if enabled
            if USE_REWARD_SHAPING:
                curr_dists = compute_distances(env.boards)
                # Only apply shaping when fruit wasn't eaten
                fruit_eaten_mask = (rewards_np == 0.5)  # FRUIT_REWARD = 0.5
                shape_reward = SHAPE_ALPHA * (prev_dists - curr_dists)
                shape_reward[fruit_eaten_mask] = 0.0
                rewards_np = rewards_np + shape_reward.astype(np.float32)
            
            # Store transition
            mb_states.append(aug_states)
            mb_actions.append(action_np)
            mb_rewards.append(rewards_np)
            mb_values.append(tf.squeeze(val).numpy())
            mb_log_probs.append(log_probs_np)
            
            # Update current state
            current_state_np = env.to_state()

        # ═════════════════════════════════════════════════════════════
        # 2. COMPUTE RETURNS AND ADVANTAGES
        # ═════════════════════════════════════════════════════════════
        
        # Bootstrap value for final state
        if USE_AUGMENTATION:
            aug_next = np.stack([
                apply_augmentation(current_state_np[i:i+1], aug_types[i])[0]
                for i in range(NUM_BOARDS)
            ])
        else:
            aug_next = current_state_np.copy()
            
        next_tensor = tf.convert_to_tensor(aug_next, dtype=tf.float32)
        _, last_val_tensor = model(next_tensor, training=False)
        last_values = tf.squeeze(last_val_tensor).numpy()
        
        mb_rewards = np.array(mb_rewards)  # (T, B)
        mb_values = np.array(mb_values)    # (T, B)
        
        # Generalized Advantage Estimation (GAE)
        advantages = np.zeros_like(mb_rewards)
        last_gae_lam = 0.0
        
        for t in reversed(range(N_STEPS)):
            next_val = last_values if t == N_STEPS - 1 else mb_values[t + 1]
            delta = mb_rewards[t] + GAMMA * next_val - mb_values[t]
            advantages[t] = last_gae_lam = delta + GAMMA * GAE_LAMBDA * last_gae_lam
        
        returns = advantages + mb_values
        
        # Flatten batch and time dimensions
        flat_states = np.concatenate(mb_states, axis=0)
        flat_actions = np.concatenate(mb_actions, axis=0)
        flat_returns = returns.flatten()
        flat_values = mb_values.flatten()
        flat_log_probs = np.concatenate(mb_log_probs, axis=0)
        flat_advantages = advantages.flatten()

        # ═════════════════════════════════════════════════════════════
        # 3. PPO UPDATE
        # ═════════════════════════════════════════════════════════════
        
        # Track losses across epochs
        epoch_losses, epoch_pl, epoch_vl, epoch_ent = [], [], [], []
        batch_size = N_STEPS * NUM_BOARDS
        
        # Multiple epochs over the same batch (batch reuse)
        for _ in range(K_EPOCHS):
            perm = np.random.permutation(batch_size)
            
            # Process mini-batches
            for start in range(0, batch_size, MINI_BATCH_SIZE):
                idx = perm[start: start + MINI_BATCH_SIZE]
                
                # Normalize advantages per mini-batch
                mb_adv = flat_advantages[idx]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                
                loss, pl, vl, ent = ppo_train_step(
                    model, optimizer,
                    tf.constant(flat_states[idx], dtype=tf.float32),
                    tf.constant(flat_actions[idx], dtype=tf.int32),
                    tf.constant(flat_returns[idx], dtype=tf.float32),
                    tf.constant(mb_adv, dtype=tf.float32),
                    tf.constant(flat_log_probs[idx], dtype=tf.float32),
                    tf.constant(flat_values[idx], dtype=tf.float32),
                    tf.constant(CLIP_RANGE, dtype=tf.float32),
                    tf.constant(current_ent, dtype=tf.float32),
                )
                epoch_losses.append(loss.numpy())
                epoch_pl.append(pl.numpy())
                epoch_vl.append(vl.numpy())
                epoch_ent.append(ent.numpy())

        # ═════════════════════════════════════════════════════════════
        # 4. BOOKKEEPING
        # ═════════════════════════════════════════════════════════════
        
        current_avg_reward = float(np.mean(mb_rewards))
        ev = explained_variance(flat_values, flat_returns)

        # Save best model weights
        if current_avg_reward > best_avg_reward:
            best_avg_reward = current_avg_reward
            model.save_weights("snake_weights_best.weights.h5")

        # Periodic weight snapshots
        if update % EVAL_INTERVAL == 0 and update > 0:
            model.save_weights(f"snake_weights_step{update}.weights.h5")

        # Console logging
        if update % LOG_INTERVAL == 0:
            tqdm.write(
                f"[{update:>5}] "
                f"Rew={current_avg_reward:+.4f}  Best={best_avg_reward:+.4f}  "
                f"PL={np.mean(epoch_pl):.4f}  VL={np.mean(epoch_vl):.4f}  "
                f"Ent={np.mean(epoch_ent):.4f}  EV={ev:.3f}  "
                f"LR={current_lr:.2e}"
            )
            history["step"].append(update)
            history["avg_reward"].append(current_avg_reward)
            history["explained_var"].append(float(ev))
            history["entropy"].append(float(np.mean(epoch_ent)))
            history["policy_loss"].append(float(np.mean(epoch_pl)))
            history["value_loss"].append(float(np.mean(epoch_vl)))

    # ══════════════════════════════════════════════════════════════════
    # 5. FINALIZE
    # ══════════════════════════════════════════════════════════════════
    
    model.save_weights("snake_weights_final.weights.h5")

    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"Best average reward: {best_avg_reward:.4f}")
    print(f"Configuration:")
    print(f"  - PPO: K_EPOCHS={K_EPOCHS}")
    print(f"  - Augmentation: {USE_AUGMENTATION}")
    print(f"  - Reward shaping: {USE_REWARD_SHAPING}")
    print(f"{'='*70}")


if __name__ == "__main__":
    train()