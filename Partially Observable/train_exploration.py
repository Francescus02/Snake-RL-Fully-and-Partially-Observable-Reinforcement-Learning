"""
train_exploration.py — PPO Training for Partially Observable Snake

This module implements Proximal Policy Optimization (PPO) with multiple enhancements:
- Recurrent policy (GRU) for temporal memory in partially observable environments
- Symmetric data augmentation (horizontal/vertical flips) at rollout time  
- Reward shaping based on Manhattan distance to fruit
- Curriculum learning (5×5 board → 7×7 board)
- Linear scheduling of learning rate and entropy coefficient

Training Algorithm: PPO with Truncated BPTT
--------------------------------------------
1. Collect N_STEPS of experience across parallel environments
2. Maintain GRU hidden states persistently across rollouts
3. Compute advantages using Generalized Advantage Estimation (GAE)
4. Update policy for K_EPOCHS on the collected batch with PPO clipping
5. Mini-batches are sliced by board (not timestep) for proper BPTT
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import json
from collections import deque
from tqdm import tqdm
from environments_partially_observable import OriginalSnakeEnvironment, BaseEnvironment
from snake_model import ActorCriticModel, GRU_UNITS


# ═══════════════════════════════════════════════════════════════════════
#  Hyperparameters
# ═══════════════════════════════════════════════════════════════════════

# Environment
NUM_BOARDS      = 500
MASK_SIZE       = 1            # Field of view: 3×3 window

# Training schedule
TOTAL_UPDATES   = 7500         # Total number of policy updates
N_STEPS         = 25           # Rollout length per update
K_EPOCHS        = 4            # Number of epochs to reuse each batch (PPO)
MINI_BATCH_BOARDS = 50         # Boards per mini-batch
assert NUM_BOARDS % MINI_BATCH_BOARDS == 0, \
    "NUM_BOARDS must be divisible by MINI_BATCH_BOARDS"

# PPO clipping
CLIP_RANGE      = 0.2          # Clamp probability ratio to [0.8, 1.2]

# Optimization
LR_START        = 2e-4         # Initial learning rate
LR_END          = 5e-5         # Final learning rate (linear decay)
GAMMA           = 0.99         # Discount factor
GAE_LAMBDA      = 0.95         # GAE λ parameter
MAX_GRAD_NORM   = 0.5          # Gradient clipping threshold

# Entropy regularization
ENT_START       = 0.04         # High initial entropy for exploration
ENT_END         = 0.005        # Low final entropy for exploitation

# Reward shaping
SHAPE_ALPHA     = 0.005        # Weight for Manhattan distance reward component
                               # Max shaping ≈ 0.04 per step (small vs FRUIT_REWARD=0.5)

# Curriculum learning
CURRICULUM_BOARD_SIZES  = [5, 7]    # Progress from 5×5 to 7×7
CURRICULUM_THRESHOLD    = 0.10      # Reward threshold to advance curriculum
CURRICULUM_WINDOW       = 150       # Window size for rolling average

# Logging and checkpointing
LOG_INTERVAL    = 50
EVAL_INTERVAL   = 500


# ═══════════════════════════════════════════════════════════════════════
#  Data Augmentation
# ═══════════════════════════════════════════════════════════════════════

# Action mapping for spatial transformations
# Actions: UP=0, RIGHT=1, DOWN=2, LEFT=3
HFLIP_ACTION = np.array([0, 3, 2, 1], dtype=np.int32)  # Horizontal flip
VFLIP_ACTION = np.array([2, 1, 0, 3], dtype=np.int32)  # Vertical flip


def augment_observations(obs, actions, aug_types):
    """
    Apply spatial augmentation to observations and transform action labels accordingly.
    
    Augmentation is applied at rollout time to ensure consistency between the
    observation, action, and log-probability stored for the PPO update.
    
    Args:
        obs: Observations (B, H, W, C)
        actions: Action indices (B,)
        aug_types: Augmentation type per board (B,) — 0:none, 1:hflip, 2:vflip, 3:both
        
    Returns:
        aug_obs: Augmented observations (B, H, W, C)
        aug_actions: Transformed action indices (B,)
    """
    aug_obs     = obs.copy()
    aug_actions = actions.copy()

    hflip_mask = (aug_types == 1) | (aug_types == 3)
    vflip_mask = (aug_types == 2) | (aug_types == 3)

    # Horizontal flip: reverse columns
    if np.any(hflip_mask):
        aug_obs[hflip_mask]     = aug_obs[hflip_mask, :, ::-1, :]
        aug_actions[hflip_mask] = HFLIP_ACTION[aug_actions[hflip_mask]]

    # Vertical flip: reverse rows
    if np.any(vflip_mask):
        aug_obs[vflip_mask]     = aug_obs[vflip_mask, ::-1, :, :]
        aug_actions[vflip_mask] = VFLIP_ACTION[aug_actions[vflip_mask]]

    return aug_obs, aug_actions


def inverse_augment_actions(actions, aug_types):
    """
    Convert actions from augmented frame back to environment frame.
    
    Since flip maps are self-inverse, this applies the same transformation.
    
    Args:
        actions: Action indices in augmented frame (B,)
        aug_types: Augmentation type per board (B,)
        
    Returns:
        env_actions: Action indices in environment frame (B,)
    """
    env_actions = actions.copy()
    env_actions[(aug_types == 1) | (aug_types == 3)] = \
        HFLIP_ACTION[env_actions[(aug_types == 1) | (aug_types == 3)]]
    env_actions[(aug_types == 2) | (aug_types == 3)] = \
        VFLIP_ACTION[env_actions[(aug_types == 2) | (aug_types == 3)]]
    return env_actions


# ═══════════════════════════════════════════════════════════════════════
#  Reward Shaping
# ═══════════════════════════════════════════════════════════════════════

def compute_distances(boards):
    """
    Calculate Manhattan distance from snake head to fruit for each board.
    
    Args:
        boards: Environment boards (N, H, W)
        
    Returns:
        distances: Manhattan distances (N,)
    """
    n = boards.shape[0]
    heads  = np.zeros((n, 2), dtype=float)
    fruits = np.zeros((n, 2), dtype=float)
    
    for i in range(n):
        h = np.argwhere(boards[i] == BaseEnvironment.HEAD)
        f = np.argwhere(boards[i] == BaseEnvironment.FRUIT)
        if len(h) > 0:
            heads[i]  = h[0]
        if len(f) > 0:
            fruits[i] = f[0]
            
    return np.abs(heads - fruits).sum(axis=1)


# ═══════════════════════════════════════════════════════════════════════
#  Utility Functions
# ═══════════════════════════════════════════════════════════════════════

def linear_schedule(start: float, end: float, progress: float) -> float:
    """Linearly interpolate between start and end based on progress [0, 1]."""
    return start + (end - start) * progress


def explained_variance(v_pred: np.ndarray, v_true: np.ndarray) -> float:
    """
    Calculate fraction of variance in v_true explained by v_pred.
    
    Returns value near 1.0 if critic is well-calibrated, near 0 otherwise.
    """
    return float(1.0 - np.var(v_true - v_pred) / (np.var(v_true) + 1e-8))


# ═══════════════════════════════════════════════════════════════════════
#  PPO Update Step
# ═══════════════════════════════════════════════════════════════════════

@tf.function
def recurrent_ppo_train_step(model, optimizer,
                              obs_seq, init_hidden,
                              actions_seq, returns_seq,
                              advantages_seq, old_log_probs_seq,
                              old_values_seq,
                              clip_range, ent_coef):
    """
    Perform one PPO gradient update on a mini-batch of sequences.
    
    This function processes complete rollout sequences (B, T, ...) starting from
    stored initial hidden states, enabling truncated BPTT through all N_STEPS.
    
    Args:
        model: ActorCriticModel instance
        optimizer: TensorFlow optimizer
        obs_seq: Observation sequences (B, T, H, W, C)
        init_hidden: Initial GRU states (B, GRU_UNITS)
        actions_seq: Action sequences (B, T)
        returns_seq: Return targets (B, T)
        advantages_seq: Normalized advantages (B, T)
        old_log_probs_seq: Log probabilities from rollout (B, T)
        old_values_seq: Value estimates from rollout (B, T)
        clip_range: PPO clipping parameter
        ent_coef: Entropy coefficient
        
    Returns:
        total_loss, policy_loss, value_loss, entropy (scalars)
    """
    with tf.GradientTape() as tape:
        # Forward pass through full sequences
        logits_seq, values_seq, _ = model.call_sequence(obs_seq,
                                                         init_hidden,
                                                         training=True)
        
        # Flatten batch and time dimensions
        B  = tf.shape(obs_seq)[0]
        T  = tf.shape(obs_seq)[1]
        BT = B * T

        logits_flat   = tf.reshape(logits_seq,     (BT, -1))
        values_flat   = tf.reshape(values_seq,     (BT,))
        actions_flat  = tf.reshape(actions_seq,    (BT,))
        returns_flat  = tf.reshape(returns_seq,    (BT,))
        adv_flat      = tf.reshape(advantages_seq, (BT,))
        old_lp_flat   = tf.reshape(old_log_probs_seq, (BT,))
        old_val_flat  = tf.reshape(old_values_seq,    (BT,))

        # Policy loss: Clipped surrogate objective
        probs   = tf.nn.softmax(logits_flat)
        indices = tf.stack(
            [tf.range(BT), tf.cast(actions_flat, tf.int32)], axis=1
        )
        selected_probs  = tf.gather_nd(probs, indices)
        new_log_probs   = tf.math.log(selected_probs + 1e-10)

        ratio = tf.exp(new_log_probs - old_lp_flat)  # π_new / π_old
        surr1 = ratio * adv_flat
        surr2 = tf.clip_by_value(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv_flat
        policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

        # Value loss: Clipped value function
        v_clipped   = old_val_flat + tf.clip_by_value(
            values_flat - old_val_flat, -clip_range, clip_range
        )
        value_loss  = 0.5 * tf.reduce_mean(
            tf.maximum(tf.square(values_flat - returns_flat),
                       tf.square(v_clipped  - returns_flat))
        )

        # Entropy bonus
        entropy = -tf.reduce_mean(
            tf.reduce_sum(probs * tf.math.log(probs + 1e-10), axis=1)
        )

        total_loss = policy_loss + value_loss - ent_coef * entropy

    # Compute and apply gradients
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return total_loss, policy_loss, value_loss, entropy


# ═══════════════════════════════════════════════════════════════════════
#  Main Training Loop
# ═══════════════════════════════════════════════════════════════════════

def train():
    """Execute the full training procedure with curriculum and checkpointing."""
    
    tf.random.set_seed(0)
    np.random.seed(0)

    print("=" * 65)
    print(" Recurrent PPO — Partially Observable Snake")
    print("=" * 65)

    # Initialize curriculum
    curr_phase      = 0
    board_size      = CURRICULUM_BOARD_SIZES[curr_phase]
    reward_window   = deque(maxlen=CURRICULUM_WINDOW)

    env = OriginalSnakeEnvironment(NUM_BOARDS, board_size, MASK_SIZE)

    print(f"  Curriculum phase 0 — board_size={board_size}")
    print(f"  N_STEPS={N_STEPS} | K_EPOCHS={K_EPOCHS} | "
          f"MINI_BATCH_BOARDS={MINI_BATCH_BOARDS}")
    print(f"  Total updates={TOTAL_UPDATES}\n")

    # Initialize model and optimizer
    model     = ActorCriticModel(num_actions=4)
    input_dim = 2 * MASK_SIZE + 1
    model(tf.zeros((1, input_dim, input_dim, 4)))  # Build model

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LR_START,
        clipnorm=MAX_GRAD_NORM
    )

    # GRU hidden states (maintained across rollouts)
    hidden_states = np.zeros((NUM_BOARDS, GRU_UNITS), dtype=np.float32)

    # Training history
    history = {
        "step": [], "avg_reward": [], "explained_var": [],
        "entropy": [], "policy_loss": [], "value_loss": [],
        "curriculum_phase": []
    }
    best_avg_reward = -float('inf')

    current_state_np = env.to_state()

    # Main training loop
    for update in tqdm(range(TOTAL_UPDATES)):

        # Schedule hyperparameters
        progress    = update / max(TOTAL_UPDATES - 1, 1)
        current_lr  = linear_schedule(LR_START, LR_END, progress)
        current_ent = linear_schedule(ENT_START, ENT_END, progress)
        optimizer.learning_rate.assign(current_lr)

        # ═════════════════════════════════════════════════════════════
        # 1. ROLLOUT COLLECTION
        # ═════════════════════════════════════════════════════════════
        
        # Save initial hidden states for BPTT
        rollout_initial_hidden = hidden_states.copy()

        # Assign random augmentation type per board (fixed for entire rollout)
        aug_types = np.random.randint(0, 4, size=NUM_BOARDS)

        # Storage for rollout data
        mb_obs       = np.zeros((N_STEPS, NUM_BOARDS,
                                 input_dim, input_dim, 4), dtype=np.float32)
        mb_actions   = np.zeros((N_STEPS, NUM_BOARDS), dtype=np.int32)
        mb_rewards   = np.zeros((N_STEPS, NUM_BOARDS), dtype=np.float32)
        mb_values    = np.zeros((N_STEPS, NUM_BOARDS), dtype=np.float32)
        mb_log_probs = np.zeros((N_STEPS, NUM_BOARDS), dtype=np.float32)

        for t in range(N_STEPS):
            # Augment observation
            dummy_actions = np.zeros(NUM_BOARDS, dtype=np.int32)
            aug_obs, _ = augment_observations(current_state_np,
                                              dummy_actions, aug_types)

            # Model forward pass in augmented frame
            obs_tensor   = tf.convert_to_tensor(aug_obs, dtype=tf.float32)
            hidden_tensor = tf.convert_to_tensor(hidden_states, dtype=tf.float32)
            logits, val, new_hidden = model(obs_tensor,
                                            hidden_state=hidden_tensor,
                                            training=False)

            # Sample action in augmented frame
            action_aug = tf.random.categorical(logits, 1)
            action_aug_np = tf.squeeze(action_aug, axis=1).numpy()

            # Compute log-probability
            probs        = tf.nn.softmax(logits)
            indices      = tf.stack(
                [tf.range(NUM_BOARDS), tf.cast(action_aug_np, tf.int32)], axis=1
            )
            log_probs_np = tf.math.log(
                tf.gather_nd(probs, indices) + 1e-10
            ).numpy()

            # Convert action to environment frame
            env_actions = inverse_augment_actions(action_aug_np, aug_types)

            # Reward shaping: record distances before step
            prev_dists = compute_distances(env.boards)

            # Environment step
            rewards_tensor = env.move(env_actions.reshape(NUM_BOARDS, 1))
            rewards_np     = tf.squeeze(rewards_tensor).numpy()

            # Apply distance-based shaping reward (skip when fruit eaten)
            curr_dists       = compute_distances(env.boards)
            fruit_eaten_mask = (rewards_np == env.FRUIT_REWARD)
            shape_reward     = SHAPE_ALPHA * (prev_dists - curr_dists)
            shape_reward[fruit_eaten_mask] = 0.0
            rewards_np = rewards_np + shape_reward.astype(np.float32)

            # Store transition
            mb_obs[t]       = aug_obs
            mb_actions[t]   = action_aug_np
            mb_rewards[t]   = rewards_np
            mb_values[t]    = tf.squeeze(val).numpy()
            mb_log_probs[t] = log_probs_np

            # Update hidden states and environment state
            hidden_states    = new_hidden.numpy()
            current_state_np = env.to_state()

        # ═════════════════════════════════════════════════════════════
        # 2. COMPUTE RETURNS AND ADVANTAGES (GAE)
        # ═════════════════════════════════════════════════════════════
        
        aug_next_obs, _ = augment_observations(current_state_np,
                                               np.zeros(NUM_BOARDS, dtype=np.int32),
                                               aug_types)
        next_obs_tensor    = tf.convert_to_tensor(aug_next_obs, dtype=tf.float32)
        hidden_tensor      = tf.convert_to_tensor(hidden_states, dtype=tf.float32)
        _, last_val_tensor, _ = model(next_obs_tensor,
                                      hidden_state=hidden_tensor,
                                      training=False)
        last_values = tf.squeeze(last_val_tensor).numpy()

        # Generalized Advantage Estimation
        advantages   = np.zeros_like(mb_rewards)
        last_gae_lam = 0.0
        for t in reversed(range(N_STEPS)):
            next_val    = last_values if t == N_STEPS - 1 else mb_values[t + 1]
            delta       = mb_rewards[t] + GAMMA * next_val - mb_values[t]
            advantages[t] = last_gae_lam = delta + GAMMA * GAE_LAMBDA * last_gae_lam
        returns = advantages + mb_values

        # ═════════════════════════════════════════════════════════════
        # 3. REORGANIZE FOR SEQUENCE-BASED MINI-BATCHES
        # ═════════════════════════════════════════════════════════════
        
        # Transpose from (T, B, ...) to (B, T, ...)
        seq_obs       = np.transpose(mb_obs,       (1, 0, 2, 3, 4))
        seq_actions   = np.transpose(mb_actions,   (1, 0))
        seq_returns   = np.transpose(returns,      (1, 0))
        seq_values    = np.transpose(mb_values,    (1, 0))
        seq_log_probs = np.transpose(mb_log_probs, (1, 0))
        seq_adv       = np.transpose(advantages,   (1, 0))

        # ═════════════════════════════════════════════════════════════
        # 4. PPO UPDATE (K_EPOCHS)
        # ═════════════════════════════════════════════════════════════
        
        epoch_losses, epoch_pl, epoch_vl, epoch_ent = [], [], [], []

        for _ in range(K_EPOCHS):
            perm = np.random.permutation(NUM_BOARDS)

            for start in range(0, NUM_BOARDS, MINI_BATCH_BOARDS):
                idx = perm[start: start + MINI_BATCH_BOARDS]

                # Normalize advantages per mini-batch
                mb_adv = seq_adv[idx]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                loss, pl, vl, ent = recurrent_ppo_train_step(
                    model, optimizer,
                    tf.constant(seq_obs[idx],           dtype=tf.float32),
                    tf.constant(rollout_initial_hidden[idx], dtype=tf.float32),
                    tf.constant(seq_actions[idx],       dtype=tf.int32),
                    tf.constant(seq_returns[idx],       dtype=tf.float32),
                    tf.constant(mb_adv,                 dtype=tf.float32),
                    tf.constant(seq_log_probs[idx],     dtype=tf.float32),
                    tf.constant(seq_values[idx],        dtype=tf.float32),
                    tf.constant(CLIP_RANGE,             dtype=tf.float32),
                    tf.constant(current_ent,            dtype=tf.float32),
                )
                epoch_losses.append(loss.numpy())
                epoch_pl.append(pl.numpy())
                epoch_vl.append(vl.numpy())
                epoch_ent.append(ent.numpy())

        # ═════════════════════════════════════════════════════════════
        # 5. CURRICULUM CHECK
        # ═════════════════════════════════════════════════════════════
        
        current_avg_reward = float(np.mean(mb_rewards))
        reward_window.append(current_avg_reward)

        # Flag to prevent saving stale phase-0 scores as best in phase-1
        just_advanced = False

        if (curr_phase < len(CURRICULUM_BOARD_SIZES) - 1 and
                len(reward_window) == CURRICULUM_WINDOW and
                np.mean(reward_window) >= CURRICULUM_THRESHOLD):

            curr_phase += 1
            board_size  = CURRICULUM_BOARD_SIZES[curr_phase]
            tqdm.write(f"\n>>> Curriculum advance! Phase {curr_phase} → "
                       f"board_size={board_size} "
                       f"(rolling_avg={np.mean(reward_window):.4f})\n")

            # Save best weights from completed phase
            model.save_weights(
                f"snake_po_weights_best_phase{curr_phase - 1}.weights.h5"
            )
            tqdm.write(
                f"    Saved phase-{curr_phase - 1} best "
                f"(score={best_avg_reward:.4f}) → "
                f"snake_po_weights_best_phase{curr_phase - 1}.weights.h5"
            )

            # Reset best tracker for new phase
            best_avg_reward = -float('inf')
            just_advanced   = True

            # Rebuild environment
            env = OriginalSnakeEnvironment(NUM_BOARDS, board_size, MASK_SIZE)
            current_state_np = env.to_state()

            # Reset hidden states (spatial layout changed)
            hidden_states = np.zeros((NUM_BOARDS, GRU_UNITS), dtype=np.float32)
            reward_window.clear()

        # ═════════════════════════════════════════════════════════════
        # 6. BOOKKEEPING
        # ═════════════════════════════════════════════════════════════
        
        ev = explained_variance(seq_values.flatten(), seq_returns.flatten())

        # Save best weights (skip on curriculum transition to avoid stale scores)
        if not just_advanced and current_avg_reward > best_avg_reward:
            best_avg_reward = current_avg_reward
            model.save_weights("snake_po_weights_best.weights.h5")

        # Periodic snapshots
        if update % EVAL_INTERVAL == 0 and update > 0:
            model.save_weights(f"snake_po_weights_step{update}.weights.h5")

        # Console logging
        if update % LOG_INTERVAL == 0:
            tqdm.write(
                f"[{update:>5}] "
                f"Rew={current_avg_reward:+.4f}  Best={best_avg_reward:+.4f}  "
                f"PL={np.mean(epoch_pl):.4f}  VL={np.mean(epoch_vl):.4f}  "
                f"Ent={np.mean(epoch_ent):.4f}  EV={ev:.3f}  "
                f"LR={current_lr:.2e}  EntC={current_ent:.4f}  "
                f"Board={board_size}"
            )
            history["step"].append(update)
            history["avg_reward"].append(current_avg_reward)
            history["explained_var"].append(float(ev))
            history["entropy"].append(float(np.mean(epoch_ent)))
            history["policy_loss"].append(float(np.mean(epoch_pl)))
            history["value_loss"].append(float(np.mean(epoch_vl)))
            history["curriculum_phase"].append(curr_phase)

    # ══════════════════════════════════════════════════════════════════
    # 7. FINALIZE
    # ══════════════════════════════════════════════════════════════════
    
    model.save_weights("snake_po_weights_final.weights.h5")

    with open("training_history_po.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete.  Best average reward: {best_avg_reward:.4f}")


if __name__ == "__main__":
    train()