"""
evaluate.py — Evaluation and Visualization for Fully Observable Snake

This module evaluates trained Snake agents and generates comparative visualizations:
1. Baseline agent (tactical heuristic with conditional tail-biting)
2. Best trained weights (highest reward during training)
3. Final trained weights (end of training)
4. Periodic snapshots saved during training (every EVAL_INTERVAL updates)

All evaluations run on the standard 7×7 board with full observability.

Visualization Outputs:
- comparison_results.png: Performance curves for all evaluated agents with inset zoom
- training_progress.png: Training curve with high-degree polynomial trend fit
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json
import glob
import re
from tqdm import tqdm
from environments_fully_observable import OriginalSnakeEnvironment
from snake_model import ActorCriticModel
from baseline import TacticalAgent


# Evaluation Configuration
NUM_EVAL_BOARDS = 500
EVAL_STEPS      = 1000
BOARD_SIZE      = 7


def run_simulation(agent_type, agent_obj=None, weight_path=None):
    """
    Run a complete evaluation simulation.
    
    Args:
        agent_type: "RL" or "Baseline"
        agent_obj: TacticalAgent instance (for Baseline)
        weight_path: Path to .weights.h5 file (for RL)
        
    Returns:
        final_avg_reward: Mean cumulative reward across all boards
        history: List of cumulative average rewards at each step
    """
    env = OriginalSnakeEnvironment(NUM_EVAL_BOARDS, BOARD_SIZE)
    model = None
    
    if agent_type == "RL":
        # Load RL agent
        model = ActorCriticModel(num_actions=4)
        model(tf.zeros((1, BOARD_SIZE, BOARD_SIZE, 4)))  # Build model
        
        try:
            model.load_weights(weight_path)
            filename = os.path.basename(weight_path)
            print(f"  Loaded: {filename}")
        except Exception as e:
            print(f"  ERROR loading {weight_path}: {e}")
            return None, None

    cumulative_rewards = np.zeros(NUM_EVAL_BOARDS)
    history = []
    
    desc = f"Eval {agent_type if agent_type != 'RL' else os.path.basename(weight_path)}"
    for _ in tqdm(range(EVAL_STEPS), desc=desc, leave=False):
        
        if agent_type == "RL":
            # RL agent inference (no augmentation during evaluation)
            state = env.to_state()
            state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
            logits, _ = model(state_tensor, training=False)
            
            # Greedy action selection
            action = tf.argmax(logits, axis=1)
            action_env = tf.reshape(action, [-1, 1]).numpy()
        else:
            # Baseline agent uses raw board state
            action_env = agent_obj.predict(env.boards)
        
        # Environment step
        rewards = env.move(action_env)
        
        # Track cumulative rewards
        cumulative_rewards += tf.squeeze(rewards).numpy()
        history.append(np.mean(cumulative_rewards))
        
    return np.mean(cumulative_rewards), history


def plot_training_history(baseline_score=None):
    """
    Plot training progress with high-degree polynomial trend fit.
    
    Reads training_history.json and generates visualization showing
    reward progression over training with smooth fitted curve.
    
    Outputs:
        training_progress.png
    """
    history_path = "training_history.json"
    if not os.path.exists(history_path):
        print(f"\nWarning: {history_path} not found. Skipping training graph.")
        return

    try:
        with open(history_path, "r") as f:
            data = json.load(f)
        steps = np.array(data["step"])
        rewards = np.array(data["avg_reward"])
    except Exception as e:
        print(f"Error reading history file: {e}")
        return

    plt.figure(figsize=(12, 6))
    
    # Raw training data as scatter points
    plt.scatter(steps, rewards, color='steelblue', s=20, alpha=0.4,
                label='Average reward per update')
    
    # High-degree polynomial trend fit for smooth visualization
    if len(steps) > 10:
        degree = min(21, len(steps) - 1)  # Degree 21 for smoother fit
        try:
            z = np.polyfit(steps, rewards, degree)
            p = np.poly1d(z)
            s_smooth = np.linspace(steps.min(), steps.max(), 500)
            plt.plot(s_smooth, p(s_smooth), color='red', linewidth=2.5,
                     label=f'Trend (poly deg-{degree})', alpha=0.9)
        except np.linalg.LinAlgError:
            pass  # Skip if fitting fails

    if baseline_score is not None:
        plt.axhline(y=baseline_score/EVAL_STEPS, color='gray', linestyle='--', linewidth=2, label=f'Baseline ({(baseline_score/EVAL_STEPS):.4f})')

    plt.xlabel('Update step', fontsize=12)
    plt.ylabel('Average reward per step', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig("training_progress.png", dpi=120, bbox_inches='tight')
    print("Training graph saved as 'training_progress.png'")
    plt.show()


def main():
    """
    Main evaluation routine.
    
    Evaluates all available agents and generates comparison plot plus
    training history visualization.
    """
    tf.random.set_seed(0)
    np.random.seed(0)

    results = {}

    # 1. Baseline Agent
    print("\n=== Baseline Tactical Agent ===")
    baseline_agent = TacticalAgent(NUM_EVAL_BOARDS)
    base_score, base_history = run_simulation("Baseline", agent_obj=baseline_agent)
    results["Baseline"] = (base_score, base_history)
    print(f"  Score: {base_score:.2f}")

    # 2. Best Weights
    print("\n=== Best Weights ===")
    best_score, best_history = run_simulation(
        "RL", weight_path="snake_weights_best.weights.h5"
    )
    if best_score is not None:
        results["Best"] = (best_score, best_history)
        print(f"  Score: {best_score:.2f}")

    # 3. Final Weights
    print("\n=== Final Weights ===")
    final_score, final_history = run_simulation(
        "RL", weight_path="snake_weights_final.weights.h5"
    )
    if final_score is not None:
        results["Final"] = (final_score, final_history)
        print(f"  Score: {final_score:.2f}")

    # 4. Periodic Snapshots
    print("\n=== Periodic Snapshots ===")
    snapshot_files = sorted(glob.glob("snake_weights_step*.weights.h5"))
    
    # Extract step numbers and sort numerically
    snapshot_data = []
    for fpath in snapshot_files:
        match = re.search(r'step(\d+)', fpath)
        if match:
            step_num = int(match.group(1))
            snapshot_data.append((step_num, fpath))
    
    snapshot_data.sort()
    
    for step_num, fpath in snapshot_data:
        score, hist = run_simulation("RL", weight_path=fpath)
        if score is not None:
            results[f"Step{step_num}"] = (score, hist)
            print(f"  Step {step_num}: {score:.2f}")

    # Generate comparison plot with inset zoom
    print("\n=== Generating Comparison Plot ===")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Color and style configuration
    colors = {
        "Baseline": "gray",
        "Best": "green",
        "Final": "blue"
    }
    styles = {
        "Baseline": "--",
        "Best": "-.",
        "Final": "-"
    }
    
    # Build plot order: Baseline → Final → Best → Snapshots (descending)
    plot_order = []
    for key in ["Baseline", "Final", "Best"]:
        if key in results:
            plot_order.append(key)
    
    # Add snapshot checkpoints in reverse chronological order
    step_keys = sorted([k for k in results.keys() if k.startswith("Step")],
                      key=lambda x: int(x.replace("Step", "")),
                      reverse=True)
    plot_order.extend(step_keys)
    
    # Generate color gradient for snapshots
    n_snapshots = len(step_keys)
    if n_snapshots > 0:
        step_colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_snapshots))
    
    # Plot all agents
    snapshot_idx = 0
    for label in plot_order:
        score, hist = results[label]
        if score is not None:
            if label.startswith("Step"):
                color = step_colors[snapshot_idx]
                style = ':'
                snapshot_idx += 1
                linewidth = 1.2
            else:
                color = colors.get(label, "black")
                style = styles.get(label, "-")
                linewidth = 2.0
            
            ax.plot(hist,
                    label=f"{label} ({score:.2f})",
                    color=color,
                    linestyle=style,
                    linewidth=linewidth,
                    alpha=0.85)

    ax.set_xlabel('Evaluation Step', fontsize=12)
    ax.set_ylabel('Cumulative Average Reward', fontsize=12)
    
    # Enhanced legend with white background for better visibility
    ax.legend(fontsize=18, ncol=2, framealpha=1.0, loc='upper left',
              fancybox=True, shadow=True, edgecolor='black')
    ax.grid(True, alpha=0.3)
    
    # Add inset zoom showing final 4% of evaluation
    zoom_start = int(EVAL_STEPS * 0.96)  # Last 4% of steps
    axins = ax.inset_axes([0.55, 0.075, 0.42, 0.35])  # [x, y, width, height] in figure coords
    
    for label in plot_order:
        score, hist = results[label]
        if score is not None:
            if label.startswith("Step"):
                idx = step_keys.index(label)
                color = step_colors[idx]
                style = ':'
                linewidth = 1.5
            else:
                color = colors.get(label, "black")
                style = styles.get(label, "-")
                linewidth = 2.5
            
            axins.plot(range(zoom_start, EVAL_STEPS),
                      hist[zoom_start:],
                      color=color,
                      linestyle=style,
                      linewidth=linewidth,
                      alpha=0.9)
    
    axins.set_xlim(zoom_start, EVAL_STEPS)
    axins.set_xlabel('Step', fontsize=9)
    axins.set_ylabel('Reward', fontsize=9)
    axins.set_title('Final Phase (Close-up)', fontsize=10)
    axins.grid(True, alpha=0.3)
    axins.tick_params(labelsize=8)
    
    # Draw rectangle on main plot showing zoomed region
    ax.add_patch(Rectangle((zoom_start, ax.get_ylim()[0]),
                           EVAL_STEPS - zoom_start,
                           ax.get_ylim()[1] - ax.get_ylim()[0],
                           fill=False, edgecolor='red', linewidth=2,
                           linestyle='--', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig("comparison_results.png", dpi=120, bbox_inches='tight')
    print("Comparison graph saved as 'comparison_results.png'")
    plt.show()

    # Plot training history
    plot_training_history(base_score)


if __name__ == "__main__":
    main()