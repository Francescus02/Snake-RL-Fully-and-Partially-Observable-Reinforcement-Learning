"""
baseline.py — Rule-Based Baseline Agent for Fully Observable Snake

This module implements a tactical heuristic agent that serves as a performance
baseline for evaluating learned policies. Unlike the partially observable agent,
this agent has access to the complete board state at all times.

Agent Strategy Overview
-----------------------
The agent operates using three decision modes:

1. EFFICIENCY CHECK: Compare path length from BFS (safe path avoiding body)
   with Manhattan distance (direct path). If safe path is significantly longer,
   consider tactical tail-biting.

2. TACTICAL TAIL-BITING: When the safe path is inefficient, evaluate whether
   a 180° turn (biting the tail) would place the head closer to the fruit.
   Only execute if the reverse move improves position.

3. SURVIVAL MODE: If no safe moves exist (trapped), always execute tail-bite
   as the only option to continue the game.

The agent tracks last actions per board to enable 180° turns, which are
normally prohibited in standard Snake implementations.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tqdm import trange
import environments_fully_observable
from collections import deque


# Configuration
SEED = 0
np.random.seed(SEED)
tf.random.set_seed(SEED)


class TacticalAgent:
    """
    Tactical agent with conditional tail-biting strategy.
    
    Makes decisions based on path efficiency: if the safe path to the fruit
    is significantly longer than the direct distance, considers biting its
    own tail to reset the board and create a shorter path.
    """

    def __init__(self, n_boards):
        """
        Initialize agent for parallel environments.
        
        Args:
            n_boards: Number of parallel game boards
        """
        # Action mapping
        self.moves = {
            0: (1, 0),   # UP
            1: (0, 1),   # RIGHT
            2: (-1, 0),  # DOWN
            3: (0, -1)   # LEFT
        }
        
        # Track last action per board to enable 180° turns
        self.last_actions = np.zeros(n_boards, dtype=int)

    def get_shortest_distance(self, board, start, target):
        """
        Find shortest path distance using breadth-first search.
        
        Searches for a safe path that avoids walls and body segments.
        
        Args:
            board: Game board array
            start: Starting position (row, col)
            target: Target position (row, col)
            
        Returns:
            distance: Number of steps to reach target, or inf if unreachable
        """
        rows, cols = board.shape
        queue = deque([(start, 0)])
        visited = {tuple(start)}

        while queue:
            (r, c), dist = queue.popleft()

            if r == target[0] and c == target[1]:
                return dist
            
            # Explore neighbors
            for (dr, dc) in self.moves.values():
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if (nr, nc) not in visited:
                        val = board[nr, nc]
                        # Can move to empty cells (1) or fruit (2)
                        if val != 0 and val != 3:  # Not wall or body
                            visited.add((nr, nc))
                            queue.append(((nr, nc), dist + 1))
        
        return float('inf')
    
    def count_hugging_neighbors(self, board, r, c):
        """
        Count walls and body segments adjacent to a position.
        
        Used as a tie-breaker: positions with more "hugging" (adjacent obstacles)
        tend to be safer in tight spaces.
        
        Args:
            board: Game board array
            r, c: Position to evaluate
            
        Returns:
            count: Number of adjacent walls or body segments
        """
        count = 0
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < board.shape[0] and 0 <= nc < board.shape[1]:
                if board[nr, nc] == 0 or board[nr, nc] == 3:  # Wall or body
                    count += 1
            else:
                count += 1  # Out of bounds counts as obstacle
        return count
    
    def get_action(self, board, board_idx):
        """
        Select action for a single board based on tactical evaluation.
        
        Decision process:
        1. Find all safe moves and their path distances to fruit
        2. Check if safe path is inefficient (>> direct distance)
        3. If inefficient, evaluate tail-bite: only execute if it moves closer
        4. If trapped (no safe moves), tail-bite as last resort
        
        Args:
            board: Game board array
            board_idx: Board index (for tracking last action)
            
        Returns:
            action: Selected action (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
        """
        # Locate head and fruit
        head_pos = np.argwhere(board == 4)
        fruit_pos = np.argwhere(board == 2)

        if len(head_pos) == 0:
            return 0
        head_pos = head_pos[0]
        
        target = fruit_pos[0] if len(fruit_pos) > 0 else None
        
        # Evaluate all safe moves
        candidates = []
        d_direct = 0
        
        if target is not None:
            # Direct Manhattan distance (ignoring obstacles)
            d_direct = abs(head_pos[0] - target[0]) + abs(head_pos[1] - target[1])
            
            for action, (dr, dc) in self.moves.items():
                nr, nc = head_pos[0] + dr, head_pos[1] + dc

                if 0 <= nr < board.shape[0] and 0 <= nc < board.shape[1]:
                    val = board[nr, nc]
                    if val != 0 and val != 3:  # Safe cell
                        # BFS distance from this position to fruit
                        dist = self.get_shortest_distance(board, (nr, nc), target)
                        hugging = self.count_hugging_neighbors(board, nr, nc)
                        
                        if dist != float('inf'):
                            candidates.append({
                                'action': action,
                                'dist': dist + 1,      # +1 for this move
                                'hugging': hugging
                            })

        # Sort by distance, then by hugging (prefer more constrained moves)
        candidates.sort(key=lambda x: (x['dist'], -x['hugging']))
        best_safe_move = candidates[0] if candidates else None

        # Evaluate tail-biting strategy
        should_reset = False
        
        # Calculate potential tail-bite move (reverse of last action)
        last_act = self.last_actions[board_idx]
        neck_bite_move = (last_act + 2) % 4
        
        # Distance from neck (reverse position) to target
        neck_dr, neck_dc = self.moves[neck_bite_move]
        neck_r, neck_c = head_pos[0] + neck_dr, head_pos[1] + neck_dc
        
        dist_neck_to_target = float('inf')
        if target is not None:
            dist_neck_to_target = abs(neck_r - target[0]) + abs(neck_c - target[1])
        
        # Decision logic
        if best_safe_move is None:
            # Trapped: tail-bite is only option
            should_reset = True
            
        elif best_safe_move is not None:
            d_safe = best_safe_move['dist']
            
            # Inefficiency check: safe path is longer than direct path
            if d_safe > d_direct:
                # Only tail-bite if reverse move brings us closer to fruit
                if dist_neck_to_target < d_direct:
                    should_reset = True

        # Execute move
        if should_reset:
            return neck_bite_move
        else:
            return best_safe_move['action']

    def predict(self, boards):
        """
        Generate actions for all parallel environments.
        
        Args:
            boards: Array of game boards (N, H, W)
            
        Returns:
            actions: Array of actions (N, 1)
        """
        actions = []
        for i, board in enumerate(boards):
            act = self.get_action(board, i)
            actions.append(act)
        
        actions = np.array(actions, dtype=np.int32)
        self.last_actions = actions
        return actions.reshape(-1, 1)


def evaluate_baseline():
    """
    Standalone evaluation of tactical baseline agent.
    
    Runs the agent for 1000 steps on 500 parallel boards and reports
    average cumulative reward.
    """
    print("--- Starting Tactical Agent Evaluation ---")

    # Initialize environment and agent
    n_boards = 500
    board_size = 7
    env = environments_fully_observable.OriginalSnakeEnvironment(n_boards, board_size)
    agent = TacticalAgent(n_boards)
    total_rewards = np.zeros(n_boards)
    steps = 1000

    # Run simulation
    for _ in trange(steps, desc="Simulating"):
        current_boards = env.boards
        actions = agent.predict(current_boards)
        rewards = env.move(actions)
        total_rewards += np.array(rewards).flatten()
    
    # Report results
    avg_reward = np.mean(total_rewards)
    print(f"\n--- Results after {steps} steps with {n_boards} boards ---")
    print(f"Average Reward per Episode: {avg_reward:.4f}")
    return total_rewards


if __name__ == "__main__":
    evaluate_baseline()