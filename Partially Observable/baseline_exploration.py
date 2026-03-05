"""
baseline_exploration.py — Rule-Based Baseline Agent for Snake

This module implements a handcrafted heuristic agent that serves as a performance
baseline for evaluating learned policies. The agent operates in partially observable
environments (3×3 field of view) using explicit search and patrolling strategies.

Agent Strategy Overview:
1. HUNT Mode: When fruit is visible in FOV, path toward it using BFS
2. PATROL Mode: When fruit is not visible, follow a systematic patrol pattern
   around the board's inner perimeter to maximize fruit discovery
3. Center Avoidance: Never enter the center cell (3,3) to prevent getting trapped

The agent maintains internal state for each parallel board including:
- Current mode (HUNT/PATROL/REVERSE)
- Patrol direction (clockwise/counterclockwise)
- Snake body positions (for collision avoidance)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tqdm import trange
from collections import deque
import environments_partially_observable


# Configuration
SEED = 0
np.random.seed(SEED)
tf.random.set_seed(SEED)


class SquareExplorerAgent:
    """
    Heuristic agent for partially observable Snake environment.
    
    Operates in two primary modes:
    - HUNT: Navigate toward visible fruit using shortest-path search
    - PATROL: Systematically explore the board when fruit is not visible
    
    The agent avoids the center cell and dynamically adjusts patrol direction
    based on reachability analysis.
    """
    
    def __init__(self, n_boards):
        """
        Initialize agent for parallel environments.
        
        Args:
            n_boards: Number of parallel game boards
        """
        # Action mapping: Snake coordinate system
        self.moves = {
            0: (1, 0),   # DOWN
            1: (0, 1),   # RIGHT
            2: (-1, 0),  # UP
            3: (0, -1)   # LEFT
        }
        
        self.last_actions = np.zeros(n_boards, dtype=int)
        
        # Patrol path: inner square perimeter (clockwise ordering)
        self.patrol_path = [
            (2, 2), (2, 3), (2, 4),
            (3, 4), (4, 4),
            (4, 3), (4, 2),
            (3, 2)
        ]
        
        # Per-board state
        self.mode = ["PATROL"] * n_boards
        self.internal_bodies = [[] for _ in range(n_boards)]
        self.patrol_directions = np.ones(n_boards, dtype=int)  # 1=CW, -1=CCW

    def count_neighbors(self, board, pos, val_type):
        """
        Count neighbors of a specific cell type around a position.
        
        Args:
            board: Game board array
            pos: Position tuple (row, col)
            val_type: Cell type to count (0=wall, 1=empty, 2=fruit, 3=body, 4=head)
            
        Returns:
            count: Number of neighboring cells of specified type
        """
        count = 0
        for dr, dc in self.moves.values():
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < board.shape[0] and 0 <= nc < board.shape[1]:
                if board[nr, nc] == val_type:
                    count += 1
            elif val_type == 0:  # Count out-of-bounds as walls
                count += 1
        return count

    def get_bfs_path(self, board, start, target, ignore_body=False, allow_center=False):
        """
        Find shortest path from start to target using breadth-first search.
        
        Args:
            board: Game board array
            start: Starting position (row, col)
            target: Target position (row, col)
            ignore_body: If True, path through body segments
            allow_center: If True, allow path through center cell (3,3)
            
        Returns:
            path: List of actions to reach target, or None if unreachable
        """
        if np.array_equal(start, target):
            return []
            
        rows, cols = board.shape
        queue = deque([(start, [])])
        visited = {tuple(start)}

        while queue:
            (r, c), path = queue.popleft()
            
            if r == target[0] and c == target[1]:
                return path
            
            # Explore neighbors
            for act, (dr, dc) in self.moves.items():
                nr, nc = r + dr, c + dc
                
                if 0 <= nr < rows and 0 <= nc < cols:
                    # Center avoidance (unless target is center)
                    if not allow_center and nr == 3 and nc == 3:
                        if target[0] != 3 or target[1] != 3:
                            continue
                    
                    val = board[nr, nc]
                    is_safe = (val == 1 or val == 2) if not ignore_body else (val != 0)
                    
                    if (nr, nc) not in visited and is_safe:
                        visited.add((nr, nc))
                        queue.append(((nr, nc), path + [act]))
        
        return None  # No path found

    def evaluate_direction_score(self, board, head_pos, start_idx, direction):
        """
        Score a patrol direction based on reachability of upcoming patrol points.
        
        Args:
            board: Game board
            head_pos: Current head position
            start_idx: Starting index in patrol_path
            direction: +1 for clockwise, -1 for counterclockwise
            
        Returns:
            score: Quality score (higher is better)
            reachable_count: Number of reachable patrol points
        """
        score = 0
        reachable_count = 0
        
        # Check next 4 patrol points in the given direction
        for step in range(1, 5):
            idx = (start_idx + (step * direction) + 80) % len(self.patrol_path)
            target = self.patrol_path[idx]
            path = self.get_bfs_path(board, head_pos, target)
            
            if path is not None:
                reachable_count += 1
                score += (20 / (len(path) + 1))  # Prefer closer targets
                
        return score, reachable_count

    def find_best_patrol_strategy(self, board, head_pos, board_idx):
        """
        Determine optimal patrol entry point and direction.
        
        Scans all patrol path points to find the nearest reachable entry,
        then evaluates both patrol directions to choose the most promising.
        
        Args:
            board: Game board
            head_pos: Current head position
            board_idx: Board index
            
        Returns:
            target: Selected patrol target position
            action: First action toward target, or None if no valid path
        """
        best_entry_idx = -1
        min_path_len = float('inf')
        
        # Find nearest reachable patrol point
        for i, target in enumerate(self.patrol_path):
            if board[target[0], target[1]] != 3:  # Not occupied by body
                path = self.get_bfs_path(board, head_pos, target)
                if path is not None and len(path) < min_path_len:
                    min_path_len = len(path)
                    best_entry_idx = i

        if best_entry_idx == -1:
            return None, None

        # Evaluate both patrol directions from this entry point
        curr_dir = self.patrol_directions[board_idx]
        score_curr, reach_curr = self.evaluate_direction_score(
            board, head_pos, best_entry_idx, curr_dir
        )
        score_opp, reach_opp = self.evaluate_direction_score(
            board, head_pos, best_entry_idx, -curr_dir
        )

        # Switch direction if opposite direction is significantly better
        if reach_opp > reach_curr or (reach_opp == reach_curr and score_opp > score_curr + 2.0):
            self.patrol_directions[board_idx] *= -1
            curr_dir = self.patrol_directions[board_idx]

        target = self.patrol_path[best_entry_idx]
        path = self.get_bfs_path(board, head_pos, target)
        
        # If already at target, advance to next patrol point
        if path is not None and len(path) <= 1:
            next_idx = (best_entry_idx + curr_dir + 80) % len(self.patrol_path)
            target = self.patrol_path[next_idx]
            path = self.get_bfs_path(board, head_pos, target)
            
        if path and len(path) > 0:
            return target, path[0]
        
        return None, None

    def get_action(self, board, mask, board_idx):
        """
        Select action for a single board based on current observation.
        
        Decision hierarchy:
        1. If fruit visible in FOV → HUNT mode (navigate toward fruit)
        2. If no fruit visible → PATROL mode (explore inner square)
        3. If stuck → REVERSE mode (turn around)
        
        Args:
            board: Full game board (for internal pathfinding)
            mask: Partial observation (3×3 FOV)
            board_idx: Board index
            
        Returns:
            action: Selected action (0=DOWN, 1=RIGHT, 2=UP, 3=LEFT)
        """
        # Locate snake head
        head_pos = np.argwhere(board == 4)
        if len(head_pos) == 0:
            return 0
        head_pos = head_pos[0]
        
        # Locate tail (for collision avoidance)
        tail_pos = self.internal_bodies[board_idx][-1] if self.internal_bodies[board_idx] else head_pos
        last_act = self.last_actions[board_idx]

        # Check for fruit in field of view
        fruit_in_mask = np.argwhere(mask == 2)
        target_fruit = None
        if len(fruit_in_mask) > 0:
            # Convert FOV coordinates to board coordinates
            dr, dc = fruit_in_mask[0][0] - 1, fruit_in_mask[0][1] - 1
            target_fruit = (head_pos[0] + dr, head_pos[1] + dc)
            self.mode[board_idx] = "HUNT"

        action = None

        # HUNT MODE: Navigate toward visible fruit
        if self.mode[board_idx] == "HUNT" and target_fruit:
            candidates = []
            fruit_at_center = (target_fruit[0] == 3 and target_fruit[1] == 3)
            
            # Evaluate all possible actions
            for act, (dr, dc) in self.moves.items():
                nr, nc = head_pos[0] + dr, head_pos[1] + dc
                
                if 0 <= nr < board.shape[0] and 0 <= nc < board.shape[1]:
                    if board[nr, nc] in [1, 2]:  # Empty or fruit
                        # Skip center unless fruit is there
                        if nr == 3 and nc == 3 and not fruit_at_center:
                            continue
                        
                        path = self.get_bfs_path(board, (nr, nc), target_fruit)
                        if path is not None:
                            center_dist = abs(nr - 3) + abs(nc - 3)
                            wall_h = self.count_neighbors(board, (nr, nc), 0)
                            axis_change = 1 if (act % 2 != last_act % 2) else 0
                            
                            candidates.append({
                                'action': act,
                                'dist': len(path),
                                'center_dist': center_dist,  # Prefer corners
                                'wall': wall_h,              # Prefer wall-hugging
                                'axis': axis_change          # Prefer axis changes
                            })
            
            if candidates:
                # Priority: Distance → Corner preference → Wall-hugging → Axis change
                candidates.sort(key=lambda x: (x['dist'], -x['center_dist'], -x['wall'], -x['axis']))
                action = candidates[0]['action']
            else:
                # Fallback: ignore body constraints
                path_f = self.get_bfs_path(board, head_pos, target_fruit,
                                           ignore_body=True,
                                           allow_center=fruit_at_center)
                if path_f:
                    action = path_f[0]

        # PATROL MODE: Systematic exploration when no fruit is visible
        if action is None:
            target, act_bfs = self.find_best_patrol_strategy(board, head_pos, board_idx)
            
            if act_bfs is not None:
                candidates = []
                
                for act, (dr, dc) in self.moves.items():
                    nr, nc = head_pos[0] + dr, head_pos[1] + dc
                    
                    if 0 <= nr < board.shape[0] and 0 <= nc < board.shape[1]:
                        if board[nr, nc] in [1, 2]:
                            if nr == 3 and nc == 3:  # Skip center
                                continue
                            
                            path = self.get_bfs_path(board, (nr, nc), target)
                            if path is not None:
                                body_h = self.count_neighbors(board, (nr, nc), 3)
                                inner = 2 <= nr <= 4 and 2 <= nc <= 4
                                
                                candidates.append({
                                    'action': act,
                                    'dist': len(path),
                                    'inner': inner,  # Prefer inner area
                                    'body': body_h   # Prefer body-adjacent moves
                                })
                
                if candidates:
                    # Priority: Distance → Inner area → Body-hugging
                    candidates.sort(key=lambda x: (x['dist'], not x['inner'], -x['body']))
                    action = candidates[0]['action']
                else:
                    action = act_bfs
                    
                self.mode[board_idx] = "PATROL"
            else:
                # REVERSE: No valid patrol path, turn around
                action = (last_act + 2) % 4
                self.mode[board_idx] = "REVERSE"

        # Safety check: avoid biting tail
        if action is not None:
            dr, dc = self.moves[action]
            nr, nc = head_pos[0] + dr, head_pos[1] + dc
            if (nr == tail_pos[0] and nc == tail_pos[1]):
                action = (last_act + 2) % 4  # Turn around

        self.last_actions[board_idx] = action if action is not None else 0
        return self.last_actions[board_idx]

    def predict(self, env):
        """
        Generate actions for all parallel environments.
        
        Args:
            env: Environment instance
            
        Returns:
            actions: Array of actions (N, 1)
        """
        boards = env.boards
        states = env.to_state()
        masks = np.argmax(states, axis=-1) + 1  # Convert one-hot to cell values
        
        actions = [self.get_action(boards[i], masks[i], i) 
                   for i in range(len(boards))]
        return np.array(actions).reshape(-1, 1)


def evaluate():
    """
    Standalone evaluation of baseline agent.
    
    Runs the agent for 1000 steps on 500 parallel boards and reports
    average cumulative reward.
    """
    n_boards = 500
    board_size = 7
    mask_size = 1
    
    env = environments_partially_observable.OriginalSnakeEnvironment(
        n_boards, board_size, mask_size
    )
    agent = SquareExplorerAgent(n_boards)
    steps = 1000
    total_reward = 0

    print(f"--- Baseline Agent Evaluation ---")
    print(f"Steps: {steps}  |  Boards: {n_boards}")
    
    for _ in trange(steps):
        actions = agent.predict(env)
        rewards = env.move(actions)
        total_reward += np.sum(rewards)
        
        # Update agent's internal body tracking
        for i in range(n_boards):
            agent.internal_bodies[i] = env.bodies[i]
    
    print(f"\nAverage Reward per Board: {total_reward / n_boards:.2f}")


if __name__ == "__main__":
    evaluate()