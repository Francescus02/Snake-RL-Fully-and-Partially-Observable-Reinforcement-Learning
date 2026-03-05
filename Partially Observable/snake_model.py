"""
snake_model.py — Recurrent Actor-Critic Network for Snake

This module implements a CNN+GRU architecture designed for partially observable
environments where the agent has limited field of view (3×3 window).

Architecture Overview
---------------------
1. CNN Backbone: Single 3×3 convolution extracts spatial features from the FOV
2. GRU Layer: Maintains temporal memory across timesteps
3. Actor Head: Outputs action logits (policy distribution)
4. Critic Head: Estimates state value V(s)

Key Design Decisions
--------------------
- Single convolution layer: For a 3×3 input, one 3×3 kernel covers the entire
  receptive field, making additional convolutional layers redundant.
- GRU instead of LSTM: Fewer parameters (~25% reduction) with comparable
  performance on short-to-medium sequences.
- Recurrent memory: Essential for partial observability—allows the agent to
  remember fruit locations and body positions outside current FOV.

Model Parameters: ~111,000 (significantly lighter than feedforward alternatives)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

GRU_UNITS = 128  # Size of GRU hidden state


class ActorCriticModel(Model):
    """
    Recurrent Actor-Critic model for partially observable Snake.
    
    The model maintains a hidden state that persists across timesteps, allowing
    it to build an internal representation of the environment despite limited
    visual information.
    """

    def __init__(self, num_actions: int = 4):
        """
        Initialize the Actor-Critic model.
        
        Args:
            num_actions: Number of possible actions (default: 4 for Snake)
        """
        super().__init__()

        # CNN Backbone: Spatial feature extraction
        # A single 3×3 convolution on a 3×3 input produces a 1×1 feature map,
        # capturing all spatial relationships in one operation.
        self.conv1   = layers.Conv2D(64, (3, 3), activation='relu',
                                     padding='valid', name='conv_fov')
        self.flatten = layers.Flatten(name='flatten')
        self.proj    = layers.Dense(128, activation='relu', name='proj')
        self.feat_dim = 128

        # Recurrent Layer: Temporal memory
        # return_sequences=True: Output at every timestep for gradient flow (BPTT)
        # return_state=True: Return final hidden state for persistence across rollouts
        self.gru = layers.GRU(GRU_UNITS,
                              return_sequences=True,
                              return_state=True,
                              name='gru_memory')

        # Output Heads
        self.actor_head  = layers.Dense(num_actions, name='actor')   # Policy logits
        self.critic_head = layers.Dense(1, name='critic')            # State value

    def _cnn_features(self, obs):
        """
        Extract spatial features from observations.
        
        Args:
            obs: Observations tensor (B, H, W, C)
            
        Returns:
            Feature tensor (B, feat_dim)
        """
        x = self.conv1(obs)      # (B, 1, 1, 64) for 3×3 input
        x = self.flatten(x)      # (B, 64)
        return self.proj(x)      # (B, 128)

    def call(self, obs, hidden_state=None, training=False):
        """
        Single-step forward pass (used during rollout and evaluation).
        
        Args:
            obs: Current observation (B, H, W, C)
            hidden_state: Previous GRU hidden state (B, GRU_UNITS), or None
            training: Whether in training mode
            
        Returns:
            logits: Action logits (B, num_actions)
            value: State value estimate (B, 1)
            new_hidden: Updated GRU hidden state (B, GRU_UNITS)
        """
        features = self._cnn_features(obs)              # Extract spatial features
        features_seq = tf.expand_dims(features, axis=1) # Add time dimension (B, 1, feat_dim)

        # GRU forward pass
        if hidden_state is not None:
            gru_out, new_hidden = self.gru(features_seq,
                                           initial_state=hidden_state,
                                           training=training)
        else:
            gru_out, new_hidden = self.gru(features_seq, training=training)

        gru_out = tf.squeeze(gru_out, axis=1)  # Remove time dimension (B, GRU_UNITS)
        return self.actor_head(gru_out), self.critic_head(gru_out), new_hidden

    def call_sequence(self, obs_seq, initial_hidden=None, training=False):
        """
        Full-sequence forward pass (used during training with BPTT).
        
        Processes entire rollout sequences at once, allowing gradients to flow
        backward through time for all N steps.
        
        Args:
            obs_seq: Sequence of observations (B, T, H, W, C)
            initial_hidden: Initial GRU hidden state (B, GRU_UNITS), or None
            training: Whether in training mode
            
        Returns:
            logits: Action logits for all timesteps (B, T, num_actions)
            values: State value estimates for all timesteps (B, T, 1)
            final_hidden: Final GRU hidden state (B, GRU_UNITS)
        """
        B = tf.shape(obs_seq)[0]
        T = tf.shape(obs_seq)[1]
        H, W, C = obs_seq.shape[2], obs_seq.shape[3], obs_seq.shape[4]

        # Process all timesteps through CNN in one batch
        obs_flat = tf.reshape(obs_seq, (B * T, H, W, C))
        features = self._cnn_features(obs_flat)               # (B*T, feat_dim)
        features = tf.reshape(features, (B, T, self.feat_dim)) # (B, T, feat_dim)

        # GRU processes the temporal sequence
        if initial_hidden is not None:
            gru_out, final_hidden = self.gru(features,
                                             initial_state=initial_hidden,
                                             training=training)
        else:
            gru_out, final_hidden = self.gru(features, training=training)

        logits = self.actor_head(gru_out)   # (B, T, num_actions)
        values = self.critic_head(gru_out)  # (B, T, 1)

        return logits, values, final_hidden