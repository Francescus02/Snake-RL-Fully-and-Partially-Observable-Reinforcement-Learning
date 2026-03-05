"""
snake_model.py — Actor-Critic Network for Fully Observable Snake

This module implements a CNN-based Actor-Critic architecture optimized for
fully observable Snake environments. The network processes the complete 7×7
board state to extract spatial features and predict both policy and value.

Architecture Overview
---------------------
1. CNN Backbone: Three convolutional layers with optional residual connections
2. Dense Layers: Two fully connected layers for high-level reasoning
3. Actor Head: Outputs action logits (policy distribution)
4. Critic Head: Estimates state value V(s)

Design Considerations
---------------------
- Residual connections: Optional skip connections can help gradient flow and
  allow the network to learn both low-level (spatial patterns) and high-level
  (strategic planning) features simultaneously.
- Batch normalization: Stabilizes training but adds computation overhead.
  Generally not necessary for small networks like this.
- Current architecture (~527K params) is well-suited for 7×7 Snake without
  overfitting risk given sufficient data augmentation.

Model Parameters: ~527,000 (base) or ~532,000 (with residuals)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


class ActorCriticModel(Model):
    """
    CNN-based Actor-Critic model for fully observable Snake.
    
    Processes complete board state through convolutional layers to extract
    spatial features, then uses dense layers for policy and value estimation.
    """

    def __init__(self, num_actions=4, use_residual=False):
        """
        Initialize the Actor-Critic model.
        
        Args:
            num_actions: Number of possible actions (default: 4 for Snake)
            use_residual: If True, add residual connections in CNN backbone
        """
        super(ActorCriticModel, self).__init__()
        
        self.use_residual = use_residual
        
        # CNN Backbone: Spatial feature extraction from 7×7 board
        # Using 'same' padding maintains spatial dimensions (7×7) through all layers
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', 
                                   padding='same', name='conv1')
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu', 
                                   padding='same', name='conv2')
        self.conv3 = layers.Conv2D(64, (3, 3), activation='relu', 
                                   padding='same', name='conv3')
        
        # Optional: Projection layer for residual connections
        # Projects 32 channels to 64 to match conv2/conv3 output
        if use_residual:
            self.projection = layers.Conv2D(64, (1, 1), padding='same', 
                                           name='residual_proj')
        
        self.flatten = layers.Flatten()
        
        # Dense layers: High-level feature processing
        self.dense1 = layers.Dense(256, activation='relu', name='dense1')
        self.dense2 = layers.Dense(128, activation='relu', name='dense2')
        
        # Output heads
        self.actor_head  = layers.Dense(num_actions, name='actor')   # Policy logits
        self.critic_head = layers.Dense(1, name='critic')            # State value

    def call(self, inputs, training=False):
        """
        Forward pass through the network.
        
        Args:
            inputs: Board state tensor (B, H, W, C) where C=4 one-hot channels:
                    [empty, wall, fruit, body/head]
            training: Whether in training mode (for potential dropout/batchnorm)
        
        Returns:
            logits: Action logits (B, num_actions)
            value: State value estimate (B, 1)
        """
        # Convolutional feature extraction
        x1 = self.conv1(inputs)  # (B, 7, 7, 32)
        x2 = self.conv2(x1)      # (B, 7, 7, 64)
        
        # Optional residual connection: x2 = x2 + proj(x1)
        if self.use_residual:
            x2 = x2 + self.projection(x1)
        
        x3 = self.conv3(x2)      # (B, 7, 7, 64)
        
        # Optional residual connection: x3 = x3 + x2
        if self.use_residual:
            x3 = x3 + x2
        
        # Dense processing
        x = self.flatten(x3)
        x = self.dense1(x)
        x = self.dense2(x)
        
        return self.actor_head(x), self.critic_head(x)