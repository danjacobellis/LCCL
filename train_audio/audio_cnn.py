import torch
import torch.nn as nn
from einops import rearrange
from diffusers.models.autoencoders.autoencoder_oobleck import Snake1d

class AudioCNN(nn.Module):
    def __init__(self, channels, embed_dim, patch_size, depth):
        super(AudioCNN, self).__init__()
        self.channels = channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.depth = depth

        self.conv_stem = nn.Conv1d(
            in_channels=self.channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
        )

        self.conv_layers = nn.ModuleList([
            nn.Conv1d(
                in_channels=self.embed_dim,
                out_channels=self.embed_dim,
                kernel_size=11,
                stride=1,
                padding=5,
                padding_mode='reflect'
            ) for _ in range(self.depth)
        ])
        self.snake_activations = nn.ModuleList([
            Snake1d(hidden_dim=self.embed_dim) for _ in range(self.depth)
        ])


        self.linear_projection = nn.Linear(self.embed_dim, self.patch_size * self.channels)
        self.clamp = nn.Hardtanh(min_val=-1.0, max_val=1.0)


    def forward(self, x):
        x = self.conv_stem(x)
        for conv_layer, snake_activation in zip(self.conv_layers, self.snake_activations):
            residual = x
            x = conv_layer(x)
            x = x + residual
            x = snake_activation(x)
        x = rearrange(x, 'b c l -> b l c')
        x = self.linear_projection(x)
        x = self.clamp(x)
        x = rearrange(x, 'b l (p c) -> b c (l p)', p=self.patch_size, c=self.channels)
        return x