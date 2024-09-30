import torch
import torch.nn as nn
import torch.nn.functional as F
import diffusers.models.autoencoders as autoencoders
import einops

class IsotropicOobleckAutoencoder(nn.Module):
    def __init__(self, channels, patch_size, embed_dim, depth):
        super().__init__()
        self.channels = channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth=depth

        self.patch_embedding = nn.Conv1d(
            in_channels=self.channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0
        )

        self.encoder = nn.Sequential(
            autoencoders.autoencoder_oobleck.OobleckEncoder(
                encoder_hidden_size=self.embed_dim,
                audio_channels=self.embed_dim,
                downsampling_ratios=(self.depth//2) * [1],
                channel_multiples=(self.depth//2) * [1],
            )
        )

        self.decoder = nn.Sequential(
            autoencoders.autoencoder_oobleck.OobleckDecoder(
                channels=self.embed_dim,
                input_channels=self.embed_dim,
                audio_channels=self.embed_dim,
                upsampling_ratios=(self.depth//2) * [1],
                channel_multiples=(self.depth//2) * [1],
            )
        )

        self.patch_unembedding = nn.Linear(self.embed_dim, self.channels * self.patch_size)

        self.clamp = torch.nn.Hardtanh(min_val=-1, max_val=1)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = einops.rearrange(x, 'b c l -> b l c')
        x = self.patch_unembedding(x)
        x = einops.rearrange(x, 'b l (c p) -> b c (l p)', c=self.channels, p=self.patch_size)
        return self.clamp(x)
