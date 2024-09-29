# 1D MaxViT adapted for audio signals
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/max_vit_with_registers.py

from functools import partial

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Sequential

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange, Reduce

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pack_one(x, pattern):
    return pack([x], pattern)

def unpack_one(x, ps, pattern):
    return unpack(x, ps, pattern)[0]

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# helper classes

def FeedForward(dim, mult = 4, dropout = 0.):
    inner_dim = int(dim * mult)
    return Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim),
        nn.Dropout(dropout)
    )

# MBConv

class SqueezeExcitation(Module):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = Sequential(
            Reduce('b c l -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias = False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias = False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1')
        )

    def forward(self, x):
        return x * self.gate(x)

class MBConvResidual(Module):
    def __init__(self, fn, dropout = 0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x

class Dropsample(Module):
    def __init__(self, prob = 0):
        super().__init__()
        self.prob = prob
  
    def forward(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1), device = device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)

def MBConv(
    dim_in,
    dim_out,
    *,
    downsample,
    expansion_rate = 4,
    shrinkage_rate = 0.25,
    dropout = 0.
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = Sequential(
        nn.Conv1d(dim_in, hidden_dim, 1),
        nn.BatchNorm1d(hidden_dim),
        nn.GELU(),
        nn.Conv1d(hidden_dim, hidden_dim, 3, stride = stride, padding = 1, groups = hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate = shrinkage_rate),
        nn.Conv1d(hidden_dim, dim_out, 1),
        nn.BatchNorm1d(dim_out)
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout = dropout)

    return net

# attention related classes

class Attention1D(Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        dropout = 0.,
        window_size = 7,
        num_registers = 1
    ):
        super().__init__()
        assert num_registers > 0
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias = False),
            nn.Dropout(dropout)
        )

        # relative positional bias

        num_rel_pos_bias = 2 * window_size - 1

        self.rel_pos_bias = nn.Embedding(num_rel_pos_bias + 1, self.heads)

        pos = torch.arange(window_size)
        rel_pos = rearrange(pos, 'i -> i 1') - rearrange(pos, 'j -> 1 j')
        rel_pos += window_size - 1  # make indices non-negative
        rel_pos_indices = rel_pos  # shape (window_size, window_size)

        rel_pos_indices = F.pad(rel_pos_indices, (num_registers, 0, num_registers, 0), value = num_rel_pos_bias)
        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)

    def forward(self, x):
        device, h, bias_indices = x.device, self.heads, self.rel_pos_indices

        x = self.norm(x)

        # project for queries, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # scale

        q = q * self.scale

        # sim

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add positional bias

        bias = self.rel_pos_bias(bias_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # attention

        attn = self.attend(sim)

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # combine heads out

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class MaxViT1D(Module):
    def __init__(
        self,
        *,
        patch_size = 1024,
        num_classes = 1000,
        dim = 192,
        depth = (2,2,5,2),
        dim_head = 32,
        dim_conv_stem = None,
        window_size = 7,
        mbconv_expansion_rate = 4,
        mbconv_shrinkage_rate = 0.25,
        dropout = 0.1,
        channels = 2,
        num_register_tokens = 4,
        downsample = True,
        dense_prediction = False,
    ):
        super().__init__()
        assert isinstance(depth, tuple), 'depth needs to be tuple of integers indicating number of transformer blocks at each stage'
        assert num_register_tokens > 0

        self.channels = channels
        self.patch_size = patch_size
        self.dense_prediction = dense_prediction

        # convolutional stem

        dim_conv_stem = default(dim_conv_stem, dim)

        self.conv_stem = Sequential(
            nn.Conv1d(channels, dim_conv_stem, patch_size, stride = patch_size, padding = 0),
        )

        # variables

        num_stages = len(depth)

        dims = tuple(map(lambda i: (2 ** i) * dim, range(num_stages)))
        dims = (dim_conv_stem, *dims)
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        self.layers = nn.ModuleList([])

        # window size

        self.window_size = window_size

        self.register_tokens = nn.ParameterList([])

        # iterate through stages

        for ind, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(dim_pairs, depth)):
            for stage_ind in range(layer_depth):
                is_first = stage_ind == 0
                stage_dim_in = layer_dim_in if is_first else layer_dim

                conv = MBConv(
                    stage_dim_in,
                    layer_dim,
                    downsample = is_first if downsample else False,
                    expansion_rate = mbconv_expansion_rate,
                    shrinkage_rate = mbconv_shrinkage_rate
                )

                block_attn = Attention1D(dim = layer_dim, dim_head = dim_head, dropout = dropout, window_size = window_size, num_registers = num_register_tokens)
                block_ff = FeedForward(dim = layer_dim, dropout = dropout)

                grid_attn = Attention1D(dim = layer_dim, dim_head = dim_head, dropout = dropout, window_size = window_size, num_registers = num_register_tokens)
                grid_ff = FeedForward(dim = layer_dim, dropout = dropout)

                register_tokens = nn.Parameter(torch.randn(num_register_tokens, layer_dim))

                self.layers.append(ModuleList([
                    conv,
                    ModuleList([block_attn, block_ff]),
                    ModuleList([grid_attn, grid_ff])
                ]))

                self.register_tokens.append(register_tokens)

        # mlp head out

        if dense_prediction:
            self.mlp_head = nn.Sequential(
                Rearrange('b d l -> b l d'),
                nn.LayerNorm(dims[-1]),
                nn.Linear(dims[-1], num_classes)
            )
        else:
            self.mlp_head = nn.Sequential(
                Reduce('b d l -> b d', 'mean'),
                nn.LayerNorm(dims[-1]),
                nn.Linear(dims[-1], num_classes)
            )

    def forward(self, x):
        b, w = x.shape[0], self.window_size

        x = self.conv_stem(x)

        for (conv, (block_attn, block_ff), (grid_attn, grid_ff)), register_tokens in zip(self.layers, self.register_tokens):
            x = conv(x)

            # block-like attention

            x = rearrange(x, 'b d (l w1) -> b l w1 d', w1 = w)

            # prepare register tokens

            r = repeat(register_tokens, 'n d -> b l n d', b = b, l = x.shape[1])
            r, register_batch_ps = pack_one(r, '* n d')

            x, window_ps = pack_one(x, 'b l * d')
            x, batch_ps  = pack_one(x, '* n d')
            x, register_ps = pack([r, x], 'b * d')

            x = block_attn(x) + x
            x = block_ff(x) + x

            r, x = unpack(x, register_ps, 'b * d')

            x = unpack_one(x, batch_ps, '* n d')
            x = unpack_one(x, window_ps, 'b l * d')
            x = rearrange(x, 'b l w1 d -> b d (l w1)')

            r = unpack_one(r, register_batch_ps, '* n d')

            # grid-like attention

            x = rearrange(x, 'b d (w1 l) -> b l w1 d', w1 = w)

            # prepare register tokens

            r = reduce(r, 'b l n d -> b n d', 'mean')
            r = repeat(r, 'b n d -> b l n d', l = x.shape[1])
            r, register_batch_ps = pack_one(r, '* n d')

            x, window_ps = pack_one(x, 'b l * d')
            x, batch_ps  = pack_one(x, '* n d')
            x, register_ps = pack([r, x], 'b * d')

            x = grid_attn(x) + x

            r, x = unpack(x, register_ps, 'b * d')

            x = grid_ff(x) + x

            x = unpack_one(x, batch_ps, '* n d')
            x = unpack_one(x, window_ps, 'b l * d')
            x = rearrange(x, 'b l w1 d -> b d (l w1)')

        x = self.mlp_head(x)
        if self.dense_prediction:
            l = x.shape[1]
            pL = self.patch_size
            x = rearrange(x,'b (l) (pL c) -> b c (l pL)', pL=pL, c=self.channels)
        return x
