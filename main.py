import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
try:
    import multiprocessing
    multiprocessing.set_start_method("fork", force=True)
except RuntimeError:
    pass


torch.set_float32_matmul_precision("high")


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gelu_scratch(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x ** 3))))


def dropout_scratch(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    if (not training) or p == 0.0:
        return x
    if not 0.0 <= p < 1.0:
        raise ValueError("dropout probability has to be in [0, 1)")
    keep_prob = 1.0 - p
    mask = (torch.rand_like(x) < keep_prob).to(x.dtype)
    return (x * mask) / keep_prob


def cross_entropy_scratch(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    max_logits = logits.max(dim=1, keepdim=True).values
    stabilized = logits - max_logits
    logsumexp = torch.log(torch.exp(stabilized).sum(dim=1, keepdim=True))
    log_probs = stabilized - logsumexp
    losses = -log_probs.gather(1, targets.unsqueeze(1))
    return losses.mean()

def softmax_scratch(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    max_vals = x.max(dim=dim, keepdim=True).values
    stabilized = x - max_vals
    exp_x = torch.exp(stabilized)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def image_to_patches_scratch(
    x: torch.Tensor,
    patch_size: int,
    expected_channels: int,
) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError(f"expected input of shape [batch, channels, height, width], got {tuple(x.shape)}")

    bsz, chans, height, width = x.shape
    if chans != expected_channels:
        raise ValueError(f"expected {expected_channels} input channels, got {chans}")
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"image size ({height}, {width}) must be divisible by patch_size={patch_size}"
        )

    h_patches = height // patch_size
    w_patches = width // patch_size
    patches = x.reshape(bsz, chans, h_patches, patch_size, w_patches, patch_size)
    patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
    return patches.view(bsz, h_patches * w_patches, chans * patch_size * patch_size)

class LinearScratch(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.transpose(0, 1) + (self.bias if self.bias is not None else 0.0)


class LayerNormScratch(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_hat + self.bias


class PatchEmbedScratch(nn.Module):
    def __init__(self, img_size: int = 64, patch_size: int = 8, in_chans: int = 3, embed_dim: int = 256):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_chans * patch_size * patch_size
        self.proj = LinearScratch(self.patch_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = image_to_patches_scratch(
            x,
            patch_size=self.patch_size,
            expected_channels=self.in_chans,
        )
        return self.proj(patches)


class MultiHeadSelfAttentionScratch(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = LinearScratch(dim, dim)
        self.k_proj = LinearScratch(dim, dim)
        self.v_proj = LinearScratch(dim, dim)
        self.out_proj = LinearScratch(dim, dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = softmax_scratch(attn_scores, dim=-1)
        attn_weights = dropout_scratch(attn_weights, p=self.dropout, training=self.training)

        attn_out = attn_weights @ v
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.dim)
        attn_out = self.out_proj(attn_out)
        attn_out = dropout_scratch(attn_out, p=self.dropout, training=self.training)
        return attn_out


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 64, patch_size: int = 8, in_chans: int = 3, embed_dim: int = 256):
        super().__init__()
        self.embed = PatchEmbedScratch(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = self.embed.num_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = LinearScratch(dim, hidden)
        self.fc2 = LinearScratch(hidden, dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = gelu_scratch(x)
        x = dropout_scratch(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = dropout_scratch(x, p=self.dropout, training=self.training)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = LayerNormScratch(dim)
        self.attn = MultiHeadSelfAttentionScratch(dim=dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = LayerNormScratch(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_in = self.norm1(x)
        attn_out = self.attn(attn_in)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

def sincos2d(n, dim):
    h = w = int(n ** 0.5)
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")

    pos = torch.stack([x, y], dim=-1).reshape(-1, 2).float()

    omega = torch.arange(dim // 4).float()
    omega = 1. / (10000 ** (omega / (dim // 4)))

    out = pos[..., None] * omega

    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)
    return emb.flatten(1)

class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_chans: int = 3,
        num_classes: int = 10,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        pos_type = 'learned'
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_type = pos_type
        if self.pos_type == 'learned':
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        elif self.pos_type == '2d':
            pos = sincos2d(num_patches, embed_dim)
            pos = torch.cat([torch.zeros(1, embed_dim), pos], dim=0)
            self.register_buffer("pos_embed", pos.unsqueeze(0))
        elif self.pos_type == 'none':
            self.register_buffer("pos_embed",torch.zeros(1, num_patches + 1, embed_dim))

        self.pos_drop_p = dropout

        self.blocks = nn.Sequential(*[
            EncoderBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = LayerNormScratch(embed_dim)
        self.head = LinearScratch(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if isinstance(self.pos_embed, nn.Parameter):
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        bsz = x.shape[0]
        cls_tokens = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = dropout_scratch(x, p=self.pos_drop_p, training=self.training)
        x = self.blocks(x)
        x = self.norm(x)
        #cls_rep = x[:, 0]
        patches = x[:, 1:]
        res = patches.mean(dim=1)
        return self.head(res)


# -----------------------------
# Swin Transformer
# -----------------------------

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    bsz, height, width, chans = x.shape
    if height % window_size != 0 or width % window_size != 0:
        raise ValueError(
            f"feature map size ({height}, {width}) must be divisible by window_size={window_size}"
        )
    h_windows = height // window_size
    w_windows = width // window_size
    x = x.view(bsz, h_windows, window_size, w_windows, window_size, chans)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(bsz * h_windows * w_windows, window_size, window_size, chans)


def window_reverse(windows: torch.Tensor, window_size: int, height: int, width: int) -> torch.Tensor:
    h_windows = height // window_size
    w_windows = width // window_size
    bsz = windows.shape[0] // (h_windows * w_windows)
    x = windows.view(bsz, h_windows, w_windows, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(bsz, height, width, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = LinearScratch(dim, dim)
        self.k_proj = LinearScratch(dim, dim)
        self.v_proj = LinearScratch(dim, dim)
        self.out_proj = LinearScratch(dim, dim)
        self.dropout = dropout


    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            num_windows = attn_mask.shape[0]
            attn_scores = attn_scores.view(
                bsz // num_windows, num_windows, self.num_heads, seq_len, seq_len
            )
            attn_scores = attn_scores + attn_mask.view(1, num_windows, 1, seq_len, seq_len)
            attn_scores = attn_scores.view(bsz, self.num_heads, seq_len, seq_len)

        attn_weights = softmax_scratch(attn_scores, dim=-1)
        attn_weights = dropout_scratch(attn_weights, p=self.dropout, training=self.training)

        attn_out = attn_weights @ v
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.dim)
        attn_out = self.out_proj(attn_out)
        attn_out = dropout_scratch(attn_out, p=self.dropout, training=self.training)
        return attn_out


class SwinBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: int = 4,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert 0 <= shift_size < window_size, "shift_size must be in [0, window_size)"
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = LayerNormScratch(dim)
        self.attn = WindowAttention(
            dim=dim, window_size=window_size, num_heads=num_heads, dropout=dropout
        )
        self.norm2 = LayerNormScratch(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def compute_attn_mask(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        if self.shift_size == 0:
            return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = self.input_resolution
        bsz, seq_len, chans = x.shape
        assert seq_len == height * width, "input feature has wrong size"

        residual = x
        x = self.norm1(x)
        x = x.view(bsz, height, width, chans)

        if self.shift_size > 0:
            shifted = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted = x

        x_windows = window_partition(shifted, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, chans)

        attn_mask = self.compute_attn_mask(height, width, x.device)
        attn_windows = self.attn(x_windows, attn_mask=attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, chans)
        shifted = window_reverse(attn_windows, self.window_size, height, width)

        if self.shift_size > 0:
            x = torch.roll(shifted, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted

        x = x.view(bsz, height * width, chans)
        x = residual + x
        x = x + self.mlp(self.norm2(x))
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution: Tuple[int, int], dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.norm = LayerNormScratch(4 * dim)
        self.reduction = LinearScratch(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = self.input_resolution
        bsz, seq_len, chans = x.shape
        assert seq_len == height * width, "input feature has wrong size"
        assert height % 2 == 0 and width % 2 == 0, "H and W must be even for patch merging"

        x = x.view(bsz, height, width, chans)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(bsz, (height // 2) * (width // 2), 4 * chans)

        x = self.norm(x)
        x = self.reduction(x)
        return x


class SwinPatchEmbed(nn.Module):
    def __init__(self, img_size: int = 64, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 96):
        super().__init__()
        self.embed = PatchEmbedScratch(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = self.embed.num_patches
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        downsample: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.blocks = nn.ModuleList([
            SwinBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for i in range(depth)
        ])

        if downsample:
            self.downsample = PatchMerging(input_resolution=input_resolution, dim=dim)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class SwinTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 10,
        embed_dim: int = 96,
        depths: Tuple[int, ...] = (2, 2, 6),
        num_heads: Tuple[int, ...] = (3, 6, 12),
        window_size: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert len(depths) == len(num_heads), "depths and num_heads must have the same length"
        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_stages - 1))

        self.patch_embed = SwinPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.pos_drop_p = dropout

        self.layers = nn.ModuleList()
        cur_resolution = self.patch_embed.patches_resolution
        for i_stage in range(self.num_stages):
            stage_dim = int(embed_dim * 2 ** i_stage)
            is_last = i_stage == self.num_stages - 1
            layer = BasicLayer(
                dim=stage_dim,
                input_resolution=cur_resolution,
                depth=depths[i_stage],
                num_heads=num_heads[i_stage],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                downsample=not is_last,
            )
            self.layers.append(layer)
            if not is_last:
                cur_resolution = (cur_resolution[0] // 2, cur_resolution[1] // 2)

        self.norm = LayerNormScratch(self.num_features)
        self.head = LinearScratch(self.num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = dropout_scratch(x, p=self.pos_drop_p, training=self.training)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


@dataclass
class TrainConfig:
    data_dir: str
    img_size: int = 64
    batch_size: int = 256
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    val_split: float = 0.1
    test_split: float = 0.1
    num_workers: int = 8
    seed: int = 42
    patch_size: int = 8
    embed_dim: int = 192
    depth: int = 4
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    in_chans: int = 3
    pos_type: str = 'learned'
    model: str = 'vit'
    window_size: int = 4
    swin_embed_dim: int = 96
    swin_depths: Tuple[int, ...] = (2, 2, 6)
    swin_num_heads: Tuple[int, ...] = (3, 6, 12)


def build_dataloaders(cfg: TrainConfig):
    normalize = transforms.Normalize(mean=[0.5] * cfg.in_chans, std=[0.5] * cfg.in_chans)

    train_tfms = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        normalize,
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        normalize,
    ])

    full_for_split = datasets.ImageFolder(cfg.data_dir)
    num_total = len(full_for_split)
    num_test = int(cfg.test_split * num_total)
    num_val = int(cfg.val_split * num_total)
    num_train = num_total - num_val - num_test

    generator = torch.Generator().manual_seed(cfg.seed)
    train_subset, val_subset, test_subset = random_split(
        full_for_split,
        [num_train, num_val, num_test],
        generator=generator,
    )

    train_ds = datasets.ImageFolder(cfg.data_dir, transform=train_tfms)
    train_ds.samples = [train_ds.samples[i] for i in train_subset.indices]
    train_ds.targets = [s[1] for s in train_ds.samples]

    val_ds = datasets.ImageFolder(cfg.data_dir, transform=eval_tfms)
    val_ds.samples = [val_ds.samples[i] for i in val_subset.indices]
    val_ds.targets = [s[1] for s in val_ds.samples]

    test_ds = datasets.ImageFolder(cfg.data_dir, transform=eval_tfms)
    test_ds.samples = [test_ds.samples[i] for i in test_subset.indices]
    test_ds.targets = [s[1] for s in test_ds.samples]

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=False,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=4 if cfg.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=4 if cfg.num_workers > 0 else None,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=4 if cfg.num_workers > 0 else None,
    )

    classes = full_for_split.classes
    return train_loader, val_loader, test_loader, classes


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, total_correct / total


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, total_correct / total


def main(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, classes = build_dataloaders(cfg)
    print(f"Classes ({len(classes)}): {classes}")

    if cfg.model == "vit":
        model = VisionTransformer(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_chans,
            num_classes=len(classes),
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
            pos_type = cfg.pos_type
        ).to(device)
    elif cfg.model == "swin":
        model = SwinTransformer(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_chans,
            num_classes=len(classes),
            embed_dim=cfg.swin_embed_dim,
            depths=tuple(cfg.swin_depths),
            num_heads=tuple(cfg.swin_num_heads),
            window_size=cfg.window_size,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
        ).to(device)
    else:
        raise ValueError(f"unknown model: {cfg.model}")

    #Resnet
    # model = models.resnet50(weights=None)
    # model.fc = nn.Linear(model.fc.in_features, len(classes))
    # model = model.to(device)

    criterion = cross_entropy_scratch
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_val_acc = -math.inf
    best_path = Path(f"best_eurosat_{cfg.model}_scratch_mps_fast.pt")

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                    "config": cfg.__dict__,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Best val_acc={best_val_acc:.4f}")
    print(f"Test loss={test_loss:.4f} | Test acc={test_acc:.4f}")
    print(f"Saved best checkpoint to: {best_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a scratch Vision Transformer on EuroSAT")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to EuroSAT root folder organized by class")
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--embed_dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--in_chans", type=int, default=3)
    parser.add_argument("--pos_type", type=str, default='learned')
    parser.add_argument("--model", type=str, default='vit', choices=['vit', 'swin'])
    parser.add_argument("--window_size", type=int, default=4)
    parser.add_argument("--swin_embed_dim", type=int, default=96)
    parser.add_argument("--swin_depths", type=int, nargs="+", default=[2, 2, 6])
    parser.add_argument("--swin_num_heads", type=int, nargs="+", default=[3, 6, 12])
    args = parser.parse_args()

    cfg = TrainConfig(**vars(args))
    main(cfg)
