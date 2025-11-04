import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import math
import torch.cuda

from networks.common.se import SE
from networks.common.mca import MCA
from networks.common.mca import MCALayer

class PositionalEncodingFourier(nn.Module):
    """
    Positional encoding relying on a fourier kernel matching the one used in the
    "Attention is all of Need" paper. The implementation builds on DeTR code
    https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    """

    def __init__(self, hidden_dim=32, dim=768, temperature=10000):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim

    def forward(self, B, H, W):
        mask = torch.zeros(B, H, W).bool().to(self.token_projection.weight.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                             pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                             pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.token_projection(pos)
        return pos


class XCA(nn.Module):
    """ Cross-Covariance Attention (XCA) operation where the channels are updated using a weighted
     sum. The weights are obtained from the (softmax normalized) Cross-covariance
    matrix (Q^T K \\in d_h \\times d_h)
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)


    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class BNGELU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-5)
        self.act = nn.GELU()

    def forward(self, x):
        output = self.bn(x)
        output = self.act(output)

        return output


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding=0, dilation=(1, 1), groups=1, bn_act=False, bias=False):
        super().__init__()

        self.bn_act = bn_act

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_act:
            self.bn_gelu = BNGELU(nOut)

    def forward(self, x):
        output = self.conv(x)

        if self.bn_act:
            output = self.bn_gelu(output)

        return output


class CDilated(nn.Module):
    """
    This class defines the dilated convolution.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1, bias=False):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=bias,
                              dilation=d, groups=groups)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """

        output = self.conv(input)
        return output


class DilatedConv(nn.Module):
    """
    A single Dilated Convolution layer in the Consecutive Dilated Convolutions (CDC) module.
    """
    def __init__(self, dim, k, dilation=1, stride=1, drop_path=0.,
                 layer_scale_init_value=1e-6, expan_ratio=6):
        """
        :param dim: input dimension
        :param k: kernel size
        :param dilation: dilation rate
        :param drop_path: drop_path rate
        :param layer_scale_init_value:
        :param expan_ratio: inverted bottelneck residual
        """

        super().__init__()

        self.ddwconv = CDilated(dim, dim, kSize=k, stride=stride, groups=dim, d=dilation)
        self.bn1 = nn.BatchNorm2d(dim)

        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x

        x = self.ddwconv(x)
        x = self.bn1(x)

        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)

        return x

# 原版LGFI
# class LGFI(nn.Module):
#     """
#     Local-Global Features Interaction
#     """
#     def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=6,
#                  use_pos_emb=True, num_heads=6, qkv_bias=True, attn_drop=0., drop=0.):
#         super().__init__()

#         self.dim = dim
#         self.pos_embd = None
#         if use_pos_emb:
#             self.pos_embd = PositionalEncodingFourier(dim=self.dim)

#         self.norm_xca = LayerNorm(self.dim, eps=1e-6)

#         self.gamma_xca = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
#                                       requires_grad=True) if layer_scale_init_value > 0 else None
#         self.xca = XCA(self.dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

#         self.norm = LayerNorm(self.dim, eps=1e-6)
#         self.pwconv1 = nn.Linear(self.dim, expan_ratio * self.dim)
#         self.act = nn.GELU()
#         self.pwconv2 = nn.Linear(expan_ratio * self.dim, self.dim)
#         self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((self.dim)),
#                                   requires_grad=True) if layer_scale_init_value > 0 else None
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def forward(self, x):
#         input_ = x

#         # XCA
#         B, C, H, W = x.shape
#         x = x.reshape(B, C, H * W).permute(0, 2, 1)

#         if self.pos_embd:
#             pos_encoding = self.pos_embd(B, H, W).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
#             x = x + pos_encoding

#         x = x + self.gamma_xca * self.xca(self.norm_xca(x))

#         x = x.reshape(B, H, W, C)

#         # Inverted Bottleneck
#         x = self.norm(x)
#         x = self.pwconv1(x)
#         x = self.act(x)
#         x = self.pwconv2(x)
#         if self.gamma is not None:
#             x = self.gamma * x
#         x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

#         x = input_ + self.drop_path(x)

#         return x


# class LGFI(nn.Module):
#     """
#     Local-Global Features Interaction
#     """
#     def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=6,
#                  use_pos_emb=True, num_heads=6, qkv_bias=True, attn_drop=0., drop=0.):
#         super().__init__()

#         self.dim = dim
#         self.pos_embd = PositionalEncodingFourier(dim=self.dim) if use_pos_emb else None
#         if use_pos_emb:
#             self.pos_embd = PositionalEncodingFourier(dim=self.dim)

#         # MCA 模块配置
#         self.norm_mca = LayerNorm(self.dim, eps=1e-6)
#         self.gamma_mca = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
#                                       requires_grad=True) if layer_scale_init_value > 0 else None
#         self.mca = MCA(dim=dim)   # 万能壳，内部跑原版 MCALayer

#         # 瓶颈层配置
#         self.norm = LayerNorm(self.dim, eps=1e-6)
#         self.pwconv1 = nn.Linear(self.dim, expan_ratio * self.dim)
#         self.act = nn.GELU()
#         self.pwconv2 = nn.Linear(expan_ratio * self.dim, self.dim)
#         self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((self.dim)),
#                                   requires_grad=True) if layer_scale_init_value > 0 else None
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def forward(self, x):
#         input_ = x
#           # MCA 分支
#         B, C, H, W = x.shape
#         x = x.reshape(B, C, H * W).permute(0, 2, 1)          # (B, N, C)

#         if self.pos_embd:
#             pos_encoding = self.pos_embd(B, H, W).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
#             x = x + pos_encoding

#         x = x + self.gamma_mca * self.mca(self.norm_mca(x))  # MCA 输出同形 (B, N, C)

#         x = x.reshape(B, H, W, C)

#         # Inverted Bottleneck
#         x = self.norm(x)
#         x = self.pwconv1(x)
#         x = self.act(x)
#         x = self.pwconv2(x)
#         if self.gamma is not None:
#             x = self.gamma * x
#         x = x.permute(0, 3, 1, 2)                              # (N, H, W, C) -> (N, C, H, W)

#         x = input_ + self.drop_path(x)
#         return x


# class AvgPool(nn.Module):
#     def __init__(self, ratio):
#         super().__init__()
#         self.pool = nn.ModuleList()
#         for i in range(0, ratio):
#             self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

#     def forward(self, x):
#         for pool in self.pool:
#             x = pool(x)

#         return x


# mca换xca,性能反降，猜测是“特征拉成 (B,N,C) 才喂给 MCA，导致 MCA 的 4D 卷积门控在空间维度上错位或退化
#v1.0
# class LGFI(nn.Module):
#     """
#     Local-Global Features Interaction
#     """
#     def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=6,
#                  use_pos_emb=True, num_heads=6, qkv_bias=True, attn_drop=0., drop=0.):
#         super().__init__()

#         self.dim = dim
#         self.pos_embd = PositionalEncodingFourier(dim=self.dim) if use_pos_emb else None
#         if use_pos_emb:
#             self.pos_embd = PositionalEncodingFourier(dim=self.dim)

#         # MCA 模块配置
#         self.norm_mca = LayerNorm(self.dim, eps=1e-6)
#         self.gamma_mca = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
#                                       requires_grad=True) if layer_scale_init_value > 0 else None
#         self.mca = MCA(dim=dim)   # 万能壳，内部跑原版 MCALayer

#         # 瓶颈层配置
#         self.norm = LayerNorm(self.dim, eps=1e-6)
#         self.pwconv1 = nn.Linear(self.dim, expan_ratio * self.dim)
#         self.act = nn.GELU()
#         self.pwconv2 = nn.Linear(expan_ratio * self.dim, self.dim)
#         self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((self.dim)),
#                                   requires_grad=True) if layer_scale_init_value > 0 else None
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def forward(self, x):
#         input_ = x
#           # MCA 分支
#         B, C, H, W = x.shape
#         x = x.reshape(B, C, H * W).permute(0, 2, 1)          # (B, N, C)

#         if self.pos_embd:
#             pos_encoding = self.pos_embd(B, H, W).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
#             x = x + pos_encoding

#         x = x + self.gamma_mca * self.mca(self.norm_mca(x))  # MCA 输出同形 (B, N, C)

#         x = x.reshape(B, H, W, C)

#         # Inverted Bottleneck
#         x = self.norm(x)
#         x = self.pwconv1(x)
#         x = self.act(x)
#         x = self.pwconv2(x)
#         if self.gamma is not None:
#             x = self.gamma * x
#         x = x.permute(0, 3, 1, 2)                              # (N, H, W, C) -> (N, C, H, W)

#         x = input_ + self.drop_path(x)
#         return x

# v1.4
# class LGFI(nn.Module):
#     """
#     Local-Global Feature Interaction with MCA (纯 4D 版)
#     - 位置编码直接加在 (B,C,H,W)
#     - 通道归一化用 channels_first
#     - gamma 正确广播到 (1,C,1,1)
#     - Inverted Bottleneck 仍走 channels_last 分支
#     """
#     def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=6,
#                  use_pos_emb=True, num_heads=6, qkv_bias=True, attn_drop=0., drop=0.):
#         super().__init__()
#         self.dim = dim

#         # 位置编码：(B,C,H,W)
#         self.pos_embd = PositionalEncodingFourier(dim=self.dim) if use_pos_emb else None

#         # 4D 注意力分支
#         self.norm_mca = LayerNorm(self.dim, eps=1e-6, data_format="channels_first")
#         self.gamma_mca = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
#                                       requires_grad=True) if layer_scale_init_value > 0 else None
#         self.mca = MCA(dim=dim)

#         # Inverted Bottleneck（channels_last）
#         self.norm = LayerNorm(self.dim, eps=1e-6)   # channels_last
#         self.pwconv1 = nn.Linear(self.dim, expan_ratio * self.dim)
#         self.act = nn.GELU()
#         self.pwconv2 = nn.Linear(expan_ratio * self.dim, self.dim)
#         self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
#                                   requires_grad=True) if layer_scale_init_value > 0 else None

#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def forward(self, x):
#         # x: (B,C,H,W)
#         identity = x
#         B, C, H, W = x.shape

#         # 位置编码
#         if self.pos_embd is not None:
#             x = x + self.pos_embd(B, H, W)  # (B,C,H,W)

#         # 4D 注意力
#         x_norm = self.norm_mca(x)
#         y = self.mca(x_norm)  # (B,C,H,W)
#         # 形状强校验（正常不会触发）
#         if y.shape != x.shape:
#             raise RuntimeError(f"LGFI: MCA 输出 {y.shape} 与输入 {x.shape} 不一致")

#         # 按通道维广播 gamma
#         if self.gamma_mca is not None:
#             x = x + self.gamma_mca.view(1, -1, 1, 1) * y
#         else:
#             x = x + y

#         # Inverted Bottleneck（与原版一致）
#         x = x.permute(0, 2, 3, 1)          # (B,C,H,W) -> (B,H,W,C)
#         x = self.norm(x)
#         x = self.pwconv1(x)
#         x = self.act(x)
#         x = self.pwconv2(x)
#         if self.gamma is not None:
#             x = self.gamma * x             # (B,H,W,C)，沿 C 广播
#         x = x.permute(0, 3, 1, 2)          # (B,H,W,C) -> (B,C,H,W)

#         # 残差
#         x = identity + self.drop_path(x)
#         return x
# v1.4.5  XCA长程为主，轻量化MCA（不含空间通道等）局部为辅
class LGFI(nn.Module):
    """
    最终混合方案：XCA(全局) + MCA(局部增强)
    关键：XCA为主，MCA为辅
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=6,
                 use_pos_emb=True, num_heads=6, qkv_bias=True, attn_drop=0., drop=0.):
        super().__init__()
        self.dim = dim

        # ===== 位置编码 =====
        self.pos_embd = PositionalEncodingFourier(dim=self.dim) if use_pos_emb else None

        # ===== 主分支：XCA（保留原版全局能力） =====
        self.norm_xca = LayerNorm(self.dim, eps=1e-6)
        self.gamma_xca = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
                                      requires_grad=True) if layer_scale_init_value > 0 else None
        self.xca = XCA(self.dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                       attn_drop=attn_drop, proj_drop=drop)

        # ===== 辅助分支：轻量MCA（仅增强局部细节） =====
        self.norm_mca = LayerNorm(self.dim, eps=1e-6, data_format="channels_first")
        self.gamma_mca = nn.Parameter(0.5 * layer_scale_init_value * torch.ones(self.dim),
                                      requires_grad=True) if layer_scale_init_value > 0 else None
        # 关键：轻量MCA（无空间分支，降低计算量）
        self.mca_light = MCALayer(channels=dim, no_spatial=True)  # 仅h_cw + w_hc

        # ===== Inverted Bottleneck =====
        self.norm = LayerNorm(self.dim, eps=1e-6)
        self.pwconv1 = nn.Linear(self.dim, expan_ratio * self.dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expan_ratio * self.dim, self.dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        identity = x
        B, C, H, W = x.shape

        # ===== 位置编码（3D，与原版一致） =====
        x_flat = x.reshape(B, C, H * W).permute(0, 2, 1)
        if self.pos_embd is not None:
            pos_encoding = self.pos_embd(B, H, W).reshape(B, -1, x_flat.shape[1]).permute(0, 2, 1)
            x_flat = x_flat + pos_encoding

        # ===== 主分支：XCA全局注意力 =====
        x_flat = x_flat + self.gamma_xca * self.xca(self.norm_xca(x_flat))
        x = x_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)  # 转回4D

        # ===== 辅助分支：MCA局部增强（权重减半） =====
        x_mca_norm = self.norm_mca(x)
        x_mca_out = self.mca_light(x_mca_norm)
        if self.gamma_mca is not None:
            x = x + self.gamma_mca.view(1, -1, 1, 1) * x_mca_out

        # ===== Inverted Bottleneck =====
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)

        x = identity + self.drop_path(x)
        return x
# # v3.6,并行MCA,XCA
# class LGFI(nn.Module):
#     """自适应融合：网络自学习MCA和XCA的最佳比例"""
#     def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=6,
#                  use_pos_emb=True, num_heads=6, qkv_bias=True, attn_drop=0., drop=0.):
#         super().__init__()
#         self.dim = dim

#         self.pos_embd = PositionalEncodingFourier(dim=self.dim) if use_pos_emb else None

#         # MCA分支
#         self.norm_mca = LayerNorm(self.dim, eps=1e-6, data_format="channels_first")
#         self.gamma_mca = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
#                                       requires_grad=True) if layer_scale_init_value > 0 else None
#         self.mca = MCA(dim=dim)

#         # XCA分支
#         self.norm_xca = LayerNorm(self.dim, eps=1e-6)
#         self.gamma_xca = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
#                                       requires_grad=True) if layer_scale_init_value > 0 else None
#         self.xca = XCA(self.dim, num_heads=num_heads, 
#                        qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

#         # ===== 关键：自适应门控 =====
#         self.fusion_gate = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(dim, dim // 4, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(dim // 4, 1, 1),
#             nn.Sigmoid()
#         )

#         # Inverted Bottleneck
#         self.norm = LayerNorm(self.dim, eps=1e-6)
#         self.pwconv1 = nn.Linear(self.dim, expan_ratio * self.dim)
#         self.act = nn.GELU()
#         self.pwconv2 = nn.Linear(expan_ratio * self.dim, self.dim)
#         self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
#                                   requires_grad=True) if layer_scale_init_value > 0 else None

#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def forward(self, x):
#         identity = x
#         B, C, H, W = x.shape

#         if self.pos_embd is not None:
#             x = x + self.pos_embd(B, H, W)

#         # 并行计算
#         x_mca_norm = self.norm_mca(x)
#         feat_mca = self.mca(x_mca_norm)
#         if self.gamma_mca is not None:
#             feat_mca = self.gamma_mca.view(1, -1, 1, 1) * feat_mca

#         x_for_xca = x.reshape(B, C, H * W).permute(0, 2, 1)
#         feat_xca = self.xca(self.norm_xca(x_for_xca))
#         if self.gamma_xca is not None:
#             feat_xca = self.gamma_xca * feat_xca
#         feat_xca = feat_xca.reshape(B, H, W, C).permute(0, 3, 1, 2)

#         # ===== 自适应融合（网络学习最佳权重） =====
#         gate = self.fusion_gate(x)  # (B, 1, 1, 1)
#         feat_fused = gate * feat_mca + (1 - gate) * feat_xca
#         x = x + feat_fused

#         # Inverted Bottleneck
#         x = x.permute(0, 2, 3, 1)
#         x = self.norm(x)
#         x = self.pwconv1(x)
#         x = self.act(x)
#         x = self.pwconv2(x)
#         if self.gamma is not None:
#             x = self.gamma * x
#         x = x.permute(0, 3, 1, 2)

#         x = identity + self.drop_path(x)

#         return x

# v3.5 ,混合混合注意力LGFI，基于V1.4, abs=1.09
# class LGFI(nn.Module):
#     """
#     混合注意力：MCA(局部) + XCA(跨通道全局)
#     关键修改：保持XCA头数与预训练一致，确保权重可加载
#     """
#     def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=6,
#                  use_pos_emb=True, num_heads=6, qkv_bias=True, attn_drop=0., drop=0.):
#         super().__init__()
#         self.dim = dim

#         # ===== 位置编码（4D） =====
#         self.pos_embd = PositionalEncodingFourier(dim=self.dim) if use_pos_emb else None

#         # ===== 分支1：MCA（4D局部） =====
#         self.norm_mca = LayerNorm(self.dim, eps=1e-6, data_format="channels_first")
#         self.gamma_mca = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
#                                       requires_grad=True) if layer_scale_init_value > 0 else None
#         self.mca = MCA(dim=dim)

#         # ===== 分支2：XCA（3D跨通道） =====
#         self.norm_xca = LayerNorm(self.dim, eps=1e-6)
#         self.gamma_xca = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
#                                       requires_grad=True) if layer_scale_init_value > 0 else None
#         # 关键：保持num_heads不变，确保预训练权重可加载
#         self.xca = XCA(self.dim, num_heads=num_heads, 
#                        qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

#         # ===== Inverted Bottleneck =====
#         self.norm = LayerNorm(self.dim, eps=1e-6)
#         self.pwconv1 = nn.Linear(self.dim, expan_ratio * self.dim)
#         self.act = nn.GELU()
#         self.pwconv2 = nn.Linear(expan_ratio * self.dim, self.dim)
#         self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
#                                   requires_grad=True) if layer_scale_init_value > 0 else None

#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def forward(self, x):
#         identity = x
#         B, C, H, W = x.shape

#         # ===== 位置编码 =====
#         if self.pos_embd is not None:
#             x = x + self.pos_embd(B, H, W)

#         # ===== MCA分支（4D） =====
#         x_mca_norm = self.norm_mca(x)
#         x_mca_out = self.mca(x_mca_norm)
        
#         if x_mca_out.shape != x.shape:
#             raise RuntimeError(f"MCA输出 {x_mca_out.shape} 与输入 {x.shape} 不一致")
        
#         if self.gamma_mca is not None:
#             x = x + self.gamma_mca.view(1, -1, 1, 1) * x_mca_out
#         else:
#             x = x + x_mca_out

#         # ===== XCA分支（3D） =====
#         x_for_xca = x.reshape(B, C, H * W).permute(0, 2, 1)
#         x_xca_out = self.xca(self.norm_xca(x_for_xca))
#         x_for_xca = x_for_xca + self.gamma_xca * x_xca_out
#         x = x_for_xca.reshape(B, H, W, C)

#         # ===== Inverted Bottleneck =====
#         x = self.norm(x)
#         x = self.pwconv1(x)
#         x = self.act(x)
#         x = self.pwconv2(x)
#         if self.gamma is not None:
#             x = self.gamma * x
#         x = x.permute(0, 3, 1, 2)

#         # ===== 最终残差 =====
#         x = identity + self.drop_path(x)

#         return x
        
# v3.0 混合注意力LGFI  （trans+MCA
# class LGFI(nn.Module):
#     """
#     混合局部-全局特征交互（MCA + 轻量Transformer）
#     - MCA: 快速局部多维注意力
#     - Lightweight Transformer: 关键长程依赖（仅在低分辨率特征）
#     - 自适应门控融合
#     """
#     def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=6,
#                  use_pos_emb=True, num_heads=4, qkv_bias=True, attn_drop=0., drop=0.,
#                  use_global=True):  # 新增：控制是否启用全局分支
#         super().__init__()
#         self.dim = dim
#         self.use_global = use_global

#         # ===== 位置编码 =====
#         self.pos_embd = PositionalEncodingFourier(dim=self.dim) if use_pos_emb else None

#         # ===== 局部分支：MCA =====
#         self.norm_local = LayerNorm(self.dim, eps=1e-6, data_format="channels_first")
#         self.gamma_local = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
#                                        requires_grad=True) if layer_scale_init_value > 0 else None
#         self.mca = MCA(dim=dim)

#         # ===== 全局分支：轻量Transformer（可选） =====
#         if self.use_global:
#             self.norm_global = LayerNorm(self.dim, eps=1e-6, data_format="channels_first")
#             self.gamma_global = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
#                                            requires_grad=True) if layer_scale_init_value > 0 else None
            
#             # 轻量多头自注意力（仅4头，降低计算量）
#             self.global_attn = nn.MultiheadAttention(
#                 embed_dim=dim,
#                 num_heads=num_heads,  # 默认4
#                 dropout=attn_drop,
#                 bias=qkv_bias,
#                 batch_first=True
#             )
            
#             # 门控融合
#             self.fusion_gate = nn.Sequential(
#                 nn.AdaptiveAvgPool2d(1),  # (B,C,H,W) -> (B,C,1,1)
#                 nn.Conv2d(dim * 2, dim, 1),
#                 nn.Sigmoid()
#             )

#         # ===== Inverted Bottleneck（保持原版） =====
#         self.norm = LayerNorm(self.dim, eps=1e-6)  # channels_last
#         self.pwconv1 = nn.Linear(self.dim, expan_ratio * self.dim)
#         self.act = nn.GELU()
#         self.pwconv2 = nn.Linear(expan_ratio * self.dim, self.dim)
#         self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
#                                   requires_grad=True) if layer_scale_init_value > 0 else None

#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def forward(self, x):
#         identity = x
#         B, C, H, W = x.shape

#         # ===== 位置编码 =====
#         if self.pos_embd is not None:
#             x = x + self.pos_embd(B, H, W)

#         # ===== 局部分支：MCA =====
#         x_local_norm = self.norm_local(x)
#         feat_local = self.mca(x_local_norm)
#         if self.gamma_local is not None:
#             feat_local = self.gamma_local.view(1, -1, 1, 1) * feat_local

#         # ===== 全局分支：Transformer（仅在特征图较小时） =====
#         if self.use_global and H * W <= 1600:  # 关键优化：仅40×40以下启用
#             x_global_norm = self.norm_global(x)
            
#             # (B,C,H,W) -> (B,H*W,C)
#             x_flat = x_global_norm.flatten(2).transpose(1, 2)
            
#             # 自注意力
#             attn_out, _ = self.global_attn(x_flat, x_flat, x_flat)
            
#             # (B,H*W,C) -> (B,C,H,W)
#             feat_global = attn_out.transpose(1, 2).reshape(B, C, H, W)
            
#             if self.gamma_global is not None:
#                 feat_global = self.gamma_global.view(1, -1, 1, 1) * feat_global
            
#             # 自适应融合
#             gate = self.fusion_gate(torch.cat([feat_local, feat_global], dim=1))
#             feat_fused = gate * feat_global + (1 - gate) * feat_local
            
#         else:
#             # 高分辨率特征仅用MCA（保持速度）
#             feat_fused = feat_local

#         x = x + feat_fused

#         # ===== Inverted Bottleneck（与原版一致） =====
#         x = x.permute(0, 2, 3, 1)  # (B,C,H,W) -> (B,H,W,C)
#         x_norm = self.norm(x)
#         x_ffn = self.pwconv1(x_norm)
#         x_ffn = self.act(x_ffn)
#         x_ffn = self.pwconv2(x_ffn)
#         if self.gamma is not None:
#             x_ffn = self.gamma * x_ffn
#         x = x.permute(0, 3, 1, 2)  # (B,H,W,C) -> (B,C,H,W)

#         x = identity + self.drop_path(x)
#         return x

#v1.2 wy
# class LGFI(nn.Module):
#     """
#     Local-Global Feature Interaction with MCA (纯 4D 版)
#     - 位置编码直接加在 (B,C,H,W)
#     - 通道归一化用 channels_first
#     - gamma 正确广播到 (1,C,1,1)
#     - Inverted Bottleneck 仍走 channels_last 分支
#     """
#     def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=6,
#                  use_pos_emb=True, num_heads=6, qkv_bias=True, attn_drop=0., drop=0.):
#         super().__init__()
#         self.dim = dim

#         # 位置编码：(B,C,H,W)
#         self.pos_embd = PositionalEncodingFourier(dim=self.dim) if use_pos_emb else None

#         # 4D 注意力分支
#         self.norm_mca = LayerNorm(self.dim, eps=1e-6, data_format="channels_first")
#         self.gamma_mca = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
#                                       requires_grad=True) if layer_scale_init_value > 0 else None
#         self.mca = MCA(dim=dim)
       

#         # Inverted Bottleneck（channels_last）
#         self.norm = LayerNorm(self.dim, eps=1e-6)   # channels_last
#         self.pwconv1 = nn.Linear(self.dim, expan_ratio * self.dim)
#         self.act = nn.GELU()
#         self.pwconv2 = nn.Linear(expan_ratio * self.dim, self.dim)
#         self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
#                                   requires_grad=True) if layer_scale_init_value > 0 else None

#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def forward(self, x):
#         # x: (B,C,H,W)

#         # wty-
#         identity = x

#         B, C, H, W = x.shape

#         # wty-
#         # # 位置编码
#         # if self.pos_embd is not None:
#         #     x = x + self.pos_embd(B, H, W)  # (B,C,H,W)

#         # wty+  4D 注意力
#         x_norm = self.norm_mca(x)
#         y = self.mca(x_norm)  # (B,C,H,W)

        
        
#        x = x.reshape(B, H, W, C)
#         # wty-
#         # 形状强校验（正常不会触发）
#         if y.shape != x.shape:
#             raise RuntimeError(f"LGFI: MCA 输出 {y.shape} 与输入 {x.shape} 不一致")

#         # 按通道维广播 gamma
#         if self.gamma_mca is not None:
#             x = x + self.gamma_mca.view(1, -1, 1, 1) * y
#         else:
#             x = x + y

#         # Inverted Bottleneck（与原版一致）
#         x = x.permute(0, 2, 3, 1)          # (B,C,H,W) -> (B,H,W,C)
#         x = self.norm(x)
#         x = self.pwconv1(x)
#         x = self.act(x)
#         x = self.pwconv2(x)
#         if self.gamma is not None:
#             x = self.gamma * x             # (B,H,W,C)，沿 C 广播
#         x = x.permute(0, 3, 1, 2)          # (B,H,W,C) -> (B,C,H,W)

#         # 残差,wty change
#         x = identity + self.drop_path(x)
#         # x = input_ + self.drop_path(x)

#         return x

class AvgPool(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, ratio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        for pool in self.pool:
            x = pool(x)

        return x

class LiteMono(nn.Module):
    """
    Lite-Mono
    """
    def __init__(self, in_chans=3, model='lite-mono', height=192, width=640,
                 global_block=[1, 1, 1], global_block_type=['LGFI', 'LGFI', 'LGFI'],
                 drop_path_rate=0.2, layer_scale_init_value=1e-6, expan_ratio=6,
                 heads=[8, 8, 8], use_pos_embd_xca=[True, False, False], **kwargs):

        super().__init__()

        if model == 'lite-mono':
            self.num_ch_enc = np.array([48, 80, 128])
            self.depth = [4, 4, 10]
            self.dims = [48, 80, 128]
            if height == 192 and width == 640:
                self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 1, 2, 3, 2, 4, 6]]
            elif height == 320 and width == 1024:
                self.dilation = [[1, 2, 5], [1, 2, 5], [1, 2, 5, 1, 2, 5, 2, 4, 10]]

        elif model == 'lite-mono-small':
            self.num_ch_enc = np.array([48, 80, 128])
            self.depth = [4, 4, 7]
            self.dims = [48, 80, 128]
            if height == 192 and width == 640:
                self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 2, 4, 6]]
            elif height == 320 and width == 1024:
                self.dilation = [[1, 2, 5], [1, 2, 5], [1, 2, 5, 2, 4, 10]]

        elif model == 'lite-mono-tiny':
            self.num_ch_enc = np.array([32, 64, 128])
            self.depth = [4, 4, 7]
            self.dims = [32, 64, 128]
            if height == 192 and width == 640:
                self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 2, 4, 6]]
            elif height == 320 and width == 1024:
                self.dilation = [[1, 2, 5], [1, 2, 5], [1, 2, 5, 2, 4, 10]]

        elif model == 'lite-mono-8m':
            self.num_ch_enc = np.array([64, 128, 224])
            self.depth = [4, 4, 10]
            self.dims = [64, 128, 224]
            if height == 192 and width == 640:
                self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 1, 2, 3, 2, 4, 6]]
            elif height == 320 and width == 1024:
                self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 1, 2, 3, 2, 4, 6]]

        for g in global_block_type:
            assert g in ['None', 'LGFI']

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem1 = nn.Sequential(
            Conv(in_chans, self.dims[0], kSize=3, stride=2, padding=1, bn_act=True),
            Conv(self.dims[0], self.dims[0], kSize=3, stride=1, padding=1, bn_act=True),
            Conv(self.dims[0], self.dims[0], kSize=3, stride=1, padding=1, bn_act=True),
        )

        self.stem2 = nn.Sequential(
            Conv(self.dims[0]+3, self.dims[0], kSize=3, stride=2, padding=1, bn_act=False),
        )

        self.downsample_layers.append(stem1)

        self.input_downsample = nn.ModuleList()
        for i in range(1, 5):
            self.input_downsample.append(AvgPool(i))

        for i in range(2):
            downsample_layer = nn.Sequential(
                Conv(self.dims[i]*2+3, self.dims[i+1], kSize=3, stride=2, padding=1, bn_act=False),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depth))]
        cur = 0
        for i in range(3):
            stage_blocks = []
            for j in range(self.depth[i]):
                if j > self.depth[i] - global_block[i] - 1:
                    if global_block_type[i] == 'LGFI':
                        stage_blocks.append(LGFI(dim=self.dims[i], drop_path=dp_rates[cur + j],
                                                 expan_ratio=expan_ratio,
                                                 use_pos_emb=use_pos_embd_xca[i], num_heads=heads[i],
                                                 layer_scale_init_value=layer_scale_init_value,
                                                 ))

                    else:
                        raise NotImplementedError
                else:
                    stage_blocks.append(DilatedConv(dim=self.dims[i], k=3, dilation=self.dilation[i][j], drop_path=dp_rates[cur + j],
                                                    layer_scale_init_value=layer_scale_init_value,
                                                    expan_ratio=expan_ratio))

            self.stages.append(nn.Sequential(*stage_blocks))
            cur += self.depth[i]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        features = []
        x = (x - 0.45) / 0.225

        x_down = []
        for i in range(4):
            x_down.append(self.input_downsample[i](x))

        tmp_x = []
        x = self.downsample_layers[0](x)
        x = self.stem2(torch.cat((x, x_down[0]), dim=1))
        tmp_x.append(x)

        for s in range(len(self.stages[0])-1):
            x = self.stages[0][s](x)
        x = self.stages[0][-1](x)
        tmp_x.append(x)
        features.append(x)

        for i in range(1, 3):
            tmp_x.append(x_down[i])
            x = torch.cat(tmp_x, dim=1)
            x = self.downsample_layers[i](x)

            tmp_x = [x]
            for s in range(len(self.stages[i]) - 1):
                x = self.stages[i][s](x)
            x = self.stages[i][-1](x)
            tmp_x.append(x)

            features.append(x)

        return features

    def forward(self, x):
        x = self.forward_features(x)

        return x
