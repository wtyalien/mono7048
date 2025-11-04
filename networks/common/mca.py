# import torch
# from torch import nn
# import math
# __all__ = ['MCALayer', 'MCAGate', 'MCA']     # 仅末尾多导出 MCA


# class StdPool(nn.Module):
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         std = x.view(b, c, -1).std(dim=2, keepdim=True)
#         std = std.reshape(b, c, 1, 1)
#         return std


# class MCAGate(nn.Module):
#     def __init__(self, k_size, pool_types=('avg', 'std')):
#         super().__init__()
#         self.pools = nn.ModuleList()
#         for pool_type in pool_types:
#             if pool_type == 'avg':
#                 self.pools.append(nn.AdaptiveAvgPool2d(1))
#             elif pool_type == 'std':
#                 self.pools.append(StdPool())
#             else:
#                 raise NotImplementedError

#         self.conv = nn.Conv2d(1, 1, kernel_size=(1, k_size), stride=1,
#                               padding=(0, (k_size - 1) // 2), bias=False)
#         self.sigmoid = nn.Sigmoid()
#         self.weight = nn.Parameter(torch.rand(2))

#     def forward(self, x):
#         feats = [pool(x) for pool in self.pools]
#         weight = torch.sigmoid(self.weight)
#         out = 0.5 * (feats[0] + feats[1]) + weight[0] * feats[0] + weight[1] * feats[1]
#         out = out.permute(0, 3, 2, 1).contiguous()
#         out = self.conv(out)
#         out = out.permute(0, 3, 2, 1).contiguous()
#         out = self.sigmoid(out).expand_as(x)
#         return x * out


# class MCALayer(nn.Module):
#     def __init__(self, inp, no_spatial=False):
#         super().__init__()
#         lambd, gamma = 1.5, 1
#         temp = round(abs((math.log2(inp) - gamma) / lambd))
#         kernel = temp if temp % 2 else temp - 1

#         self.h_cw = MCAGate(3)
#         self.w_hc = MCAGate(3)
#         self.no_spatial = no_spatial
#         if not no_spatial:
#             self.c_hw = MCAGate(kernel)

#     def forward(self, x):
#         x_h = x.permute(0, 2, 1, 3).contiguous()
#         x_h = self.h_cw(x_h)
#         x_h = x_h.permute(0, 2, 1, 3).contiguous()

#         x_w = x.permute(0, 3, 2, 1).contiguous()
#         x_w = self.w_hc(x_w)
#         x_w = x_w.permute(0, 3, 2, 1).contiguous()

#         if not self.no_spatial:
#             x_c = self.c_hw(x)
#             x_out = (x_c + x_h + x_w) / 3.0
#         else:
#             x_out = (x_h + x_w) / 2.0
#         return x_out


# # ====== 2. 万能形状适配器（不改动上面任何一行） ======
# class MCA(nn.Module):
#     """3-D/4-D 通用入口，内部永远跑原版 4-D MCALayer"""
#     def __init__(self, dim, **kwargs):
#         super().__init__()
#         self.layer = MCALayer(inp=dim, no_spatial=False)

#     def forward(self, x):
#         # 1. 3-D → 4-D
#         if x.dim() == 3:
#             B, N, C = x.shape
#             # 用真实 H,W 反推（整除 N）
#             H = int(torch.tensor(N).float().sqrt().ceil())
#             while H * (N // H) != N:
#                 H -= 1
#             W = N // H
#             x4 = x.transpose(1, 2).view(B, C, H, W)
#         else:                       # 4-D 直接进入
#             x4 = x
#             B, C, H, W = x4.shape
#             N = H * W

#         # 2. 原版 MCA（4-D）
#         out4 = self.layer(x4)

#         # 3. 输出与输入同形
#         if x.dim() == 3:
#             return out4.flatten(2).transpose(1, 2)   # 回到 3-D
#         return out4                                  # 保持 4-D

# # if __name__ == '__main__':
# #     x = torch.randn(4, 512, 7, 7)
# #     model = MCALayer(512)
# #     output = model(x)
# #     print(output.shape)

# v1.0与v1.1
# import math
# import torch
# from torch import nn

# __all__ = ["MCAGate", "MCALayer", "MCA"]

# class StdPool(nn.Module):
#     def forward(self, x):
#         # x: (B,C,H,W) → (B,C,1,1) 的标准差
#         b, c, h, w = x.size()
#         std = x.view(b, c, -1).std(dim=2, keepdim=True)  # (B,C,1)
#         return std.view(b, c, 1, 1)

# class MCAGate(nn.Module):
#     """
#     通道-空间门控：对 (B,C,H,W) 做池化得到 (B,C,1,1)，
#     然后把 C 当作“宽度”做 1xk conv，再 sigmoid 回乘。
#     """
#     def __init__(self, k_size: int, pool_types=('avg', 'std')):
#         super().__init__()
#         assert k_size >= 3 and k_size % 2 == 1, "k_size 必须为 >=3 的奇数"
#         self.pools = nn.ModuleList()
#         for p in pool_types:
#             if p == "avg":
#                 self.pools.append(nn.AdaptiveAvgPool2d(1))
#             elif p == "std":
#                 self.pools.append(StdPool())
#             else:
#                 raise NotImplementedError(f"unsupported pool: {p}")

#         # 把通道维当作“宽度”做 1xk 卷积（in_ch=1）
#         self.conv = nn.Conv2d(1, 1, kernel_size=(1, k_size),
#                               stride=1, padding=(0, (k_size - 1) // 2), bias=False)
#         self.sigmoid = nn.Sigmoid()
#         self.weight = nn.Parameter(torch.zeros(len(pool_types)))  # 初始均衡

#     def forward(self, x):
#         # x: (B,C,H,W)
#         feats = [pool(x) for pool in self.pools]          # 每个都是 (B,C,1,1)
#         w = torch.sigmoid(self.weight)                    # 可学加权
#         base = sum(w[i] * feats[i] for i in range(len(feats)))
#         # 也可加入均值作为先验：0.5*(avg+std) + w0*avg + w1*std
#         if len(feats) == 2:
#             base = 0.5 * (feats[0] + feats[1]) + w[0] * feats[0] + w[1] * feats[1]

#         # 将通道维挪到“宽度”维做 1xk 卷积
#         y = base.permute(0, 3, 2, 1).contiguous()         # (B,1,1,C)
#         y = self.conv(y)                                  # (B,1,1,C)
#         y = y.permute(0, 3, 2, 1).contiguous()            # (B,C,1,1)
#         y = self.sigmoid(y)
#         return x * y.expand_as(x)                         # 通道注意力（含跨通道依赖）

# class MCALayer(nn.Module):
#     """
#     原子 MCA 层：H/C/W 三个互补门控（含空间与通道交互）。
#     输入输出保持 (B,C,H,W)。
#     """
#     def __init__(self, channels: int, no_spatial: bool = False):
#         super().__init__()
#         # 自适应确定空间核大小，至少为奇数且 >=3
#         lambd, gamma = 1.5, 1.0
#         temp = int(round(abs((math.log2(max(channels, 2)) - gamma) / lambd)))
#         k_spatial = max(3, temp if temp % 2 == 1 else temp + 1)

#         # 三个方向的门控
#         self.h_cw = MCAGate(3)           # 沿 H 展开时的 C↔W 交互（通过通道门控体现）
#         self.w_hc = MCAGate(3)           # 沿 W 展开时的 H↔C 交互
#         self.no_spatial = no_spatial
#         if not no_spatial:
#             self.c_hw = MCAGate(k_spatial)  # 显式空间门控（H/W）

#     def forward(self, x):
#         # x: (B,C,H,W)
#         # H-branch：把 H 挪到通道位置做门控（等价于通道-宽度交互的变体）
#         x_h = x.permute(0, 2, 1, 3).contiguous()  # (B,H,C,W)
#         x_h = self.h_cw(x_h)
#         x_h = x_h.permute(0, 2, 1, 3).contiguous()

#         # W-branch：把 W 挪到通道位置
#         x_w = x.permute(0, 3, 2, 1).contiguous()  # (B,W,H,C)
#         x_w = self.w_hc(x_w)
#         x_w = x_w.permute(0, 3, 2, 1).contiguous()

#         if not self.no_spatial:
#             # 直接在 (B,C,H,W) 上做空间门控
#             x_c = self.c_hw(x)
#             out = (x_c + x_h + x_w) / 3.0
#         else:
#             out = (x_h + x_w) / 2.0
#         return out

# class MCA(nn.Module):
#     """
#     安全版 MCA 封装：仅接受/返回 (B,C,H,W)，不做 3D flatten。
#     """
#     def __init__(self, dim: int, **kwargs):
#         super().__init__()
#         self.layer = MCALayer(channels=dim, no_spatial=False)

#     def forward(self, x):
#         assert x.dim() == 4, f"MCA 仅支持 4D (B,C,H,W)，但得到 {x.shape}"
#         y = self.layer(x)
#         # 强约束：输出必须同形
#         if y.shape != x.shape:
#             raise RuntimeError(f"MCA 输出形状 {y.shape} 与输入 {x.shape} 不一致")
#         return y

# v1.2,性能反而下降
# 删除了一些conv操作
# Dropout 被添加到 MCAGate 中，以防止过拟合。
# 残差连接 被添加到 MCALayer 中，通过 Identity() 保持输入不变并与输出相加，提升了模型的训练效果。

# import math
# import torch
# from torch import nn

# __all__ = ["MCAGate", "MCALayer", "MCA"]

# class StdPool(nn.Module):
#     def forward(self, x):
#         # x: (B,C,H,W) → (B,C,1,1) 的标准差
#         b, c, h, w = x.size()
#         std = x.view(b, c, -1).std(dim=2, keepdim=True)  # (B,C,1)
#         return std.view(b, c, 1, 1)

# class MCAGate(nn.Module):
#     """
#     通道-空间门控：对 (B,C,H,W) 做池化得到 (B,C,1,1)，
#     然后把 C 当作“宽度”做 1xk conv，再 sigmoid 回乘。
#     """
#     def __init__(self, k_size: int, pool_types=('avg', 'std')):
#         super().__init__()
#         assert k_size >= 3 and k_size % 2 == 1, "k_size 必须为 >=3 的奇数"
#         self.pools = nn.ModuleList()
#         for p in pool_types:
#             if p == "avg":
#                 self.pools.append(nn.AdaptiveAvgPool2d(1))
#             elif p == "std":
#                 self.pools.append(StdPool())
#             else:
#                 raise NotImplementedError(f"unsupported pool: {p}")

#         # v1.1 -> v1.2: 增加了 Dropout 层（防止过拟合）
#         self.sigmoid = nn.Sigmoid()
#         self.weight = nn.Parameter(torch.zeros(len(pool_types)))  # 初始均衡
#         self.dropout = nn.Dropout(0.3)  # 增加 Dropout

#     def forward(self, x):
#         # x: (B,C,H,W)
#         feats = [pool(x) for pool in self.pools]          # 每个都是 (B,C,1,1)
#         w = torch.sigmoid(self.weight)                    # 可学加权
#         base = sum(w[i] * feats[i] for i in range(len(feats)))

#         # v1.1 -> v1.2: 加入 Dropout
#         base = self.dropout(base)  # Dropout 防止过拟合

#         if len(feats) == 2:
#             base = 0.5 * (feats[0] + feats[1]) + w[0] * feats[0] + w[1] * feats[1]

#         # # 通道注意力，增强空间通道之间的交互
#         # y = base.permute(0, 3, 2, 1).contiguous()         # (B,1,1,C)
#         # y = self.sigmoid(y)
#         # return x * y.expand_as(x)                         # 通道注意力（含跨通道依赖）
        
#         # ---- 保留通道注意力，同时修正维度 ----
#         y = base.permute(0, 3, 2, 1).contiguous()         # (B,1,1,C)
#         y = self.sigmoid(y)                               # 通道注意力
#         y = y.permute(0, 3, 2, 1).contiguous()            # 回到 (B,C,1,1)
        
#         return x * y                                      # 自动广播到 (B,C,H,W)
# class MCALayer(nn.Module):
#     """
#     原子 MCA 层：H/C/W 三个互补门控（含空间与通道交互）。
#     输入输出保持 (B,C,H,W)。
#     """
#     def __init__(self, channels: int, no_spatial: bool = False):
#         super().__init__()
#         lambd, gamma = 1.5, 1.0
#         temp = int(round(abs((math.log2(max(channels, 2)) - gamma) / lambd)))
#         k_spatial = max(3, temp if temp % 2 == 1 else temp + 1)

#         # v1.1
#         self.h_cw = MCAGate(3)           # 沿 H 展开时的 C↔W 交互（通过通道门控体现）
#         self.w_hc = MCAGate(3)           # 沿 W 展开时的 H↔C 交互
#         self.no_spatial = no_spatial
#         if not no_spatial:
#             self.c_hw = MCAGate(k_spatial)  # 显式空间门控（H/W）

#         # v1.1 -> v1.2: 增加了残差连接
#         self.residual = nn.Identity()  # 作为残差连接

#     def forward(self, x):
#         # x: (B,C,H,W)
#         x_h = x.permute(0, 2, 1, 3).contiguous()  # (B,H,C,W)
#         x_h = self.h_cw(x_h)
#         x_h = x_h.permute(0, 2, 1, 3).contiguous()

#         # W-branch：把 W 挪到通道位置
#         x_w = x.permute(0, 3, 2, 1).contiguous()  # (B,W,H,C)
#         x_w = self.w_hc(x_w)
#         x_w = x_w.permute(0, 3, 2, 1).contiguous()

#         if not self.no_spatial:
#             x_c = self.c_hw(x)
#             out = (x_c + x_h + x_w) / 3.0
#         else:
#             out = (x_h + x_w) / 2.0

#         # v1.1 -> v1.2: 残差连接增强了特征的传递，帮助缓解梯度问题
#         return out + self.residual(x)

# class MCA(nn.Module):
#     """
#     安全版 MCA 封装：仅接受/返回 (B,C,H,W)，不做 3D flatten。
#     """
#     def __init__(self, dim: int, **kwargs):
#         super().__init__()
#         self.layer = MCALayer(channels=dim, no_spatial=False)

#     def forward(self, x):
#         assert x.dim() == 4, f"MCA 仅支持 4D (B,C,H,W)，但得到 {x.shape}"
#         y = self.layer(x)
#         if y.shape != x.shape:
#             raise RuntimeError(f"MCA 输出形状 {y.shape} 与输入 {x.shape} 不一致")
#         return y

#新v1.2.1
# import math
# import torch
# from torch import nn

# __all__ = ["MCAGate", "MCALayer", "MCA"]

# class StdPool(nn.Module):
#     def forward(self, x):
#         # x: (B,C,H,W) → (B,C,1,1) 的标准差
#         b, c, h, w = x.size()
#         std = x.view(b, c, -1).std(dim=2, keepdim=True)  # (B,C,1)
#         return std.view(b, c, 1, 1)

# class MCAGate(nn.Module):
#     """
#     通道-空间门控：对 (B,C,H,W) 做池化得到 (B,C,1,1)，
#     然后把 C 当作“宽度”做 1xk conv，再 sigmoid 回乘。
#     """
#     def __init__(self, k_size: int, pool_types=('avg', 'std')):
#         super().__init__()
#         assert k_size >= 3 and k_size % 2 == 1, "k_size 必须为 >=3 的奇数"
#         self.pools = nn.ModuleList()
#         for p in pool_types:
#             if p == "avg":
#                 self.pools.append(nn.AdaptiveAvgPool2d(1))
#             elif p == "std":
#                 self.pools.append(StdPool())
#             else:
#                 raise NotImplementedError(f"unsupported pool: {p}")

#         # === v1.2: 保留1xk卷积 + 加入Dropout ===
#         self.conv = nn.Conv2d(1, 1, kernel_size=(1, k_size),
#                               stride=1, padding=(0, (k_size - 1) // 2), bias=False)
#         self.sigmoid = nn.Sigmoid()
#         self.dropout = nn.Dropout(0.3)  # Dropout在卷积后
#         self.weight = nn.Parameter(torch.zeros(len(pool_types)))  # 可学习池化加权

#     def forward(self, x):
#         # x: (B,C,H,W)
#         feats = [pool(x) for pool in self.pools]          
#         w = torch.sigmoid(self.weight)                    
#         base = sum(w[i] * feats[i] for i in range(len(feats)))

#         if len(feats) == 2:
#             base = 0.5 * (feats[0] + feats[1]) + w[0] * feats[0] + w[1] * feats[1]

#         # === 1xk卷积建模通道依赖，卷积后Dropout再Sigmoid ===
#         y = base.permute(0, 3, 2, 1).contiguous()         # (B,1,1,C)
#         y = self.conv(y)                                  # 1xk卷积
#         y = self.dropout(y)                               # Dropout防止过拟合
#         y = y.permute(0, 3, 2, 1).contiguous()            # (B,C,1,1)
#         y = self.sigmoid(y)

#         return x * y.expand_as(x)                         # 通道注意力

# class MCALayer(nn.Module):
#     """
#     原子 MCA 层：H/C/W 三个互补门控（含空间与通道交互）。
#     输入输出保持 (B,C,H,W)。
#     """
#     def __init__(self, channels: int, no_spatial: bool = False):
#         super().__init__()
#         lambd, gamma = 1.5, 1.0
#         temp = int(round(abs((math.log2(max(channels, 2)) - gamma) / lambd)))
#         k_spatial = max(3, temp if temp % 2 == 1 else temp + 1)

#         self.h_cw = MCAGate(3)           
#         self.w_hc = MCAGate(3)           
#         self.no_spatial = no_spatial
#         if not no_spatial:
#             self.c_hw = MCAGate(k_spatial)  

#         # === v1.2: 增加残差连接 ===
#         self.residual = nn.Identity()

#     def forward(self, x):
#         x_h = x.permute(0, 2, 1, 3).contiguous()  
#         x_h = self.h_cw(x_h)
#         x_h = x_h.permute(0, 2, 1, 3).contiguous()

#         x_w = x.permute(0, 3, 2, 1).contiguous()  
#         x_w = self.w_hc(x_w)
#         x_w = x_w.permute(0, 3, 2, 1).contiguous()

#         if not self.no_spatial:
#             x_c = self.c_hw(x)
#             out = (x_c + x_h + x_w) / 3.0
#         else:
#             out = (x_h + x_w) / 2.0

#         # 残差连接
#         return out + self.residual(x)

# class MCA(nn.Module):
#     """
#     安全版 MCA 封装：仅接受/返回 (B,C,H,W)，不做 3D flatten。
#     """
#     def __init__(self, dim: int, **kwargs):
#         super().__init__()
#         self.layer = MCALayer(channels=dim, no_spatial=False)

#     def forward(self, x):
#         assert x.dim() == 4, f"MCA 仅支持 4D (B,C,H,W)，但得到 {x.shape}"
#         y = self.layer(x)
#         if y.shape != x.shape:
#             raise RuntimeError(f"MCA 输出形状 {y.shape} 与输入 {x.shape} 不一致")
#         return y


# v1.4 残差 + LayerScale + DropPath + LayerNorm
import math
import torch
from torch import nn

__all__ = ["MCAGate", "MCALayer", "MCA"]

class StdPool(nn.Module):
    def forward(self, x):
        b, c, h, w = x.size()
        std = x.view(b, c, -1).std(dim=2, keepdim=True)
        return std.view(b, c, 1, 1)

class MCAGate(nn.Module):
    def __init__(self, k_size: int, pool_types=('avg', 'std'), dropout_p=0.1):
        super().__init__()
        assert k_size >= 3 and k_size % 2 == 1
        self.pools = nn.ModuleList()
        for p in pool_types:
            if p == "avg":
                self.pools.append(nn.AdaptiveAvgPool2d(1))
            elif p == "std":
                self.pools.append(StdPool())
            else:
                raise NotImplementedError

        self.conv = nn.Conv2d(1, 1, kernel_size=(1, k_size),
                              stride=1, padding=(0, (k_size - 1) // 2), bias=False)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_p)
        self.weight = nn.Parameter(torch.zeros(len(pool_types)))

    def forward(self, x):
        feats = [pool(x) for pool in self.pools]
        w = torch.sigmoid(self.weight)
        base = sum(w[i] * feats[i] for i in range(len(feats)))

        if len(feats) == 2:
            base = 0.5 * (feats[0] + feats[1]) + w[0] * feats[0] + w[1] * feats[1]

        y = base.permute(0, 3, 2, 1).contiguous()
        y = self.conv(y)
        y = self.dropout(y)
        y = y.permute(0, 3, 2, 1).contiguous()
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class MCALayer(nn.Module):
    def __init__(self, channels: int, no_spatial: bool = False, drop_path=0.1):
        super().__init__()
        lambd, gamma = 1.5, 1.0
        temp = int(round(abs((math.log2(max(channels, 2)) - gamma) / lambd)))
        k_spatial = max(3, temp if temp % 2 == 1 else temp + 1)

        self.h_cw = MCAGate(3)
        self.w_hc = MCAGate(3)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.c_hw = MCAGate(k_spatial)

        # === LayerNorm + LayerScale + DropPath ===
        self.norm = nn.LayerNorm(channels)
        self.layer_scale = nn.Parameter(1e-6 * torch.ones(channels))
        self.drop_path = nn.Dropout(drop_path)

    def forward(self, x):
        shortcut = x
        x_h = x.permute(0, 2, 1, 3).contiguous()
        x_h = self.h_cw(x_h)
        x_h = x_h.permute(0, 2, 1, 3).contiguous()

        x_w = x.permute(0, 3, 2, 1).contiguous()
        x_w = self.w_hc(x_w)
        x_w = x_w.permute(0, 3, 2, 1).contiguous()

        if not self.no_spatial:
            x_c = self.c_hw(x)
            out = (x_c + x_h + x_w) / 3.0
        else:
            out = (x_h + x_w) / 2.0

        # === 残差 + LayerScale + DropPath + LayerNorm ===
        out = self.drop_path(out)
        out = self.layer_scale.view(1, -1, 1, 1) * out
        out = out + shortcut
        out = self.norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return out

class MCA(nn.Module):
    def __init__(self, dim: int, **kwargs):
        super().__init__()
        self.layer = MCALayer(channels=dim, no_spatial=False)

    def forward(self, x):
        assert x.dim() == 4
        y = self.layer(x)
        if y.shape != x.shape:
            raise RuntimeError(f"MCA 输出形状 {y.shape} 与输入 {x.shape} 不一致")
        return y



#v1.4.1 不如v1.4
# -        self.layer_scale = nn.Parameter(1e-6 * torch.ones(channels))
# +        self.layer_scale = nn.Parameter(1e-4 * torch.ones(channels))

# +         x_c = self.drop_path(x_c)
# import math
# import torch
# from torch import nn

# __all__ = ["MCAGate", "MCALayer", "MCA"]

# class StdPool(nn.Module):
#     def forward(self, x):
#         b, c, h, w = x.size()
#         std = x.view(b, c, -1).std(dim=2, keepdim=True)
#         return std.view(b, c, 1, 1)

# class MCAGate(nn.Module):
#     def __init__(self, k_size: int, pool_types=('avg', 'std'), dropout_p=0.1):
#         super().__init__()
#         assert k_size >= 3 and k_size % 2 == 1
#         self.pools = nn.ModuleList()
#         for p in pool_types:
#             if p == "avg":
#                 self.pools.append(nn.AdaptiveAvgPool2d(1))
#             elif p == "std":
#                 self.pools.append(StdPool())
#             else:
#                 raise NotImplementedError

#         self.conv = nn.Conv2d(1, 1, kernel_size=(1, k_size),
#                               stride=1, padding=(0, (k_size - 1) // 2), bias=False)
#         self.sigmoid = nn.Sigmoid()
#         self.dropout = nn.Dropout(dropout_p)
#         self.weight = nn.Parameter(torch.zeros(len(pool_types)))

#     def forward(self, x):
#         feats = [pool(x) for pool in self.pools]
#         w = torch.sigmoid(self.weight)
#         base = sum(w[i] * feats[i] for i in range(len(feats)))

#         if len(feats) == 2:
#             base = 0.5 * (feats[0] + feats[1]) + w[0] * feats[0] + w[1] * feats[1]

#         y = base.permute(0, 3, 2, 1).contiguous()
#         y = self.conv(y)
#         y = self.dropout(y)
#         y = y.permute(0, 3, 2, 1).contiguous()
#         y = self.sigmoid(y)
#         return x * y.expand_as(x)

# class MCALayer(nn.Module):
#     def __init__(self, channels: int, no_spatial: bool = False, drop_path=0.1):
#         super().__init__()
#         lambd, gamma = 1.5, 1.0
#         temp = int(round(abs((math.log2(max(channels, 2)) - gamma) / lambd)))
#         k_spatial = max(3, temp if temp % 2 == 1 else temp + 1)

#         self.h_cw = MCAGate(3)
#         self.w_hc = MCAGate(3)
#         self.no_spatial = no_spatial
#         if not no_spatial:
#             self.c_hw = MCAGate(k_spatial)

#         # === LayerNorm + LayerScale + DropPath ===
#         self.norm = nn.LayerNorm(channels)
#         self.layer_scale = nn.Parameter(1e-4 * torch.ones(channels))
#         self.drop_path = nn.Dropout(drop_path)

#     def forward(self, x):
#         shortcut = x
#         x_h = x.permute(0, 2, 1, 3).contiguous()
#         x_h = self.h_cw(x_h)
#         x_h = x_h.permute(0, 2, 1, 3).contiguous()

#         x_w = x.permute(0, 3, 2, 1).contiguous()
#         x_w = self.w_hc(x_w)
#         x_w = x_w.permute(0, 3, 2, 1).contiguous()

#         if not self.no_spatial:
#             x_c = self.c_hw(x)
#             x_c = self.drop_path(x_c)
#             out = (x_c + x_h + x_w) / 3.0
#         else:
#             out = (x_h + x_w) / 2.0

#         # === 残差 + LayerScale + DropPath + LayerNorm ===
#         out = self.drop_path(out)
#         out = self.layer_scale.view(1, -1, 1, 1) * out
#         out = out + shortcut
#         out = self.norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
#         return out

# class MCA(nn.Module):
#     def __init__(self, dim: int, **kwargs):
#         super().__init__()
#         self.layer = MCALayer(channels=dim, no_spatial=False)

#     def forward(self, x):
#         assert x.dim() == 4
#         y = self.layer(x)
#         if y.shape != x.shape:
#             raise RuntimeError(f"MCA 输出形状 {y.shape} 与输入 {x.shape} 不一致")
#         return y


# # v1.4.2，同v.1.4但trainer中加了EMA
# import math
# import torch
# from torch import nn

# __all__ = ["MCAGate", "MCALayer", "MCA"]

# class StdPool(nn.Module):
#     def forward(self, x):
#         b, c, h, w = x.size()
#         std = x.view(b, c, -1).std(dim=2, keepdim=True)
#         return std.view(b, c, 1, 1)

# class MCAGate(nn.Module):
#     def __init__(self, k_size: int, pool_types=('avg', 'std'), dropout_p=0.1):
#         super().__init__()
#         assert k_size >= 3 and k_size % 2 == 1
#         self.pools = nn.ModuleList()
#         for p in pool_types:
#             if p == "avg":
#                 self.pools.append(nn.AdaptiveAvgPool2d(1))
#             elif p == "std":
#                 self.pools.append(StdPool())
#             else:
#                 raise NotImplementedError

#         self.conv = nn.Conv2d(1, 1, kernel_size=(1, k_size),
#                               stride=1, padding=(0, (k_size - 1) // 2), bias=False)
#         self.sigmoid = nn.Sigmoid()
#         self.dropout = nn.Dropout(dropout_p)
#         self.weight = nn.Parameter(torch.zeros(len(pool_types)))

#     def forward(self, x):
#         feats = [pool(x) for pool in self.pools]
#         w = torch.sigmoid(self.weight)
#         base = sum(w[i] * feats[i] for i in range(len(feats)))

#         if len(feats) == 2:
#             base = 0.5 * (feats[0] + feats[1]) + w[0] * feats[0] + w[1] * feats[1]

#         y = base.permute(0, 3, 2, 1).contiguous()
#         y = self.conv(y)
#         y = self.dropout(y)
#         y = y.permute(0, 3, 2, 1).contiguous()
#         y = self.sigmoid(y)
#         return x * y.expand_as(x)

# class MCALayer(nn.Module):
#     def __init__(self, channels: int, no_spatial: bool = False, drop_path=0.1):
#         super().__init__()
#         lambd, gamma = 1.5, 1.0
#         temp = int(round(abs((math.log2(max(channels, 2)) - gamma) / lambd)))
#         k_spatial = max(3, temp if temp % 2 == 1 else temp + 1)

#         self.h_cw = MCAGate(3)
#         self.w_hc = MCAGate(3)
#         self.no_spatial = no_spatial
#         if not no_spatial:
#             self.c_hw = MCAGate(k_spatial)

#         # === LayerNorm + LayerScale + DropPath ===
#         self.norm = nn.LayerNorm(channels)
#         self.layer_scale = nn.Parameter(1e-6 * torch.ones(channels))
#         self.drop_path = nn.Dropout(drop_path)

#     def forward(self, x):
#         shortcut = x
#         x_h = x.permute(0, 2, 1, 3).contiguous()
#         x_h = self.h_cw(x_h)
#         x_h = x_h.permute(0, 2, 1, 3).contiguous()

#         x_w = x.permute(0, 3, 2, 1).contiguous()
#         x_w = self.w_hc(x_w)
#         x_w = x_w.permute(0, 3, 2, 1).contiguous()

#         if not self.no_spatial:
#             x_c = self.c_hw(x)
#             out = (x_c + x_h + x_w) / 3.0
#         else:
#             out = (x_h + x_w) / 2.0

#         # === 残差 + LayerScale + DropPath + LayerNorm ===
#         out = self.drop_path(out)
#         out = self.layer_scale.view(1, -1, 1, 1) * out
#         out = out + shortcut
#         out = self.norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
#         return out

# class MCA(nn.Module):
#     def __init__(self, dim: int, **kwargs):
#         super().__init__()
#         self.layer = MCALayer(channels=dim, no_spatial=False)

#     def forward(self, x):
#         assert x.dim() == 4
#         y = self.layer(x)
#         if y.shape != x.shape:
#             raise RuntimeError(f"MCA 输出形状 {y.shape} 与输入 {x.shape} 不一致")
#         return y




# v1.5报错
# RuntimeError: cuDNN error: CUDNN_STATUS_BAD_PARAM
# You can try to repro this exception using the following code snippet. If that doesn't trigger the error, please include your original repro script when reporting this issue.

# import torch
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = False
# torch.backends.cudnn.allow_tf32 = True
# data = torch.randn([12, 1, 1, 128], dtype=torch.float, device='cuda', requires_grad=True)
# net = torch.nn.Conv2d(1, 1, kernel_size=[1, 5], padding=[0, 2], stride=[1, 1], dilation=[1, 1], groups=1)
# net = net.cuda().float()
# out = net(data)
# out.backward(torch.randn_like(out))
# torch.cuda.synchronize()

# ConvolutionParams 
#     data_type = CUDNN_DATA_FLOAT
#     padding = [0, 2, 0]
#     stride = [1, 1, 0]
#     dilation = [1, 1, 0]
#     groups = 1
#     deterministic = false
#     allow_tf32 = true
# input: TensorDescriptor 0x11b81d650
#     type = CUDNN_DATA_FLOAT
#     nbDims = 4
#     dimA = 12, 1, 1, 128, 
#     strideA = 128, 128, 128, 1, 
# output: TensorDescriptor 0x11b83f430
#     type = CUDNN_DATA_FLOAT
#     nbDims = 4
#     dimA = 12, 1, 1, 128, 
#     strideA = 128, 128, 128, 1, 
# weight: FilterDescriptor 0x7ee2d8150c70
#     type = CUDNN_DATA_FLOAT
#     tensor_format = CUDNN_TENSOR_NCHW
#     nbDims = 4
#     dimA = 1, 1, 1, 5, 
# Pointer addresses: 
#     input: 0x7ee99b6df800
#     output: 0x7ee99b6de000
#     weight: 0x7ee99b710000
# Additional pointer addresses: 
#     grad_output: 0x7ee99b6de000
#     grad_weight: 0x7ee99b710000
# Backward filter algorithm: 1
# v1.5 == v1.4+串联ECA,为了解决报错，从conv2d改为conv1d
# import math
# import torch
# from torch import nn

# __all__ = ["MCAGate", "MCALayer", "MCA", "ECABlock"]

# # ----------- StdPool 保留 ----------
# class StdPool(nn.Module):
#     def forward(self, x):
#         b, c, h, w = x.size()
#         std = x.view(b, c, -1).std(dim=2, keepdim=True)
#         return std.view(b, c, 1, 1)

# # ----------- ECA 模块 ----------
# class ECABlock(nn.Module):
#     def __init__(self, channels, gamma=2, b=1):
#         super().__init__()
#         k = int(abs((math.log2(channels) + b) / gamma))
#         k = k if k % 2 else k + 1  # 保证 k 为奇数
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         y = self.avg_pool(x)  # (B, C, 1, 1)
#         y = self.conv(y.squeeze(-1).transpose(-1, -2))  # (B, 1, C)
#         y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))  # (B, C, 1, 1)
#         return x * y.expand_as(x)

# # ----------- MCA Gate 更新conv1d ----------
# class MCAGate(nn.Module):
#     def __init__(self, k_size: int, pool_types=('avg', 'std'), dropout_p=0.1):
#         super().__init__()
#         assert k_size >= 3 and k_size % 2 == 1
#         self.pools = nn.ModuleList()
#         for p in pool_types:
#             if p == "avg":
#                 self.pools.append(nn.AdaptiveAvgPool2d(1))
#             elif p == "std":
#                 self.pools.append(StdPool())
#             else:
#                 raise NotImplementedError

#         # 仅改动：Conv2d → Conv1d，其它保持 v1.4 完整逻辑
#         self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
#                               padding=(k_size - 1) // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()
#         self.dropout = nn.Dropout(dropout_p)
#         self.weight = nn.Parameter(torch.zeros(len(pool_types)))

#     def forward(self, x):
#         # 获取池化特征
#         feats = [pool(x) for pool in self.pools]
#         w = torch.sigmoid(self.weight)  # 获取权重
#         base = sum(w[i] * feats[i] for i in range(len(feats)))

#         if len(feats) == 2:
#             base = 0.5 * (feats[0] + feats[1]) + w[0] * feats[0] + w[1] * feats[1]

#         # 维度转换：
#         # (B, C, 1, 1) → permute → (B, 1, 1, C) → squeeze → (B, 1, C) → Conv1d → (B, 1, C)
#         # → unsqueeze → (B, 1, 1, C) → permute → (B, C, 1, 1) → dropout → sigmoid → 广播
#         y = base.permute(0, 3, 2, 1).contiguous()   # (B, 1, 1, C)
#         y = y.squeeze(2)                            # (B, 1, C)
#         y = self.conv(y)                            # Conv1d
#         y = y.unsqueeze(2)                          # (B, 1, 1, C)
#         y = y.permute(0, 3, 2, 1).contiguous()      # (B, C, 1, 1)
#         y = self.dropout(y)                         # 与 v1.4 位置一致
#         y = self.sigmoid(y)                         # 激活函数

#         # 广播：最终输出与原输入形状一致
#         return x * y.expand_as(x)                   # 广播到 (B, C, H, W)

# # ----------- MCALayer 保留 v1.4 逻辑 + 串联 ECA ----------
# class MCALayer(nn.Module):
#     def __init__(self, channels: int, no_spatial: bool = False, drop_path=0.1):
#         super().__init__()
#         lambd, gamma = 1.5, 1.0
#         temp = int(round(abs((math.log2(max(channels, 2)) - gamma) / lambd)))
#         k_spatial = max(3, temp if temp % 2 == 1 else temp + 1)

#         self.h_cw = MCAGate(3)
#         self.w_hc = MCAGate(3)
#         self.no_spatial = no_spatial
#         if not no_spatial:
#             self.c_hw = MCAGate(k_spatial)

#         # === LayerNorm + LayerScale + DropPath ===
#         self.norm = nn.LayerNorm(channels)
#         self.layer_scale = nn.Parameter(1e-6 * torch.ones(channels))
#         self.drop_path = nn.Dropout(drop_path)

#         # === 串联 ECA ===
#         self.eca = ECABlock(channels)

#     def forward(self, x):
#         shortcut = x
#         x_h = x.permute(0, 2, 1, 3).contiguous()
#         x_h = self.h_cw(x_h)
#         x_h = x_h.permute(0, 2, 1, 3).contiguous()

#         x_w = x.permute(0, 3, 2, 1).contiguous()
#         x_w = self.w_hc(x_w)
#         x_w = x_w.permute(0, 3, 2, 1).contiguous()

#         if not self.no_spatial:
#             x_c = self.c_hw(x)
#             out = (x_c + x_h + x_w) / 3.0
#         else:
#             out = (x_h + x_w) / 2.0

#         # === 残差 + LayerScale + DropPath + LayerNorm ===
#         out = self.drop_path(out)
#         out = self.layer_scale.view(1, -1, 1, 1) * out
#         out = out + shortcut
#         out = self.norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

#         # === 串联 ECA ===
#         out = self.eca(out)

#         return out

# # ----------- MCA-ECA 封装 ----------
# class MCA(nn.Module):
#     def __init__(self, dim: int, **kwargs):
#         super().__init__()
#         self.layer = MCALayer(channels=dim, no_spatial=False)

#     def forward(self, x):
#         assert x.dim() == 4
#         y = self.layer(x)
#         if y.shape != x.shape:
#             raise RuntimeError(f"MCA_ECA 输出形状 {y.shape} 与输入 {x.shape} 不一致")
#         return y

# v1.5.1
# import math
# import torch
# from torch import nn

# __all__ = ["MCAGate", "MCALayer", "MCA", "ECABlock"]

# # ----------- StdPool 保留 ----------
# class StdPool(nn.Module):
#     def forward(self, x):
#         b, c, h, w = x.size()
#         std = x.view(b, c, -1).std(dim=2, keepdim=True)
#         return std.view(b, c, 1, 1)

# # ----------- ECA 模块 ----------
# class ECABlock(nn.Module):
#     def __init__(self, channels, gamma=2, b=1):
#         super().__init__()
#         k = int(abs((math.log2(channels) + b) / gamma))
#         k = k if k % 2 else k + 1  # 保证 k 为奇数
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         y = self.avg_pool(x)  # (B, C, 1, 1)
#         y = self.conv(y.squeeze(-1).transpose(-1, -2))  # (B, 1, C)
#         y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))  # (B, C, 1, 1)
#         return x * y.expand_as(x)

# # ----------- MCA Gate 更新conv1d ----------
# class MCAGate(nn.Module):
#     def __init__(self, k_size: int, pool_types=('avg', 'std'), dropout_p=0.1):
#         super().__init__()
#         assert k_size >= 3 and k_size % 2 == 1
#         self.pools = nn.ModuleList()
#         for p in pool_types:
#             if p == "avg":
#                 self.pools.append(nn.AdaptiveAvgPool2d(1))
#             elif p == "std":
#                 self.pools.append(StdPool())
#             else:
#                 raise NotImplementedError

#         # 仅改动：Conv2d → Conv1d，其它保持 v1.4 完整逻辑
#         self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
#                               padding=(k_size - 1) // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()
#         self.dropout = nn.Dropout(dropout_p)
#         self.weight = nn.Parameter(torch.zeros(len(pool_types)))

#     def forward(self, x):
#         # 获取池化特征
#         feats = [pool(x) for pool in self.pools]
#         w = torch.sigmoid(self.weight)  # 获取权重
#         base = sum(w[i] * feats[i] for i in range(len(feats)))

#         if len(feats) == 2:
#             base = 0.5 * (feats[0] + feats[1]) + w[0] * feats[0] + w[1] * feats[1]

#         # 维度转换：
#         # (B, C, 1, 1) → permute → (B, 1, 1, C) → squeeze → (B, 1, C) → Conv1d → (B, 1, C)
#         # → unsqueeze → (B, 1, 1, C) → permute → (B, C, 1, 1) → dropout → sigmoid → 广播
#         y = base.permute(0, 3, 2, 1).contiguous()   # (B, 1, 1, C)
#         y = y.squeeze(2)                            # (B, 1, C)
#         y = self.conv(y)                            # Conv1d
#         y = y.unsqueeze(2)                          # (B, 1, 1, C)
#         y = y.permute(0, 3, 2, 1).contiguous()      # (B, C, 1, 1)
#         y = self.dropout(y)                         # 与 v1.4 位置一致
#         y = self.sigmoid(y)                         # 激活函数

#         # 广播：最终输出与原输入形状一致
#         return x * y.expand_as(x)                   # 广播到 (B, C, H, W)

# # ----------- MCALayer 保留 v1.4 逻辑 + 串联 ECA ----------
# class MCALayer(nn.Module):
#     def __init__(self, channels: int, no_spatial: bool = False, drop_path=0.1, eca_alpha=0.3):
#         super().__init__()
#         lambd, gamma = 1.5, 1.0
#         temp = int(round(abs((math.log2(max(channels, 2)) - gamma) / lambd)))
#         k_spatial = max(3, temp if temp % 2 == 1 else temp + 1)

#         self.h_cw = MCAGate(3)
#         self.w_hc = MCAGate(3)
#         self.no_spatial = no_spatial
#         if not no_spatial:
#             self.c_hw = MCAGate(k_spatial)

#         self.norm = nn.LayerNorm(channels)
#         self.layer_scale = nn.Parameter(1e-6 * torch.ones(channels))
#         self.drop_path = nn.Dropout(drop_path)
#         self.eca = ECABlock(channels)
#         self.eca_alpha = eca_alpha          # 改动1：幅度缩放

#     def forward(self, x):
#         shortcut = x
#         x_h = x.permute(0, 2, 1, 3).contiguous()
#         x_h = self.h_cw(x_h)
#         x_h = x_h.permute(0, 2, 1, 3).contiguous()

#         x_w = x.permute(0, 3, 2, 1).contiguous()
#         x_w = self.w_hc(x_w)
#         x_w = x_w.permute(0, 3, 2, 1).contiguous()

#         if not self.no_spatial:
#             x_c = self.c_hw(x)
#             out = (x_c + x_h + x_w) / 3.0
#         else:
#             out = (x_h + x_w) / 2.0

#         out = self.drop_path(out)
#         out = self.layer_scale.view(1, -1, 1, 1) * out
#         out = out + shortcut                      # 残差相加
#         out = self.eca(out) * self.eca_alpha      # 改动2：ECA 在 Norm 前，且带缩放
#         out = self.norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
#         return out

# # ----------- MCA-ECA 封装 ----------
# class MCA(nn.Module):
#     def __init__(self, dim: int, **kwargs):
#         super().__init__()
#         self.layer = MCALayer(channels=dim, no_spatial=False)

#     def forward(self, x):
#         assert x.dim() == 4
#         y = self.layer(x)
#         if y.shape != x.shape:
#             raise RuntimeError(f"MCA_ECA 输出形状 {y.shape} 与输入 {x.shape} 不一致")
#         return y

# # v1.5.2
# import math
# import torch
# from torch import nn

# __all__ = ["MCAGate", "MCALayer", "MCA", "ECABlock"]

# # ----------- StdPool 保留 ----------
# class StdPool(nn.Module):
#     def forward(self, x):
#         b, c, h, w = x.size()
#         std = x.view(b, c, -1).std(dim=2, keepdim=True)
#         return std.view(b, c, 1, 1)

# # ----------- ECA 模块 ----------
# class ECABlock(nn.Module):
#     def __init__(self, channels, gamma=2, b=1):
#         super().__init__()
#         k = int(abs((math.log2(channels) + b) / gamma))
#         k = k if k % 2 else k + 1  # 保证 k 为奇数
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         y = self.avg_pool(x)  # (B, C, 1, 1)
#         y = self.conv(y.squeeze(-1).transpose(-1, -2))  # (B, 1, C)
#         y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))  # (B, C, 1, 1)
#         return x * y.expand_as(x)

# # ----------- MCA Gate 更新conv1d ----------
# class MCAGate(nn.Module):
#     def __init__(self, k_size: int, pool_types=('avg', 'std'), dropout_p=0.1):
#         super().__init__()
#         assert k_size >= 3 and k_size % 2 == 1
#         self.pools = nn.ModuleList()
#         for p in pool_types:
#             if p == "avg":
#                 self.pools.append(nn.AdaptiveAvgPool2d(1))
#             elif p == "std":
#                 self.pools.append(StdPool())
#             else:
#                 raise NotImplementedError

#         # 仅改动：Conv2d → Conv1d，其它保持 v1.4 完整逻辑
#         self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
#                               padding=(k_size - 1) // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()
#         self.dropout = nn.Dropout(dropout_p)
#         self.weight = nn.Parameter(torch.zeros(len(pool_types)))

#     def forward(self, x):
#         # 获取池化特征
#         feats = [pool(x) for pool in self.pools]
#         w = torch.sigmoid(self.weight)  # 获取权重
#         base = sum(w[i] * feats[i] for i in range(len(feats)))

#         if len(feats) == 2:
#             base = 0.5 * (feats[0] + feats[1]) + w[0] * feats[0] + w[1] * feats[1]

#         # 维度转换：
#         # (B, C, 1, 1) → permute → (B, 1, 1, C) → squeeze → (B, 1, C) → Conv1d → (B, 1, C)
#         # → unsqueeze → (B, 1, 1, C) → permute → (B, C, 1, 1) → dropout → sigmoid → 广播
#         y = base.permute(0, 3, 2, 1).contiguous()   # (B, 1, 1, C)
#         y = y.squeeze(2)                            # (B, 1, C)
#         y = self.conv(y)                            # Conv1d
#         y = y.unsqueeze(2)                          # (B, 1, 1, C)
#         y = y.permute(0, 3, 2, 1).contiguous()      # (B, C, 1, 1)
#         y = self.dropout(y)                         # 与 v1.4 位置一致
#         y = self.sigmoid(y)                         # 激活函数

#         # 广播：最终输出与原输入形状一致
#         return x * y.expand_as(x)                   # 广播到 (B, C, H, W)

# class MCALayer(nn.Module):
#     def __init__(self, channels: int, no_spatial: bool = False, drop_path=0.1, eca_alpha=1.0):
#         super().__init__()
#         # 改动3：统一 kernel size
#         k = max(3, int(round(abs((math.log2(max(channels, 2)) - 1.0) / 1.5))) | 1)

#         self.h_cw = MCAGate(k)
#         self.w_hc = MCAGate(k)
#         self.no_spatial = no_spatial
#         if not self.no_spatial:
#             self.c_hw = MCAGate(k)

#         self.norm = nn.LayerNorm(channels)
#         self.layer_scale = nn.Parameter(1e-6 * torch.ones(channels))
#         self.drop_path = nn.Dropout(drop_path)   # 仅用于 ECA 分支
#         self.eca = ECABlock(channels)
#         self.eca_alpha = nn.Parameter(torch.tensor(eca_alpha))  # 可学习缩放

#     def forward(self, x):
#         shortcut = x
#         x_h = x.permute(0, 2, 1, 3).contiguous()
#         x_h = self.h_cw(x_h)
#         x_h = x_h.permute(0, 2, 1, 3).contiguous()

#         x_w = x.permute(0, 3, 2, 1).contiguous()
#         x_w = self.w_hc(x_w)
#         x_w = x_w.permute(0, 3, 2, 1).contiguous()

#         if not self.no_spatial:
#             x_c = self.c_hw(x)
#             out = (x_c + x_h + x_w) / 3.0
#         else:
#             out = (x_h + x_w) / 2.0

#         out = self.layer_scale.view(1, -1, 1, 1) * out
#         out = out + shortcut                      # 主残差

#         # 改动4+5：并行 ECA，仅给 ECA 加 Stochastic Depth
#         eca_out = self.eca(out)
#         eca_out = self.drop_path(eca_out)         # 只丢 ECA
#         out = out + self.eca_alpha * eca_out      # 并行相加

#         out = self.norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
#         return out

# # ----------- MCA-ECA 封装 ----------
# class MCA(nn.Module):
#     def __init__(self, dim: int, **kwargs):
#         super().__init__()
#         self.layer = MCALayer(channels=dim, no_spatial=False)

#     def forward(self, x):
#         assert x.dim() == 4
#         y = self.layer(x)
#         if y.shape != x.shape:
#             raise RuntimeError(f"MCA_ECA 输出形状 {y.shape} 与输入 {x.shape} 不一致")
#         return y

# MCA v2.0 终极涨点版：全局上下文 + 跨轴交互 + 多尺度 + 深度先验
# import math
# import torch
# from torch import nn
# import torch.nn.functional as F

# __all__ = ["MCAGate", "MCALayer", "MCA"]

# class StdPool(nn.Module):
#     """标准差池化"""
#     def forward(self, x):
#         b, c, h, w = x.size()
#         std = x.view(b, c, -1).std(dim=2, keepdim=True)
#         return std.view(b, c, 1, 1)

# class GlobalContextBlock(nn.Module):
#     """
#     增强全局上下文模块：avg+max+std三池化 + 非局部注意力
#     """
#     def __init__(self, channels, reduction=16):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.std_pool = StdPool()
        
#         # 三路MLP融合
#         self.fc = nn.Sequential(
#             nn.Conv2d(channels, channels // reduction, 1, bias=False),
#             nn.LayerNorm([channels // reduction, 1, 1]),
#             nn.GELU(),  # GELU比ReLU更平滑
#             nn.Conv2d(channels // reduction, channels, 1, bias=False)
#         )
        
#         # 非局部注意力分支（轻量版）
#         self.non_local = nn.Sequential(
#             nn.Conv2d(channels, channels // 4, 1),
#             nn.LayerNorm([channels // 4, 1, 1]),
#             nn.GELU(),
#             nn.Conv2d(channels // 4, channels, 1)
#         )
        
#         self.sigmoid = nn.Sigmoid()
#         self.alpha = nn.Parameter(torch.tensor(0.5))  # 可学习融合权重
        
#     def forward(self, x):
#         # 三池化策略
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         std_out = self.fc(self.std_pool(x))
        
#         # 非局部分支
#         nl_out = self.non_local(self.avg_pool(x))
        
#         # 自适应融合
#         pool_attention = self.sigmoid(avg_out + max_out + std_out)
#         nl_attention = torch.sigmoid(nl_out)
        
#         alpha = torch.sigmoid(self.alpha)
#         attention = alpha * pool_attention + (1 - alpha) * nl_attention
        
#         return x * attention

# class CrossAxisInteraction(nn.Module):
#     """
#     跨轴交互模块：显式建模H和W分支间的依赖
#     """
#     def __init__(self, channels):
#         super().__init__()
#         # 双分支融合卷积
#         self.conv_hw = nn.Sequential(
#             nn.Conv2d(channels * 2, channels, 1, bias=False),
#             nn.LayerNorm([channels, 1, 1]),
#             nn.GELU(),
#             nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),  # DWConv
#             nn.Conv2d(channels, channels, 1, bias=False)
#         )
        
#         # 门控机制
#         self.gate = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels * 2, channels, 1),
#             nn.Sigmoid()
#         )
        
#     def forward(self, x_h, x_w):
#         # 拼接H和W分支
#         cat = torch.cat([x_h, x_w], dim=1)  # (B, 2C, H, W)
        
#         # 门控融合
#         gate = self.gate(cat)  # (B, C, 1, 1)
#         out = self.conv_hw(cat)
        
#         return out * gate

# class DepthPriorEncoding(nn.Module):
#     """
#     深度先验位置编码：为深度估计任务定制
#     使用可学习的高斯位置编码强化远近距离感知
#     """
#     def __init__(self, channels, height, width):
#         super().__init__()
#         # 垂直方向位置编码（深度通常与Y轴相关）
#         y_pos = torch.linspace(-1, 1, height).view(1, 1, height, 1)
#         x_pos = torch.linspace(-1, 1, width).view(1, 1, 1, width)
        
#         # 可学习的高斯参数
#         self.sigma_y = nn.Parameter(torch.tensor(0.5))
#         self.sigma_x = nn.Parameter(torch.tensor(0.5))
        
#         # 位置编码投影
#         self.pos_proj = nn.Conv2d(2, channels, 1)
        
#         self.register_buffer('y_pos', y_pos)
#         self.register_buffer('x_pos', x_pos)
        
#     def forward(self, x):
#         b, c, h, w = x.shape
        
#         # 自适应调整位置编码
#         y_enc = torch.exp(-(self.y_pos ** 2) / (2 * self.sigma_y ** 2))
#         x_enc = torch.exp(-(self.x_pos ** 2) / (2 * self.sigma_x ** 2))
        
#         # 插值到输入尺寸
#         y_enc = F.interpolate(y_enc.expand(b, -1, -1, -1), size=(h, w), mode='bilinear', align_corners=False)
#         x_enc = F.interpolate(x_enc.expand(b, -1, -1, -1), size=(h, w), mode='bilinear', align_corners=False)
        
#         pos_enc = torch.cat([y_enc, x_enc], dim=1)  # (B, 2, H, W)
#         pos_enc = self.pos_proj(pos_enc)  # (B, C, H, W)
        
#         return x + pos_enc

# class MCAGate(nn.Module):
#     """
#     增强门控：池化 + 1D卷积 + 注意力
#     """
#     def __init__(self, k_size: int, pool_types=('avg', 'std'), dropout_p=0.03):
#         super().__init__()
#         assert k_size >= 3 and k_size % 2 == 1
        
#         self.pools = nn.ModuleList()
#         for p in pool_types:
#             if p == "avg":
#                 self.pools.append(nn.AdaptiveAvgPool2d(1))
#             elif p == "std":
#                 self.pools.append(StdPool())
#             else:
#                 raise NotImplementedError
        
#         # 1D卷积 + LayerNorm
#         self.conv = nn.Sequential(
#             nn.Conv2d(1, 1, kernel_size=(1, k_size),
#                       stride=1, padding=(0, (k_size - 1) // 2), bias=False),
#             nn.LayerNorm([1, 1, 1])  # 归一化
#         )
        
#         self.sigmoid = nn.Sigmoid()
#         self.dropout = nn.Dropout(dropout_p)
#         self.weight = nn.Parameter(torch.ones(len(pool_types)) / len(pool_types))  # 均匀初始化

#     def forward(self, x):
#         feats = [pool(x) for pool in self.pools]
#         w = torch.softmax(self.weight, dim=0)  # softmax归一化
#         base = sum(w[i] * feats[i] for i in range(len(feats)))
        
#         y = base.permute(0, 3, 2, 1).contiguous()
#         y = self.conv(y)
#         y = self.dropout(y)
#         y = y.permute(0, 3, 2, 1).contiguous()
#         y = self.sigmoid(y)
        
#         return x * y.expand_as(x)

# class MultiScaleMCABranch(nn.Module):
#     """
#     多尺度分支：在不同分辨率上应用MCA
#     """
#     def __init__(self, channels, scales=[1.0, 0.5]):
#         super().__init__()
#         self.scales = scales
#         self.branches = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
#                 nn.Conv2d(channels, channels, 1, bias=False)
#             ) for _ in scales
#         ])
        
#         # 融合卷积
#         self.fusion = nn.Sequential(
#             nn.Conv2d(channels * len(scales), channels, 1, bias=False),
#             nn.LayerNorm([channels, 1, 1]),
#             nn.GELU()
#         )
        
#     def forward(self, x):
#         b, c, h, w = x.shape
#         feats = []
        
#         for i, scale in enumerate(self.scales):
#             if scale == 1.0:
#                 feat = self.branches[i](x)
#             else:
#                 # 下采样 -> 处理 -> 上采样
#                 scaled_h, scaled_w = int(h * scale), int(w * scale)
#                 x_scaled = F.interpolate(x, size=(scaled_h, scaled_w), mode='bilinear', align_corners=False)
#                 feat = self.branches[i](x_scaled)
#                 feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
#             feats.append(feat)
        
#         # 多尺度融合
#         out = self.fusion(torch.cat(feats, dim=1))
#         return out

# class MCALayer(nn.Module):
#     """
#     终极MCA层：集成所有改进
#     """
#     def __init__(self, channels: int, height=24, width=80, 
#                  no_spatial: bool = False, drop_path=0.03,
#                  use_global_context=True, use_cross_axis=True,
#                  use_depth_prior=True, use_multiscale=True):
#         super().__init__()
        
#         # 自适应空间核大小
#         lambd, gamma = 1.5, 1.0
#         temp = int(round(abs((math.log2(max(channels, 2)) - gamma) / lambd)))
#         k_spatial = max(3, temp if temp % 2 == 1 else temp + 1)
        
#         # === 核心门控 ===
#         self.h_cw = MCAGate(3, dropout_p=0.03)
#         self.w_hc = MCAGate(3, dropout_p=0.03)
#         self.no_spatial = no_spatial
#         if not no_spatial:
#             self.c_hw = MCAGate(k_spatial, dropout_p=0.03)
        
#         # === 新增模块 ===
#         self.use_global_context = use_global_context
#         if use_global_context:
#             self.global_context = GlobalContextBlock(channels, reduction=16)
        
#         self.use_cross_axis = use_cross_axis
#         if use_cross_axis:
#             self.cross_axis = CrossAxisInteraction(channels)
        
#         self.use_depth_prior = use_depth_prior
#         if use_depth_prior:
#             self.depth_prior = DepthPriorEncoding(channels, height, width)
        
#         self.use_multiscale = use_multiscale
#         if use_multiscale:
#             self.multiscale = MultiScaleMCABranch(channels, scales=[1.0, 0.5])
        
#         # === 正则化 ===
#         self.norm = nn.LayerNorm(channels)
#         self.layer_scale = nn.Parameter(1e-4 * torch.ones(channels))  # 更大初始值
#         self.drop_path = nn.Dropout(drop_path) if drop_path > 0 else nn.Identity()
        
#         # 残差门控
#         self.residual_gate = nn.Parameter(torch.tensor(1.0))

#     def forward(self, x):
#         shortcut = x
        
#         # === 深度先验编码 ===
#         if self.use_depth_prior:
#             x = self.depth_prior(x)
        
#         # === H和W分支 ===
#         x_h = x.permute(0, 2, 1, 3).contiguous()
#         x_h = self.h_cw(x_h)
#         x_h = x_h.permute(0, 2, 1, 3).contiguous()
        
#         x_w = x.permute(0, 3, 2, 1).contiguous()
#         x_w = self.w_hc(x_w)
#         x_w = x_w.permute(0, 3, 2, 1).contiguous()
        
#         # === 跨轴交互（关键改进）===
#         if self.use_cross_axis:
#             x_hw_fused = self.cross_axis(x_h, x_w)
#         else:
#             x_hw_fused = (x_h + x_w) / 2.0
        
#         # === C分支融合 ===
#         if not self.no_spatial:
#             x_c = self.c_hw(x)
#             out = (x_c + x_hw_fused) / 2.0
#         else:
#             out = x_hw_fused
        
#         # === 多尺度增强 ===
#         if self.use_multiscale:
#             out = out + self.multiscale(out)
        
#         # === 全局上下文 ===
#         if self.use_global_context:
#             out = out + self.global_context(out)
        
#         # === 残差连接 + 自适应门控 ===
#         out = self.drop_path(out)
#         out = self.layer_scale.view(1, -1, 1, 1) * out
        
#         # 可学习残差权重
#         gate = torch.sigmoid(self.residual_gate)
#         out = gate * out + shortcut
        
#         # LayerNorm
#         out = self.norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
#         return out

# class MCA(nn.Module):
#     """
#     MCA v2.0 终极版封装
#     """
#     def __init__(self, dim: int, height=24, width=80, drop_path=0.03,
#                  use_global_context=True, use_cross_axis=True,
#                  use_depth_prior=True, use_multiscale=True, **kwargs):
#         super().__init__()
#         self.layer = MCALayer(
#             channels=dim,
#             height=height,
#             width=width,
#             no_spatial=False,
#             drop_path=drop_path,
#             use_global_context=use_global_context,
#             use_cross_axis=use_cross_axis,
#             use_depth_prior=use_depth_prior,
#             use_multiscale=use_multiscale
#         )

#     def forward(self, x):
#         assert x.dim() == 4, f"MCA仅支持4D(B,C,H,W)，但得到{x.shape}"
#         y = self.layer(x)
#         if y.shape != x.shape:
#             raise RuntimeError(f"MCA输出形状{y.shape}与输入{x.shape}不一致")
#         return y

