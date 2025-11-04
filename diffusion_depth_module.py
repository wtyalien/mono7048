# """
# 扩散引导的深度估计优化模块
# 结合Lite-Mono架构，在训练时引入轻量扩散先验
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math


# class SinusoidalPositionEmbeddings(nn.Module):
#     """时间步嵌入 (扩散必需)"""
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim

#     def forward(self, time):
#         device = time.device
#         half_dim = self.dim // 2
#         embeddings = math.log(10000) / (half_dim - 1)
#         embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
#         embeddings = time[:, None] * embeddings[None, :]
#         embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
#         return embeddings


# class LightweightDiffusionBlock(nn.Module):
#     """轻量扩散块 - 适配MCA的特征维度"""
#     def __init__(self, channels, time_emb_dim=128):
#         super().__init__()
#         self.time_mlp = nn.Sequential(
#             nn.Linear(time_emb_dim, channels),
#             nn.SiLU(),
#             nn.Linear(channels, channels)
#         )
        
#         # 深度特征处理
#         self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
#         self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
#         self.norm1 = nn.GroupNorm(8, channels)
#         self.norm2 = nn.GroupNorm(8, channels)
#         self.act = nn.SiLU()
        
#     def forward(self, x, t_emb):
#         """
#         x: [B, C, H, W] 深度特征
#         t_emb: [B, time_emb_dim] 时间步嵌入
#         """
#         # 时间步调制
#         t_mod = self.time_mlp(t_emb)[:, :, None, None]
        
#         h = self.norm1(x)
#         h = self.act(h)
#         h = self.conv1(h)
#         h = h + t_mod  # 注入时间信息
        
#         h = self.norm2(h)
#         h = self.act(h)
#         h = self.conv2(h)
        
#         return x + h  # 残差连接


# class DiffusionDepthRefiner(nn.Module):
#     """扩散深度细化器 - 插入到Decoder后"""
#     def __init__(self, 
#                  depth_channels=1,
#                  feature_channels=64,
#                  time_emb_dim=128,
#                  num_steps=5):
#         super().__init__()
#         self.num_steps = num_steps
#         self.time_emb_dim = time_emb_dim
        
#         # 时间步编码器
#         self.time_embed = SinusoidalPositionEmbeddings(time_emb_dim)
        
#         # 深度到特征的映射
#         self.depth_proj = nn.Conv2d(depth_channels, feature_channels, 1)
        
#         # 轻量扩散块
#         self.diffusion_blocks = nn.ModuleList([
#             LightweightDiffusionBlock(feature_channels, time_emb_dim)
#             for _ in range(3)
#         ])
        
#         # 特征到深度的映射
#         self.depth_out = nn.Sequential(
#             nn.Conv2d(feature_channels, feature_channels // 2, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(feature_channels // 2, depth_channels, 1),
#             nn.Sigmoid()
#         )
        
#         # 噪声调度参数 (DDPM)
#         self.register_buffer('betas', self._cosine_beta_schedule(1000))
#         self.register_buffer('alphas', 1.0 - self.betas)
#         self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
#     def _cosine_beta_schedule(self, timesteps, s=0.008):
#         """余弦噪声调度"""
#         steps = timesteps + 1
#         x = torch.linspace(0, timesteps, steps)
#         alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
#         alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
#         betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
#         return torch.clip(betas, 0.0001, 0.9999)
    
#     def add_noise(self, x, t):
#         """前向扩散过程 q(x_t | x_0)"""
#         sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod[t])[:, None, None, None]
#         sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - self.alphas_cumprod[t])[:, None, None, None]
#         noise = torch.randn_like(x)
#         return sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise, noise
    
#     def denoise_step(self, x_t, t):
#         """单步去噪"""
#         t_emb = self.time_embed(t)
        
#         # 特征提取
#         feat = self.depth_proj(x_t)
        
#         # 扩散块处理
#         for block in self.diffusion_blocks:
#             feat = block(feat, t_emb)
        
#         # 预测深度
#         predicted_depth = self.depth_out(feat)
#         return predicted_depth
    
#     def forward(self, depth_init, num_inference_steps=None):
#         """
#         训练时: 随机时间步去噪
#         推理时: 多步迭代去噪
        
#         depth_init: [B, 1, H, W] 初始深度预测
#         """
#         if self.training:
#             # 训练模式: 单步扩散损失
#             B = depth_init.shape[0]
#             t = torch.randint(0, len(self.betas), (B,), device=depth_init.device).long()
            
#             # 加噪
#             x_noisy, noise = self.add_noise(depth_init, t)
            
#             # 去噪
#             depth_pred = self.denoise_step(x_noisy, t)
            
#             return depth_pred, noise, t
#         else:
#             # 推理模式: 迭代去噪
#             steps = num_inference_steps or self.num_steps
#             x = depth_init
            
#             # 简化的DDIM采样
#             timesteps = torch.linspace(len(self.betas)-1, 0, steps, device=x.device).long()
#             for t in timesteps:
#                 t_batch = t.repeat(x.shape[0])
#                 x = self.denoise_step(x, t_batch)
            
#             return x

"""
扩散深度细化器 - 法2正确实现
作为测试阶段精修器，训练阶段用teacher-student模式
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class LightweightDiffusionBlock(nn.Module):
    def __init__(self, channels, time_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, channels),
            nn.SiLU(),
            nn.Linear(channels, channels)
        )
        
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act = nn.SiLU()
        
    def forward(self, x, t_emb):
        t_mod = self.time_mlp(t_emb)[:, :, None, None]
        
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        h = h + t_mod
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        
        return x + h


class DiffusionDepthRefiner(nn.Module):
    """
    法2实现：作为深度优化器
    - 训练时：学习从粗糙深度→精细深度的残差
    - 测试时：确定性DDIM采样细化
    """
    def __init__(self, 
                 depth_channels=1,
                 feature_channels=48,
                 time_emb_dim=128,
                 num_steps=10):
        super().__init__()
        self.num_steps = num_steps
        self.time_emb_dim = time_emb_dim
        
        self.time_embed = SinusoidalPositionEmbeddings(time_emb_dim)
        
        # 深度条件编码（输入粗糙深度）
        self.depth_proj = nn.Conv2d(depth_channels, feature_channels, 1)
        
        self.diffusion_blocks = nn.ModuleList([
            LightweightDiffusionBlock(feature_channels, time_emb_dim)
            for _ in range(3)
        ])
        
        # ===== 关键修改1：输出残差而非绝对深度 =====
        self.residual_out = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels // 2, depth_channels, 1),
            nn.Tanh()  # 残差范围[-1, 1]
        )
        
        # 噪声调度
        self.register_buffer('betas', self._cosine_beta_schedule(1000))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def add_noise(self, x, t):
        """对深度图加噪声（训练teacher-student时用）"""
        sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - self.alphas_cumprod[t])[:, None, None, None]
        noise = torch.randn_like(x)
        return sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise, noise
    
    def denoise_step(self, x_noisy, x_condition, t):
        """
        单步去噪
        x_noisy: 加噪后的深度
        x_condition: 原始深度（作为条件）
        """
        t_emb = self.time_embed(t)
        
        # 以原始深度为条件
        feat = self.depth_proj(x_condition)
        
        for block in self.diffusion_blocks:
            feat = block(feat, t_emb)
        
        # ===== 关键修改2：预测残差 =====
        residual = self.residual_out(feat)
        
        # 残差缩放因子（自适应）
        scale_factor = 0.05 * x_condition.abs().mean()
        
        # 细化深度 = 条件深度 + 小残差
        refined_depth = x_condition + scale_factor * residual
        
        # 确保深度为正
        refined_depth = torch.clamp(refined_depth, min=1e-3)
        
        return refined_depth
    
    def forward(self, depth_init, num_inference_steps=None, deterministic=False):
        """
        训练模式：teacher-student学习
        测试模式：确定性DDIM细化
        """
        if self.training and not deterministic:
            # ===== 训练模式：单步去噪学习 =====
            B = depth_init.shape[0]
            t = torch.randint(0, len(self.betas), (B,), device=depth_init.device).long()
            
            # 对原始深度加噪声
            depth_noisy, noise_gt = self.add_noise(depth_init, t)
            
            # 用原始深度作为条件去噪
            depth_refined = self.denoise_step(depth_noisy, depth_init, t)
            
            # ===== 修改：返回5个值，包含noise_gt =====
            return depth_refined, depth_noisy, noise_gt, t  # 注意：noise改为noise_gt
        
        else:
            # ===== 测试模式：确定性DDIM采样 =====
            steps = num_inference_steps or self.num_steps
            x = depth_init.clone()
            
            timesteps = torch.linspace(len(self.betas)-1, 0, steps, device=x.device).long()
            
            for i, t in enumerate(timesteps):
                t_batch = t.repeat(x.shape[0])
                x = self.denoise_step(x, depth_init, t_batch)
                alpha = i / len(timesteps)
                x = (1 - alpha) * depth_init + alpha * x
            
            return x


class GeometryConsistentLoss(nn.Module):
    """几何一致性损失 - 确保扩散不破坏尺度"""
    def __init__(self):
        super().__init__()
    
    def forward(self, depth_refined, depth_init, color_image):
        """
        depth_refined: 细化后的深度
        depth_init: 初始深度
        color_image: 用于计算边缘
        """
        # 1. 尺度一致性
        scale_init = depth_init.mean()
        scale_refined = depth_refined.mean()
        scale_loss = torch.abs(scale_refined - scale_init) / (scale_init + 1e-7)
        
        # 2. 边缘保持
        def compute_gradient(img):
            grad_x = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])
            grad_y = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
            return grad_x, grad_y
        
        grad_x_init, grad_y_init = compute_gradient(depth_init)
        grad_x_refined, grad_y_refined = compute_gradient(depth_refined)
        
        edge_loss = F.l1_loss(grad_x_refined, grad_x_init.detach()) + \
                    F.l1_loss(grad_y_refined, grad_y_init.detach())
        
        # 3. 平滑约束（在非边缘区域）
        color_grad_x, color_grad_y = compute_gradient(color_image)
        
        # 边缘掩码（颜色梯度大的地方）
        edge_mask_x = (color_grad_x.mean(1, keepdim=True) > 0.1).float()
        edge_mask_y = (color_grad_y.mean(1, keepdim=True) > 0.1).float()
        
        # 非边缘区域的深度应平滑
        smooth_loss = (grad_x_refined * (1 - edge_mask_x)).mean() + \
                      (grad_y_refined * (1 - edge_mask_y)).mean()
        
        return scale_loss + 0.5 * edge_loss + 0.1 * smooth_loss