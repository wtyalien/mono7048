# from __future__ import absolute_import, division, print_function
# from collections import OrderedDict
# from layers import *
# from timm.models.layers import trunc_normal_


# class DepthDecoder(nn.Module):
#     def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
#         super().__init__()

#         self.num_output_channels = num_output_channels
#         self.use_skips = use_skips
#         self.upsample_mode = 'bilinear'
#         self.scales = scales

#         self.num_ch_enc = num_ch_enc
#         self.num_ch_dec = (self.num_ch_enc / 2).astype('int')

#         # decoder
#         self.convs = OrderedDict()
#         for i in range(2, -1, -1):
#             # upconv_0
#             num_ch_in = self.num_ch_enc[-1] if i == 2 else self.num_ch_dec[i + 1]
#             num_ch_out = self.num_ch_dec[i]
#             self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
#             # print(i, num_ch_in, num_ch_out)
#             # upconv_1
#             num_ch_in = self.num_ch_dec[i]
#             if self.use_skips and i > 0:
#                 num_ch_in += self.num_ch_enc[i - 1]
#             num_ch_out = self.num_ch_dec[i]
#             self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

#         for s in self.scales:
#             self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

#         self.decoder = nn.ModuleList(list(self.convs.values()))
#         self.sigmoid = nn.Sigmoid()

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, (nn.Conv2d, nn.Linear)):
#             trunc_normal_(m.weight, std=.02)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, input_features):
#         self.outputs = {}
#         x = input_features[-1]
#         for i in range(2, -1, -1):
#             x = self.convs[("upconv", i, 0)](x)
#             x = [upsample(x)]

#             if self.use_skips and i > 0:
#                 x += [input_features[i - 1]]
#             x = torch.cat(x, 1)
#             x = self.convs[("upconv", i, 1)](x)

#             if i in self.scales:
#                 f = upsample(self.convs[("dispconv", i)](x), mode='bilinear')
#                 self.outputs[("disp", i)] = self.sigmoid(f)

#         return self.outputs


# v1.4+MSFM ,v1.4不采用这版
from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from layers import *
from timm.models.layers import trunc_normal_
import torch.nn.functional as F


class MSFM(nn.Module):
    """Multi-Scale Feature Fusion Module
    融合来自不同level的特征,增强多尺度信息
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels
        ])
        self.output_conv = nn.Sequential(
            nn.Conv2d(out_channels * len(in_channels), out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, features, target_size):
        """
        features: list of tensors from different scales
        target_size: (H, W) to align all features
        """
        aligned = []
        for feat, lateral in zip(features, self.lateral_convs):
            x = lateral(feat)
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            aligned.append(x)
        
        fused = torch.cat(aligned, dim=1)
        out = self.output_conv(fused)
        return out


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super().__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'bilinear'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = (self.num_ch_enc / 2).astype('int')

        # === 添加多尺度融合模块 ===
        self.msfm_modules = nn.ModuleDict()
        for i in range(2, -1, -1):
            # 收集当前及更高level的通道数
            available_channels = [self.num_ch_enc[j] for j in range(i, len(self.num_ch_enc))]
            self.msfm_modules[f"msfm_{i}"] = MSFM(available_channels, self.num_ch_dec[i])

        # decoder (原有结构保留)
        self.convs = OrderedDict()
        for i in range(2, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 2 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            
            # upconv_1 - 输入加上MSFM的输出
            num_ch_in = self.num_ch_dec[i] * 2  # decoder特征 + MSFM特征
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, input_features):
        self.outputs = {}
        x = input_features[-1]
        
        for i in range(2, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = upsample(x)
            
            # === 多尺度融合 ===
            H, W = x.shape[2:]
            multi_scale_feats = [input_features[j] for j in range(i, len(input_features))]
            msfm_out = self.msfm_modules[f"msfm_{i}"](multi_scale_feats, (H, W))
            
            # 拼接: decoder特征 + MSFM特征 + skip connection
            x = [x, msfm_out]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)

            if i in self.scales:
                f = upsample(self.convs[("dispconv", i)](x), mode='bilinear')
                self.outputs[("disp", i)] = self.sigmoid(f)

        return self.outputs