from __future__ import absolute_import, division, print_function
import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed
from PIL import ImageEnhance, ImageFilter  # 🔹 新增增强库

import torch
import torch.utils.data as data
from torchvision import transforms


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def half_flip_augment(img_tensor):
    """
    对图像上半部分和下半部分分别水平翻转并拼接。
    输入: torch.Tensor, shape [3, H, W]
    输出: torch.Tensor, shape [3, H, W]
    """
    _, H, W = img_tensor.shape
    mid = H // 2
    top = img_tensor[:, :mid, :]
    bottom = img_tensor[:, mid:, :]

    # 分别水平翻转
    top_flipped = torch.flip(top, dims=[2])
    bottom_flipped = torch.flip(bottom, dims=[2])

    # 拼接
    return torch.cat([top_flipped, bottom_flipped], dim=1)


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.png'): 
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()


        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = self.check_depth()

    # v2+v3，水平翻转、灰度等
    # def preprocess(self, inputs, color_aug):
    #     """Resize colour images to the required scales and augment if required

    #     We create the color_aug object in advance and apply the same augmentation to all
    #     images in this item. This ensures that all images input to the pose network receive the
    #     same augmentation.
    #     """
    #     for k in list(inputs):
    #         frame = inputs[k]
    #         if "color" in k:
    #             n, im, i = k
    #             for i in range(self.num_scales):
    #                 inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

    #     # ======================================================
    #     # 仅当 --strong_aug 被指定时才启用增强
    #     # ======================================================
    #     strong_aug = getattr(self, "opt", None)
    #     strong_aug = getattr(strong_aug, "strong_aug", False)

    #     if self.is_train and strong_aug:
    #         for k in list(inputs):
    #             if "color" in k:
    #                 n, im, i = k
    #                 img = inputs[(n, im, 0)]

    #                 # ---- v2 颜色类增强 ----
    #                 if random.random() > 0.5:
    #                     img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    #                 if random.random() > 0.5:
    #                     img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
    #                 if random.random() > 0.5:
    #                     img = ImageEnhance.Color(img).enhance(random.uniform(0.9, 1.1))
    #                 if random.random() > 0.3:
    #                     img = ImageEnhance.Sharpness(img).enhance(random.uniform(0.85, 1.15))
    #                 if random.random() > 0.7:
    #                     img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.4)))

    #                 # ---- v3 新增：2 个涨点 trick (删除透视变换) ----
    #                 # ① 随机擦除（Random Erasing）——强制网络利用上下文
    #                 if random.random() > 0.75:  # 改为25%概率
    #                     w, h = img.size
    #                     ew = int(w * random.uniform(0.02, 0.06))  # 2-6%宽度
    #                     eh = int(h * random.uniform(0.02, 0.06))  # 2-6%高度
    #                     x0 = random.randint(0, w - ew)
    #                     y0 = random.randint(0, h - eh)
    #                     # 随机颜色(更自然)
    #                     r = random.randint(0, 255)
    #                     g = random.randint(0, 255)
    #                     b = random.randint(0, 255)
    #                     erase = Image.new('RGB', (ew, eh), color=(r, g, b))
    #                     img.paste(erase, (x0, y0))

    #                 # ② 随机灰度（Random Grayscale）——颜色不变性
    #                 if random.random() > 0.85:  # 15%概率(因v2已有颜色增强)
    #                     img = ImageEnhance.Color(img).enhance(0)  # 饱和度=0 → 灰度

    #                 # 更新所有尺度
    #                 inputs[(n, im, 0)] = img
    #                 for s in range(1, self.num_scales):
    #                     inputs[(n, im, s)] = self.resize[s](img)

    #         # ---- v2 随机水平翻转（图像 + 深度 + 内参）----
    #         if random.random() > 0.5:
    #             for k in list(inputs):
    #                 if "color" in k or "depth" in k:
    #                     for s in range(self.num_scales):
    #                         if (k[0], k[1], s) in inputs:
    #                             inputs[(k[0], k[1], s)] = inputs[(k[0], k[1], s)].transpose(Image.FLIP_LEFT_RIGHT)
    #             for s in range(self.num_scales):
    #                 if ("K", s) in inputs:
    #                     K = inputs[("K", s)].copy()
    #                     K[0, 2] = self.width // (2 ** s) - K[0, 2]
    #                     inputs[("K", s)] = K
    #     #↑ ======================================================

    #     # ============= v1v2v3公共后处理 =============
    #     for k in list(inputs):
    #         f = inputs[k]
    #         if "color" in k:
    #             n, im, i = k
    #             inputs[(n, im, i)] = self.to_tensor(f)
    #             inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
    
    # v4 垂直翻转
    # def preprocess(self, inputs, color_aug):
    #     """Resize colour images to the required scales and augment if required

    #     We create the color_aug object in advance and apply the same augmentation to all
    #     images in this item. This ensures that all images input to the pose network receive the
    #     same augmentation.
    #     """
    #     for k in list(inputs):
    #         frame = inputs[k]
    #         if "color" in k:
    #             n, im, i = k
    #             for i in range(self.num_scales):
    #                 inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

    #     # ======================================================
    #     # 仅当 --strong_aug 被指定时才启用增强
    #     # ======================================================
    #     strong_aug = getattr(self, "opt", None)
    #     strong_aug = getattr(strong_aug, "strong_aug", False)
    #     if self.is_train and strong_aug:
    #             for k in list(inputs):
    #                 if "color" in k:
    #                     n, im, i = k
    #                     # ① 垂直翻转（几何正确）
    #                     if random.random() > 0.5:
    #                         for s in range(self.num_scales):
    #                             inputs[(n, im, s)] = inputs[(n, im, s)].transpose(Image.FLIP_TOP_BOTTOM)
    #                         # 仅改 y 主点
    #                         for s in range(self.num_scales):
    #                             if ("K", s) in inputs:
    #                                 K = inputs[("K", s)].copy()
    #                                 K[1, 2] = self.height // (2 ** s) - K[1, 2]
    #                                 inputs[("K", s)] = K
    #     for k in list(inputs):
    #         f = inputs[k]
    #         if "color" in k:
    #             n, im, i = k
    #             inputs[(n, im, i)] = self.to_tensor(f)
    #             inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
    
    # ✅ 新版 V5_preprocess，支持 half_flip_aug
    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """

        # ====== 第一段：多尺度 resize（保持完全一致）======
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        # ====== 第二段：to_tensor + color_aug（保留原结构）======
        do_half_flip_aug = self.is_train and random.random() > 0.5  # 随机触发（更贴近 RobustDepth 风格

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k

                # 常规 tensor 转换
                inputs[(n, im, i)] = self.to_tensor(f)

                # 基础增强
                aug_img = color_aug(f)

                # ===== 新增：可选的“上半下半翻转拼接”增强 =====
                if do_half_flip_aug:
                    # 转成 tensor 再拼接
                    aug_tensor = self.to_tensor(aug_img)
                    _, H, W = aug_tensor.shape
                    mid = H // 2

                    # 分上下半部分
                    top = aug_tensor[:, :mid, :]
                    bottom = aug_tensor[:, mid:, :]

                    # 分别水平翻转
                    top_flip = torch.flip(top, dims=[2])
                    bottom_flip = torch.flip(bottom, dims=[2])

                    # 上下拼接
                    aug_tensor = torch.cat([top_flip, bottom_flip], dim=1)

                    inputs[(n + "_aug", im, i)] = aug_tensor
                else:
                    # 原版增强逻辑
                    inputs[(n + "_aug", im, i)] = self.to_tensor(aug_img)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            # color_aug = transforms.ColorJitter.get_params(
            #     self.brightness, self.contrast, self.saturation, self.hue)
            color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
