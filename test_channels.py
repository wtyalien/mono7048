import torch
from networks.depth_encoder import LGFI  # 替换为实际路径
from networks.common.mca import MCA

# 1. 模拟输入数据
batch_size = 2
channels = 128  # 根据实际修改
height, width = 32, 32
dummy_input = torch.randn(batch_size, channels, height, width)

# 2. 初始化模块
lgfi = LGFI(dim=channels)  # 确保dim=输入通道数

# 3. 前向传播测试
print("="*50)
print("🔥 开始通道数测试 🔥")
print(f"输入形状: {dummy_input.shape}")

try:
    output = lgfi(dummy_input)
    print("✅ 测试通过！输出形状:", output.shape)
    assert output.shape == dummy_input.shape, "❌ 错误：输入输出形状不匹配！"
except Exception as e:
    print("❌ 测试失败！错误信息:")
    print(str(e))
    print("\n💡 调试建议:")
    print("- 检查LGFI中MCA初始化channels参数")
    print("- 验证forward中reshape/permute操作")