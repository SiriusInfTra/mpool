import torch
import torchvision.models as models

import os

# 加载预训练的ResNet-152模型
model = models.resnet152(pretrained=True)
model.eval()

# 生成一个示例输入 (批次大小为1，3个通道，224x224的图片)
dummy_input = torch.randn(1, 3, 224, 224)

# 将模型转换为ONNX格式
torch.onnx.export(model, dummy_input, os.path.join(os.path.dirname(__file__), "resnet152.onnx"), verbose=True, input_names=['input'], output_names=['output'])
