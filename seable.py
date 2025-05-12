import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from model  import unet
from PIL import Image
import torchvision.transforms.functional as TF

# 加载模型
model = unet()  # 假设有 21 个类别
model.load_state_dict(torch.load('model.pth', map_location='cpu'))
model.eval()

# 加载测试图像
image_path = r"images\val_\ADE_val_00000004.jpg"
image = Image.open(image_path).convert("RGB")

# 定义预处理操作
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 预处理图像
image_tensor = transform(image).unsqueeze(0)  # 添加批量维度

# 模型推理
with torch.no_grad():
    output = model(image_tensor)

# 获取预测的类别标签
predicted_labels = torch.argmax(output, dim=1).squeeze(0).numpy()  # 去掉批量维度

# 定义颜色映射
cmap = plt.get_cmap('tab20', 21)  # 使用 21 种颜色

# 可视化原始图像和分割结果
plt.figure(figsize=(12, 6))

# 显示原始图像
plt.subplot(1, 2, 1)
plt.imshow(TF.to_pil_image(image_tensor.squeeze(0)))  # 将张量转换为 PIL 图像
plt.title("Original Image")
plt.axis('off')

# 显示分割结果
plt.subplot(1, 2, 2)
plt.imshow(predicted_labels, cmap=cmap, interpolation='nearest')
plt.title("Segmentation Mask")
plt.colorbar(ticks=range(21), label='Class')
plt.clim(-0.5, 20.5)
plt.axis('off')

plt.tight_layout()
plt.show()