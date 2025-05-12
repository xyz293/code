import torch
import torch.nn as nn

# 定义一个简单的全连接层
fc = nn.Linear(2, 1)

# 查看初始参数
print("Initial weights:", fc.weight.data)
print("Initial bias:", fc.bias.data)

# 构造一个输入
x = torch.tensor([[1.0, 2.0]])

# 前向传播
output = fc(x)
print("Output before training:", output)

# 简单训练一次
target = torch.tensor([[3.0]])
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(fc.parameters(), lr=0.1)

loss = criterion(output, target)
loss.backward()
optimizer.step()

# 查看更新后的参数
print("Updated weights:", fc.weight.data)
print("Updated bias:", fc.bias.data)

# 再次前向传播
print("Output after one update:", fc(x))