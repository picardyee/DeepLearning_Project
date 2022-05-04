import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


# 선형 회귀 모델(Linear Regression Model)

# 데이터 생성
X = torch.randn(200, 1) * 10
Y = X + 3 * torch.randn(200, 1)
plt.scatter(X.numpy(), Y.numpy())
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()

# 모델 정의 및 파라미터
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.Linear = nn.Linear(1, 1)

    def forward(self, x):
        pred = self.Linear(x)
        return pred

model = LinearRegressionModel()
print(model)
print(list(model.parameters()))

w, b = model.parameters()
w1, b1 = w[0][0].item(), b[0].item()
x1 = np.array([-30, 30])
y1 = x1 * w1 + b1

plt.plot(x1, y1, 'r')
plt.scatter(X, Y)
plt.grid()
plt.show()

# 손실 함수 및 옵티마이저
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 모델 학습
epochs = 100
losses = []

for epoch in range(epochs):
    optimizer.zero_grad()

    y_pred = model(X)
    loss = criterion(y_pred, Y)
    losses.append(loss.item())
    loss.backward()

    optimizer.step()

plt.plot(range(epochs), losses)
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.show()

# 결과 확인
w1, b1 = w[0].item(), b[0].item()
x1 = np.array([-30, 30])
y1 = x1 * w1 + b1

plt.plot(x1, y1, 'r')
plt.scatter(X, Y)
plt.grid()
plt.show()