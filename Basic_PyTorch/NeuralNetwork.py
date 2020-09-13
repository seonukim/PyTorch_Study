## 1. 패키지 로드하기
import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.init as init
import matplotlib.pyplot as plt
import numpy as np

## 2. 데이터의 수, 반복횟수 정의
num_data = 1000
num_epoch = 10000

## 3. 데이터 정의
noise = init.normal_(torch.FloatTensor(num_data, 1), std=1)
x = init.uniform_(torch.Tensor(num_data, 1), -15, 15)
y = (x**2) + 3
y_noise = y + noise

## 4. 모델 정의하기
model = nn.Sequential(
    nn.Linear(1, 6),
    nn.ReLU(),
    nn.Linear(6, 10),
    nn.ReLU(),
    nn.Linear(10, 6),
    nn.ReLU(),
    nn.Linear(6, 1),
)

loss_func = nn.L1Loss()
optimizer = opt.SGD(model.parameters(), lr=0.0002)

## 5. 모델 학습
loss_array = []
for i in range(num_epoch):
    optimizer.zero_grad()
    output = model(x)
    loss = loss_func(output, y_noise)
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print(loss)

    loss_array.append(loss)

plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
plt.plot(loss_array)

plt.subplot(2, 1, 2)
plt.scatter(x.detach().numpy(), y_noise, label='original data')
plt.scatter(x.detach().numpy(), output.detach().numpy(), label='model output')
plt.legend()
plt.show()
