import torch
import torch.nn as nn               ## 신경망 모델들이 포함된 패키지
import torch.optim as opt           ## 최적화 함수패키지
import torch.nn.init as init        ## 텐서에 초깃값을 주기 위한 패키지

num_data = 1000     ## 사용할 데이터의 갯수
num_epoch = 500     ## epoch 횟수

## 데이터 생성
x = init.uniform_(torch.Tensor(num_data,1), -10,10)          ## init.uniform_() : 균등하게 초기화
noise = init.normal_(torch.FloatTensor(num_data,1), std=1)
y = 2*x+3
y_noise = y + noise
'''
y는 종속변수이며, x의 값에 따라 -17 ~ 23 사이에 분포하게 된다
y에 노이즈를 추가하기 위해 y_noise 변수를 만들어줌
노이즈를 추가하는 이유는 일반적인 데이터는 보통 노이즈가 있기 때문(현실성 반영)
이때 노이즈는 표준정규분포를 따르는 노이즈를 사용하며, 이를 가우시안 노이즈라고 한다
이런 노이즈를 만들기 위해서는 y와 같은 모양을 가지는 노이즈 텐서를 만들어야 하기 때문에
[num_data, 1] 모양 텐서 noise를 init.normal_() 함수를 통해 초기화한다.
이때 평균은 default 0, std 1로 지정
'''

## 모델 및 손실함수 정의
model = nn.Linear(1,1)
loss_func = nn.L1Loss()
'''
파이토치의 선형회귀 모델은 nn.Linear() 함수로 구현되어 있다
Linear 클래스
- 들어오는 특성(feature)의 수
- 결과로 나오는 특성의 수
- 편차 사용 여부 .. 를 인수로 받아서 생성되고,
변수로는 가중치(weight)와 편차(bias)가 있다

우리가 만든 데이터 x는 1개의 특성을 가진 데이터 1000개이고
결과 y도 1개의 특성을 가진 데이터 1000개이기 때문에 Linear()의 인수로 1, 1을 생성

손실함수는 nn.L1Loss()를 사용했는데, 이것은 L1 손실을 뜻한다, 차이의 절댓값의 평균
'''

## 경사하강법을 위한 최적화 함수 정
optimizer = opt.SGD(model.parameters(), lr=0.01)
'''
경사하강의법을 사용하기 위해 torch.optim에서 SGD를 불러온다
optimizer란, 최적화 함수라고도 하며 경사하강법을 사용하여 오차를 줄이고
최적의 가중치와 편차를 근사할 수 있도록 해주는 역할을 한다
여러 최적화 함수 중 SGD는 Stochastic Gradient Descent의 약자로
한 번에 들어오는 데이터의 수대로 경사하강법 알고리즘을 적용하는 최적화 함수이다
학습률(learning rate, lr)도 인수로 전달해준다
'''

## 모델 training
loss_arr = []
label = y_noise
for i in range(num_epoch):
    optimizer.zero_grad()
    output = model(x)

    loss = loss_func(output, label)
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(loss.data)
        loss_arr.append(loss.detach().numpy())
param_list = list(model.parameters())
print(param_list[0].item(), param_list[1].item())

import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 15))
plt.subplot(2, 1, 1)
plt.scatter(x.numpy(), y_noise.numpy(), s=7, c='grey')
plt.axis([-12, 12, -25, 25])

plt.subplot(2, 1, 2)
plt.scatter(x.numpy(), y_noise.numpy(), s=7, c='grey')
plt.scatter(x.detach().numpy(), output.detach().numpy(), s=7, c='red')
plt.axis([-10, 10, -30, 30])
plt.show()

plt.plot(loss_arr)
plt.show()
