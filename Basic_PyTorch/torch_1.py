## 파이토치는 데이터의 기본 단위로 텐서(Tensor)를 사용한다
## 텐서는 다차원 배열(array)라고 정의할 수 있다

import torch
X = torch.Tensor(2, 3)
'''
torch.Tensor(2, 3)
이 코드는 2 x 3 shape를 가진 텐서를 생성한다
이때 임의의 난수가 들어가게 된다
텐서를 생성하면서 원하는 값으로 initialize하려면 인수로 배열을 전달한다(아래 코드)

torch.Tensor() 함수는 인수로
data, dtype, device, requires_grad를 받을 수 있다
data : 배열
dtype : 데이터를 저장할 자료형
device : 이 텐서를 어느 기기에 올릴 것인지 명시
requires_grad : 이 텐서에 대한 기울기를 저장할지 여부를 지정
'''
X = torch.Tensor([[1,2,3], [4,5,6]])
# print(X)

x_tensor = torch.tensor(data=[2.0, 3.0], requires_grad=True)
# print(x_tensor)

## 기울기 계산하기
x = torch.tensor(data=[2.0, 3.0], requires_grad=True)
y = x**2
z = 2*y+3

target = torch.tensor([3.0, 4.0])
loss = torch.sum(torch.abs(z-target))
loss.backward()

print(x.grad, y.grad, z.grad)
'''
위 코드의 해석
위 코드는 z = 2x^2 +3이라는 식에서 x에 대한 기울기를 구하는 단순한 코드이고,
1) x라는 텐서를 생성하며 기울기를 계산하도록 지정함(requires_grad=True)
2) z라는 변수에 연산 그래프의 결괏값이 저장됨
3) z와 목표값인 target의 절댓값(abs)차이를 계산하고
   torch.sum()이라는 함수를 통해3x4 모양이었던 두 값의 차이를
   숫자 하나로 바꿈
4) loss.backward() 함수를 통해 연산 그래프를 쭉 따라가면서
    노드 x에 대한 기울기를 계산한다
'''
