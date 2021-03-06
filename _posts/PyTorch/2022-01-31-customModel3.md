---
title: "[ PyTorch ]  Custom Model 제작 - part 3"
date: 2022-01-31
excerpt: "custom model 제작에 대해 알아봅시다."
categories: PyTorch
toc: true
toc_sticky: true
---


## torch.nn.backward

- Layer에 있는 **Parameter들의 미분**을 수행
- Forward의 결과값 (=model의 output=예측치)과 실제값 간의 **차이(loss) 에 대해 <br/>미분**을 수행
    - 해당 값으로 **Parameter 업데이트**

<br/>

## AutoGrad

$$
y = w^2 
\\ \, \\
z = 10*y + 25 = 10*w^2 + 25 
\\ \, \\
\frac{\partial z}{\partial w} = \frac{\partial z}{\partial y}\frac{\partial y}{\partial w} = 10*2w = 20w
$$

```python
w = torch.tensor(2.0, requires_grad=True)
y = w**2
z = 10*y + 25
z.backward()  # backward() : 현재 tensor의 gradient 계산
w.grad  # gradient 함수값
```

<br/>

## Linear Regression with AutoGrad

### train set 생성

```python
import numpy as np

# train set 생성
x_values = [i for i in range(11)]
x_train = np.array(x_values, dtype=np.float32).reshape(-1,1)

y_values = [2*i + 1 for i in x_values]
y_train = np.array(y_values, dtype=np.float32).reshape(-1,1)

"""
x_train = [[ 0.],[ 1.],[ 2.],[ 3.],[ 4.],[ 5.],[ 6.],[ 7.],[ 8.],[ 9.],[10.]]
y_train = [[ 1.],[ 3.],[ 5.],[ 7.],[ 9.],[11.],[13.],[15.],[17.],[19.],[21.]]
"""
```

### Neural Net 모델 생성 - Linear Regression

```python
import torch
from torch import nn
from torch.autograd import Variable

class LinearRegression(nn.Module):
    def __init__(self, inputSize, outputSize):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)  # linear transformation

    def forward(self, x):
        return self.linear(x)
```

### train 사전 세팅

```python
# train set이 column vector 형태이므로 모두 1로 설정한다.
inputDim = 1   
outputDim = 1 

# gradient descent를 하기 위해 필요한 변수들
learningRate = 0.01
epochs = 100

# neural net 객체
model = LinearRegression(inputDim, outputDim)

##### For GPU #######
if torch.cuda.is_available():
    model.cuda()

# MSE : mean squared error
criterion = torch.nn.MSELoss() 
# SGD : Stochastic Gradient Descent
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)  
```

### train mode (using AutoGrad)

```python
for epoch in range(epochs):
    # input, label을 Variable 객체로 변경
    if torch.cuda.is_available():
        inputs = Variable(torch.from_numpy(x_train).cuda())
        labels = Variable(torch.from_numpy(y_train).cuda())
    else:
        inputs = Variable(torch.from_numpy(x_train))
        labels = Variable(torch.from_numpy(y_train))

    # 이전에 계산했던 gradient값은 계속 필요한 게 아니므로 매 epoch 때마다 초기화해준다.
    # (축적된 gradient를 사용할 수는 없지...!)
    optimizer.zero_grad()

    # linear regression 모델 돌린 결과 (사실상 linear transformation)
    outputs = model(inputs)

    # MSEloss 계산
    loss = criterion(outputs, labels)
    print(loss)

    # parameter의 gradient 계산
    loss.backward()

    # parameter 업데이트 (SGD 이용)
    optimizer.step()

    print('epoch {}, loss {}'.format(epoch, loss.item()))

"""
출력결과
tensor(173.0244, grad_fn=<MseLossBackward0>)
epoch 0, loss 173.02435302734375
tensor(14.4643, grad_fn=<MseLossBackward0>)
epoch 1, loss 14.464330673217773
tensor(1.5272, grad_fn=<MseLossBackward0>)
epoch 2, loss 1.5271694660186768
...
tensor(0.1288, grad_fn=<MseLossBackward0>)
epoch 98, loss 0.1288181096315384
tensor(0.1274, grad_fn=<MseLossBackward0>)
epoch 99, loss 0.1273796111345291
"""
```

### test mode

- 원래는 test set을 넣어야 하지만 여기서는 그냥 train set을 넣었다.
- 이를 가지고 metric (e.g. accuracy, precision)을 계산해서 좋은 모델인지 판단할 수 있다.

```python
with torch.no_grad():
    if torch.cuda.is_available():
        predicted = model(Variable(torch.from_numpy(x_train).cuda())).cpu().data.numpy()
    else:
        predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
    print(predicted)
```

### Linear Regression으로 찾은 parameter 조합 확인하기

- linear regression의 목적은 데이터를 가장 근사하게 fitting 시킬 수 있는 선형 모델의 parameter를 찾는 것이므로 이를 확인해본다.

```python
for p in model.parameters():
    if p.requires_grad:
         print(p.name, p.data)

"""
출력결과
None tensor([[2.0956]]) → weight
None tensor([0.3361]) → bias
"""
```