---
title: "[ PyTorch ]  Custom Model 제작 - part 2"
date: 2022-01-31
excerpt: "custom model 제작에 대해 알아봅시다."
categories: PyTorch
toc: true
toc_sticky: true
---


## custom layer

- 생성자 내부에서 super 클래스의 생성자를 사용한다는 걸 표현할 때...
    - python 2.x 표현
        
        ```python
        super(ClassName, self).__init__()
        super(ClassName, self).__init__(**kwargs)
        ```
        
    - python 3.x 표현
        
        ```python
        super().__init__()
        super().__init__(**kwargs)
        ```
        
- 예시

```python
class MyLiner(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weights = nn.Parameter(torch.randn(in_features, out_features))
        # self.weights = Tensor(torch.randn(in_features, out_features))
        
        self.bias = nn.Parameter(torch.randn(out_features))
        # self.bias = Tensor(torch.randn(out_features))

    def forward(self, x : Tensor):  # parameter에 colon(:) : 파라미터 자료형 명시
        return x @ self.weights + self.bias  # linear model

x = torch.randn(5, 7)

layer = MyLiner(7, 12)  # layer 생성
# layer(x)
layer(x).shape
"""
출력결과
torch.Size([5, 12])
"""

# weight, bias 정보 출력 
for value in layer.parameters():
    print(value)
```

```python
class Add(nn.Module):
    # 생성자 : 모델에서 사용될 module, activation function 등을 정의
    def __init__(self):
        # super() 사용해서 override 방지
        super(Add, self).__init__()

    # feed forward
    def forward(self, x1, x2):
        return torch.add(x1,x2)
        

x1, x2 = torch.tensor([1]), torch.tensor([2]) 

add = Add()  # 객체 생성 시 __init__ 실행
output = add(x1, x2)  # forward() 함수 실행
output

""" 
실행결과 : 3
"""
```

## Module들을 묶어서 사용해보자 - Container

### torch.nn.Sequential

- 묶어 놓은 모듈들을 차례대로 수행한다.

```python
class Add(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return x + self.value

# y = x + 3 + 2 + 5
calculator = nn.Sequential(
    Add(3), 
    Add(2),
    Add(5)
)

x = torch.tensor([1])
output = calculator(x)  # forward 실행 (근데 Sequential을 곁들인)
output

"""
실행결과 : 11
"""
```

### torch.nn.ModuleList

- 모아두기만 하고 indexing으로 원하는 Module 접근하려 할 때 사용한다.

```python
class Add(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return x + self.value

class Calculator(nn.Module):
    def __init__(self):
        super().__init__()
        self.add_list = nn.ModuleList([Add(3), Add(2), Add(5)])  # Add 생성자 실행

    def forward(self, x):
        # y = ((x + 3) + 2) + 5 
        result = x
        for i in range(len(self.add_list)):
            result = self.add_list[i](result)

        return result

x = torch.tensor([1])
calculator = Calculator()
output = calculator(x)
output

"""
실행결과 : 11
"""
```

### torch.nn.ModuleDict

- 모듈 개수가 많아질 경우 key값을 이용해서 Module에 접근하는 방식으로 이용할 수 있다.

```python
class Add(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return x + self.value

class Calculator(nn.Module):
    def __init__(self):
        super().__init__()
        self.add_dict = nn.ModuleDict({'add2': Add(2), 'add3': Add(3), 'add5': Add(5)})

    def forward(self, x):
        # y = ((x + 3) + 2) + 5
        x = self.add_dict['add3'](x)
        x = self.add_dict['add2'](x)
        x = self.add_dict['add5'](x)

        return x

x = torch.tensor([1])
calculator = Calculator()
output = calculator(x)
output

"""
실행결과 : 11
"""
```

### Module Container 자료형을 사용하는 이유?

- 일반 list, dict와 달리 모델의 **파라미터를 전달할 수 있다.**
    - 파라미터를 업데이트하기 위해서는 이에 대한 내용이 전달되어야 한다!



## torch.nn.backward

- Layer에 있는 **Parameter들의 미분**을 수행
- Forward의 결과값 (=model의 output=예측치)과 실제값 간의 **차이(loss) 에 대해 미분**을 수행
    - 해당 값으로 **Parameter 업데이트**



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