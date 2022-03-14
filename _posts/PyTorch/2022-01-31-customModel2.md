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

<br/>

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

