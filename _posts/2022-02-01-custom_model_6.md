---
title: "[ PyTorch ]  Custom Model 제작 - part 6"
date: 2022-02-01
excerpt: "custom model 제작에 대해 알아봅시다."
categories: PyTorch
toc: true
toc_sticky: true
---



## apply

- model **graph 전반**에 걸쳐 **custom 코드를 적용**시킬 때 사용한다.
- 일반적으로 **Weight Initialization**에 많이 사용한다.
- 호출하면 apply가 **적용된** module을 return 해준다.

### model tree 예제

- 앞으로는 이 예제를 기반으로 설명을 이어나갈 예정이다.

```python
### Function ###
class func_a(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.W = Parameter(torch.rand(1))

    def forward(self, x):
        return x + self.W

class func_b(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.W = Parameter(torch.rand(1))

    def forward(self, x):
        return x - self.W

class func_c(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.W = Parameter(torch.rand(1))

    def forward(self, x):
        return x * self.W

class func_d(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.W = Parameter(torch.rand(1))

    def forward(self, x):
        return x / self.W

### Layer ###
class layer_ab(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = func_a('plus')
        self.b = func_b('substract')

    def forward(self, x):
        x = self.a(x)
        x = self.b(x)
        return x

class layer_cd(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = func_c('multiply')
        self.d = func_d('divide')

    def forward(self, x):
        x = self.c(x)
        x = self.d(x)
        return x

### Model ###
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.ab = layer_ab()
        self.cd = layer_cd()

    def forward(self, x):
        x = self.ab(x)
        x = self.cd(x)
        return x

model = Model()
```

- model tree의 구조
    
    ![1.jpg](/assets/images/posts/PyTorch/custom_model/1.jpg){:width=300}
    

### apply의 적용 순서

- apply는 **post-order traversal** 방식으로 module에 함수를 적용한다.

```python
def print_module(module):
    print(module)
    print("-" * 30)

returned_module = model.apply(print_module)

"""
<출력결과>
func_a()
------------------------------
func_b()
------------------------------
layer_ab(
  (a): func_a()
  (b): func_b()
)
------------------------------
func_c()
------------------------------
func_d()
------------------------------
layer_cd(
  (c): func_c()
  (d): func_d()
)
------------------------------
Model(
  (ab): layer_ab(
    (a): func_a()
    (b): func_b()
  )
  (cd): layer_cd(
    (c): func_c()
    (d): func_d()
  )
)
------------------------------
"""
```

- post-order traversal?
    - [이진트리순회](https://claire-1125.github.io/%EB%8D%B0%EC%9D%B4%ED%84%B0%EA%B5%AC%EC%A1%B0/tree/#%EC%9D%B4%EC%A7%84%ED%8A%B8%EB%A6%AC-%EC%88%9C%ED%9A%8C)
    - 순회 순서 : func_a → func_b → layer_ab → func_c → func_d → layer_cd → Model
            
        ![1.jpg](/assets/images/posts/PyTorch/custom_model/1.jpg){:width=300}
            

### Weight Initialization

```python
model = Model()

# 모든 Parameter 값을 1로 초기화
def weight_initialization(module):
    module_name = module.__class__.__name__

    if module_name.split('_')[0] == "func":
        module.W.data.fill_(1.)

returned_module = model.apply(weight_initialization)

x = torch.rand(1)
output = model(x)
torch.isclose(output, x)  # output이 x에 가까운가?

"""
출력결과 : tensor([ True, True, True, True])
"""
```