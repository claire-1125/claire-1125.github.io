---
title: "[ PyTorch ]  Custom Model 제작 - part 4"
date: 2022-01-31
excerpt: "custom model 제작에 대해 알아봅시다."
categories: PyTorch
toc: true
toc_sticky: true
---



## Module flow

- 최소의 기능 단위인 **function** → function들로 이루어진 **layer** → layer로 이루어진 **model**

### Module flow 예시

- 앞으로는 이 예제를 기반으로 설명을 이어나갈 예정이다.

```python
### Function ###
class func_a(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
		
		# 모델 구조 출력시 멤버변수도 같이 출력시키기
		def extra_repr(self):
        return 'name={}'.format(self.name)

    def forward(self, x):
        x = x * 2
        return x

class func_b(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = Parameter(torch.Tensor([10]))
        self.W2 = Parameter(torch.Tensor([2]))

    def forward(self, x):
        x = x / self.W1
        x = x / self.W2
        return x

class func_c(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('temp', torch.Tensor([7]), persistent=True)

    def forward(self, x):
        x = x * self.temp
        return x

class func_d(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = Parameter(torch.Tensor([3]))
        self.W2 = Parameter(torch.Tensor([5]))

    def forward(self, x):
        x = x + self.W1
				x = x * 7
        x = x / self.W2
        return x

### Layer ###
class layer_ab(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = func_a('temp')
        self.b = func_b()

    def forward(self, x):
        x = self.a(x) / 5
        x = self.b(x)
        return x

class layer_cd(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = func_c()
        self.d = func_d()

    def forward(self, x):
        x = self.c(x)
        x = self.d(x) + 1
        return x

### Model ###
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.ab = layer_ab()
        self.cd = layer_cd()

		def __doc__(self):
        """
				여기에 모듈에 대한 docstring을 작성하면 됩니다.
				"""

    def forward(self, x):
        x = self.ab(x)
        x = self.cd(x)
        return x

x = torch.tensor([7])
model = Model()
model(x)

"""
출력결과 : tensor([6.5720], grad_fn=<AddBackward0>)
"""
```

## named_children vs. named_modules

### named_children

- **직계 후손**(한 단계 아래의 submodule)만 표시
- 각 모듈 객체의 **이름까지 출력**한다.

```python
for name, child in model.named_children():
    print(f"[ Name ] : {name}\n[ Children ]\n{child}")
    print("-" * 30)

"""
<출력결과>

[ Name ] : ab
[ Children ]
layer_ab(
  (a): func_a()
  (b): func_b()
)
------------------------------
[ Name ] : cd
[ Children ]
layer_cd(
  (c): func_c()
  (d): func_d()
)
------------------------------
"""
```

### named_modules

- **모든 후손들**(submodule) 표시
- 각 모듈의 이름까지 출력한다.

```python
for name, module in model.named_modules():
    print(f"[ Name ] : {name}\n[ Module ]\n{module}")
    print("-" * 30)

"""
<출력결과>
[ Name ] : 
[ Module ]
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
[ Name ] : ab
[ Module ]
layer_ab(
  (a): func_a()
  (b): func_b()
)
------------------------------
[ Name ] : ab.a
[ Module ]
func_a()
------------------------------
[ Name ] : ab.b
[ Module ]
func_b()
------------------------------
[ Name ] : cd
[ Module ]
layer_cd(
  (c): func_c()
  (d): func_d()
)
------------------------------
[ Name ] : cd.c
[ Module ]
func_c()
------------------------------
[ Name ] : cd.d
[ Module ]
func_d()
------------------------------
"""
```

## get_submodule

- 특정 모듈만 가져오고 싶을 때 사용한다.

```python
submodule = model.get_submodule("ab.a")  # 모듈 객체 이름으로 key값을 준다.
submodule.__class__.__name__

"""
출력결과 : func_a
"""
```

## named_parameters

- 어떤 파라미터를 만들었는지 확인할 때 사용한다.

```python
for name, parameter in model.named_parameters():
    print(f"[ Name ] : {name}\n[ Parameter ]\n{parameter}")
    print("-" * 30)

"""
출력결과
[ Name ] : ab.b.W1
[ Parameter ]
Parameter containing:
tensor([10.], requires_grad=True)
------------------------------
[ Name ] : ab.b.W2
[ Parameter ]
Parameter containing:
tensor([2.], requires_grad=True)
------------------------------
[ Name ] : cd.d.W1
[ Parameter ]
Parameter containing:
tensor([3.], requires_grad=True)
------------------------------
[ Name ] : cd.d.W2
[ Parameter ]
Parameter containing:
tensor([5.], requires_grad=True)
------------------------------
"""
```

## get_parameter

- 특정 파라미터만 가져오고 싶을 때 사용한다.

```python
parameter = model.get_parameter("ab.b.W1")
parameter

"""
출력결과 : 10
"""
```

## named_buffers

- 어떤 버퍼를 만들었는지 확인할 때 사용한다.

```python
for name, buffer in model.named_buffers():
    print(f"[ Name ] : {name}\n[ Buffer ] : {buffer}")
    print("-" * 30)

"""
<출력결과>

[ Name ] : cd.c.temp
[ Buffer ] : tensor([7.])
------------------------------
"""
```

## get_buffer

- 특정 버퍼만 가져오고 싶을 때 사용한다.

```python
buffer = model.get_buffer("cd.c.temp")
buffer

"""
출력결과 : 7
"""
```