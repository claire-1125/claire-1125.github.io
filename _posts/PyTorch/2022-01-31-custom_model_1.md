---
title: "[ PyTorch ]  Custom Model 제작 - part 1"
date: 2022-01-31
excerpt: "custom model 제작에 대해 알아봅시다."
categories: PyTorch
toc: true
toc_sticky: true
---


## torch.nn

- PyTorch의 **Neural Network 패키지**
- graph를 만들기 위한 basic building blocks

## nn.Linear()

- linear transformation ($y=xA^T+b$) 해주는 함수

```python
X = torch.Tensor([[1, 2],[3, 4]])

# in : ?*2 → out : ?*5
linear_transformed = nn.Linear(2,5)  # in_features, out_features

output = linear_transformed(X)
output.size()

"""
출력결과
torch.Size([2, 5])
"""
```


## torch.nn vs. torch.nn.functional

### torch.nn.functional

- python function

### torch.nn

- python class



## torch.nn.Module

- torch.nn 패키지의 container 중 하나
- 딥러닝을 구성하는 **Layer의 base class**
    - Input, Output, Forward, Backward 정의
    - 학습의 대상이 되는 parameter(tensor) 정의

### [번외] Python의 container 자료구조

- data type의 저장 모델로 종류에 무관하게 데이터를 저장할 수 있다.
- 속성과 기능을 함께 캡슐화함으로써 데이터를 담고 있는 객체 (일종의 **클래스**)
- python 내장 컨테이너 타입 : list, tuple, set, dictionary



## torch.nn.Parameter

- Tensor 객체의 상속 객체
- nn.Module 내의 attribute가 될 때는 **required_grad=True로 지정되어 학습 대상이 되는 Tensor**
- 직접 지정할 일은 거의 없다. (대부분의 layer에는 weights 값들이 지정되어 있다.)
- 파라미터 (weight, bias)를 그냥 tensor로 만들면 gradient를 계산하지 않아 값도 **업데이트 되지 않고**, 모델을 저장할 때 **무시된다.**



## Buffer

- 값을 업데이트 시킬 필요는 없지만 저장하고 싶은 경우 사용한다.

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.parameter = Parameter(torch.Tensor([7]))
        self.tensor = torch.Tensor([7])

				# 7이라는 tensor를 버퍼에 등록
        self.register_buffer("buffer",self.tensor)

model = Model()

try:
    buffer = model.get_buffer('buffer')
    if buffer == 7:
        print(model.state_dict())
    else:
        print("버퍼에 다른 값이 저장되어 있습니다.")
except:
    print("버퍼에 저장된 값이 없습니다.")
```



## Tensor vs. Parameter vs. Buffer

### Tensor

- ❌ gradient 계산
- ❌ 값 업데이트
- ❌ 모델 저장 시 값 저장

### Parameter

- ✅ gradient 계산
- ✅ 값 업데이트
- ✅ 모델 저장 시 값 저장

### Buffer

- ❌ gradient 계산
- ❌ 값 업데이트
- ✅ 모델 저장 시 값 저장

