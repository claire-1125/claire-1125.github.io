---
title: "[ PyTorch ]  PyTorch Basics - part 1"
date: 2022-01-30
excerpt: "PyTorch 기본적인 사용법에 대해 알아봅시다."
categories: PyTorch
toc: true
toc_sticky: true
---


## Numpy vs. PyTorch

- 보통 ndarray를 가지고 tensor를 만든다.

```python
data = [[3,5],[10,5]]
x_data = np.ndarray(data)
t_data = torch.from_numpy(x_data)
```

- 기본적으로 tensor가 가질 수 있는 data type은 **Numpy와 동일**하다.
- 대부분의 Numpy 연산이 **동일하게 적용**된다.  

<br/>

## PyTorch의 GPU 모드

- 특별히 선언하지 않으면 tensor는 기본적으로 CPU 모드이다.
- GPU 모드를 사용하기 위해선 선언을 해야 한다.
    - torch.Tensor.to() : 해당 텐서를 특정 연산 모드(CPU, GPU)와 특정 자료형으로 <br/>변환한다.

```python
if torch.cuda.is_available():
  x_data_cuda = x_data.to('cuda')
x_data_cuda.device  # 현재 텐서의 연산 모드 출력
```  

<br/>

## view vs. reshape

- **차원 재구성**을 하는 함수이다.
- contiguity 보장의 차이
    - view() : 같은 메모리 공간 공유, contiguous한 tensor에만 적용 가능
    - reshape() : contiguity 여부 상관없이 적용 가능

```python
tensor_ex = torch.rand(size=(2, 3, 2))  # shape : 2*3*2
tensor_ex.view([-1, 6])  # shape : 2*6
tensor_ex.reshape([-1,6])  # shape : 2*6
"""
출력결과 (두 가지 동일)
tensor([[0.3129, 0.3980, 0.4659, 0.3447, 0.9292, 0.6549],
        [0.3425, 0.3023, 0.6238, 0.4282, 0.3271, 0.2569]])
"""
```

- contiguous vs. non-contiguous
    - stride : 새로운 row로 가기 위해 3칸 skip, 새로운 column으로 가기 위해 1칸 <br/>skip해야 한다.
    - 실제로 메모리에는  `[0, 1, 2, 3, ...]` 라는 식으로 sequential하게 저장된다.
    
    ```python
    x = torch.arange(12).view(4, 3)
    print(x, x.stride())
    > tensor([[ 0,  1,  2],
            [ 3,  4,  5],
            [ 6,  7,  8],
            [ 9, 10, 11]]) 
    > (3, 1)
    ```
    
    - 실제 tensor 구조와 stride가 맞지 않는 경우 non-contiguous 하다고 한다.
    
    ```python
    y = x.t()
    print(y, y.stride())
    print(y.is_contiguous())
    > tensor([[ 0,  3,  6,  9],
            [ 1,  4,  7, 10],
            [ 2,  5,  8, 11]]) 
    > (1, 3)  # 기존의 stride를 단순히 swap한 형태이다. (실제로는 아님.)
    > False
    ```
 
<br/>

## squeeze vs. unsqueeze

### squeeze()

- redundant한 차원 삭제
- 즉, **차원의 개수가 1**인 차원을 **삭제**

### unsqueeze()

- redundant한 차원 추가
- 즉, **차원의 개수가 1**인 차원을 **추가**

<br/>

## matmul vs. mm vs. @ vs. dot vs. *

### matmul()

- matrix multiplication, vector multiplication 가능 (**만능**)
- **broadcasting 가능**

### mm()

- **matrix multiplication** 연산 가능
- **broadcasting 불가능**
    - 즉, matrix의 shape이 정확하게 동일해야 사용 가능

### @ (from Numpy)

- **matrix multiplication** 연산 가능

### dot (from Numpy)

- **내적** 연산 가능
- 벡터.dot(벡터)의 경우 실질적으로 matrix multiplication 진행
- 만약 **벡터 간의 ‘내적’**을 하고 싶다면 **np.inner()**

### * (from Numpy)

- **스칼라곱 (hadamard product)** : 같은 위치에 존재한 원소끼리의 곱
- **broadcasting 가능**

<br/>  
  
## [번외] 행렬곱 vs. 내적

### 개념상 차이

- 계산된 행렬의 각 원소 값은 ***벡터의 내적을 표현한 것***
- 행렬 곱은 내적의 집합

### 벡터 (1차 텐서)의 경우

- **행렬곱 = 내적**
- matmul() = mm() = @ = dot()

### 텐서 (2차 이상의 텐서)의 경우

- **행렬곱 ≠ 내적**
- **matmul() = mm() = @ ≠ dot()**


