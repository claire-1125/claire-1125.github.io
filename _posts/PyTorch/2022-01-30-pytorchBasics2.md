---
title: "[ PyTorch ]  PyTorch Basics - part 2"
date: 2022-01-30
excerpt: "PyTorch 기본적인 사용법에 대해 알아봅시다."
categories: PyTorch
toc: true
toc_sticky: true
---

## torch.tensor vs. torch.Tensor

### torch.tensor

- Python **function**
- **복사본** 사용

### torch.Tensor

- Python **class**
- **torch** 데이터 입력 → **원본** 사용
- **list, numpy** 데이터 입력 → **복사**하여 새롭게 torch.Tensor를 만든 후 사용

<br/>

## Indexing

tensor에서 원하는 값만 가져오려면?

### torch.index_select()

![1.jpg](/assets/images/posts/PyTorch/basics/1.jpg){: width="300"}

```python
A = torch.Tensor([[1, 2],[3, 4]])
A_prime = A.view(4)  # [1.0, 2.0, 3.0, 4.0]
output = torch.index_select(A_prime,0,torch.IntTensor([0,2])) # input, dim, index
# output = A_prime[[0,2]] # 리스트 인덱싱처럼 표현하기
 
output

"""
출력결과
torch.Tensor([1, 3])
"""
```

### torch.gather()

- **input and index** must have the **same number of dimensions.**
- index.size(d) <= input.size(d) for all dimensions d != dim.

#### 2-dimensional (1st order Tensor; Vector)

![2.jpg](/assets/images/posts/PyTorch/basics/2.jpg){: width="400"}

```python
A = torch.Tensor([[1, 2],[3, 4]])

# torch.tensor([0,1]).unsqueeze(0)  # [[0,1]] (shape:1*2)
# torch.tensor([0,1]).unsqueeze(1)  # [[0],[1]] (shape:2*1)

output = torch.gather(A,0,torch.tensor([0,1]).unsqueeze(0))  # input, dim, index
output = output.squeeze(0)

output

"""
출력결과
torch.Tensor([1, 4])
"""
```

#### 3-dimensional (2nd order Tensor; Tensor)

![3.jpg](/assets/images/posts/PyTorch/basics/3.jpg){: width="400"}

```python
# shape : 2*2*2
A = torch.Tensor([[[1, 2],
                   [3, 4]],
                  [[5, 6],
                   [7, 8]]])

output1 = torch.gather(A[0],0,torch.tensor([0,1]).unsqueeze(0)).squeeze(0)  # [1,4]
output2 = torch.gather(A[1],0,torch.tensor([0,1]).unsqueeze(0)).squeeze(0)  # [5,8]
output = torch.stack([output1,output2])  # torch.cat()이었다면 [1,4,5,8]

output

"""
출력결과
torch.Tensor([[1, 4], [5, 8]])
"""
```

<br/>

## torch.zeros() vs. torch.zeros_like()

### torch.zeros(shape)

- shape을 가지고 모든 element가 0인 tensor 생성

### torch.zeros_like(tensor)

- tensor와 동일한 shape을 가지고 모든 element가 0인 tensor 생성