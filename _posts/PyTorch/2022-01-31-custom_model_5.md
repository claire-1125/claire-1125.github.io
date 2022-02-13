---
title: "[ PyTorch ]  Custom Model 제작 - part 5"
date: 2022-01-31
excerpt: "custom model 제작에 대해 알아봅시다."
categories: PyTorch
toc: true
toc_sticky: true
---


## hook

- **패키지화된 코드**에서 **custom 코드를 중간에 실행**시킬 수 있도록 만들어 놓은 **인터페이스**
- 사용 목적
    - 프로그램 실행 로직 분석
    - 프로그램 추가 기능 제공
- 종류
    - pre-hook  
      - **기존**의 프로그램이 **실행되기 전 수행**
    - hook  
      - **기존**의 프로그램이 **실행되고 난 후 수행**
- 사용 방법
    
    ```python
    # hook 함수 정의
    def hook_func():
    		"""
    		hook이 실행할 내용 작성
    		"""
    
    # hook을 사용할 객체(tensor, module) 생성
    model = Model()
    
    # 해당 객체에 hook 등록
    model.register_forward_hook(hook_func)
    ```

<br/>

## Tensor에 적용하는 hook

### backward hook
- 종류  
  - hook → **register_hook()**
- 예시
    
    ```python
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.W = Parameter(torch.Tensor([5]))
    
        def forward(self, x1, x2):
            output = x1 * x2
            output = output * self.W
            return output
    
    # 모델 생성
    model = Model()
    
    # Model의 Parameter W의 gradient 값을 저장하기 위한 list
    answer = []
    
    def tensor_hook(grad):
        answer.extend(grad)
    
    # 내가 정의한 hook을 실제로 모듈 내의 tensor(여기서는 weight)에 적용
    model.W.register_hook(tensor_hook)
    
    x1 = torch.rand(1, requires_grad=True)
    x2 = torch.rand(1, requires_grad=True)
    
    output = model(x1, x2)
    output.backward()
    
    if answer == [model.W.grad]:
        print(True)
    else:
        print(False)
    
    """
    출력결과 : True
    """
    ```
        

## Module에 적용하는 hook

### forward hook
- 종류
    - pre-hook → **register_forward_pre_hook()**
    - hook → **register_forward_hook()**
- 예시
    
    ```python
    class Add(nn.Module):
        def __init__(self):
            super().__init__() 
    
        def forward(self, x1, x2):
            output = torch.add(x1, x2)
            return output
    
    # 전파되는 output 값에 5를 더한다.
    def hook(module, output):
        output += 5
    
    # 모델 생성
    add = Add()
    
    # 내가 정의한 hook을 실제로 모듈에 적용하는 과정
    add.register_forward_hook(hook)
    
    x1, x2 = torch.rand(1), torch.rand(1)
    print(x1,",",x2)
    output = add(x1, x2)
    print(output)
    
    """
    실행결과 : x1 + x2의 값에 5가 더해진 값이 출력된다.
    """
    ```
        
### backward hook
- 종류  
  - hook → **register_full_backward_hook()**
- 예시
    
    ```python
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.W = Parameter(torch.Tensor([5]))
    
        def forward(self, x1, x2):
            output = x1 * x2
            output = output * self.W
    
            return output
    
    # 모델 생성
    model = Model()
    
    # x1.grad, x2.grad, output.grad 순서로 list에 넣는다.
    answer = []
    
    def module_hook(module, grad_input, grad_output):
        answer.extend(grad_input)
        answer.extend(grad_output)
    
    # 내가 정의한 hook을 실제로 모듈에 적용시키는 과정
    model.register_full_backward_hook(module_hook)
    
    x1 = torch.rand(1, requires_grad=True)
    x2 = torch.rand(1, requires_grad=True)
    
    output = model(x1, x2)
    output.retain_grad()
    output.backward()
    
    if answer == [x1.grad, x2.grad, output.grad]:
        print(True)
    else:
        print(False)
    
    """
    출력결과 : True
    """
    ```