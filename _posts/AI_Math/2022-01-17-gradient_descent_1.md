---
title: "[ Basic AI Math ]  Gradient Descent - part 1"
date: 2022-01-17
excerpt: "Gradient Descent에 대해 알아봅시다."
categories: 
    - AI_Math
toc: true
toc_sticky: true
---


## 미분 (differentiation)

- 변화율의 극한 (순간변화량)
- 접선의 기울기
- python 미분 메소드 : sympy.diff()
    
    ```python
    import sympy as sym
    from aympy.abc import x
    
    sym.diff(sym.poly(x**2 + 2*x + 3),x)
    ```
    

## 경사하강법 (Gradient Descent)

$$
w ← w - α\frac{∂Loss}{∂w}
$$

- **극소값(local min.)** 찾기 (global minimum이라고 보장할 수는 없다.)
- 목적함수(혹은 loss function)를 **최소화**할 때 사용한다.
- gradient : 기울기의 변화량 중 가장 큰(작은) 값
- 알고리즘
    
    ```python
    '''
    input
    1.gradient : 그레디언트 계산
    2. init : 시작점
    3. lr : learning rate (보통 0.01로 설정한다.)
    4. eps : 종료조건
    '''
    '''
    output
    1. var : 결국엔 weight
    '''
    
    var = init
    grad = gradient(var)
    while norm(grad) > eps: # gradient가 eps 이하로 작아질 때까지 (즉, weight 거의 수렴) 진행
    	var = var - lr * grad
    	grad = gradient(var)
    ```