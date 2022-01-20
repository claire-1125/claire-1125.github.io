---
title: "[ Basic AI Math ]  Gradient Descent part 2"
date: 2022-01-17
excerpt: "Gradient Descent에 대해 좀 더 알아봅시다."
categories: 
    - AI Math
---


# 선형회귀
![1.png](/assets/images/posts/AI_Math/gradient_descent_2/1.png){: width="400"}

데이터가 분포되어 있을 때 **선형적(~직선)으로 근사**해서 데이터를 설명하려 한다.

- $Xβ$ : $y$(정확한 정답; 완전히 맞출 수는 없음.)와 근사한 모델
    - 상수항(bias)는 평행이동에 불과하므로 무시함.
- ${\underset{β}{min}} Ε||y-\hat{y}||_2$ : $y$(실제 정답)과 $\hat{y}$(만든 모델)의 **잔차 (여기서는 $L_2$ norm)가 최소한**이 되도록 하는 $\hat{y}$를 구해야 한다.  


# 경사하강법 기반으로 선형회귀 계수 구하기

$$
{\underset{β}{argmin}} Ε||y-Xβ||_2 
$$

## 기본 사항

 $y = \begin{vmatrix} 
y_1 \\
y_2 \\
... \\
y_n
\end{vmatrix}$,  $X = \begin{vmatrix} 
x_{11} & x_{12} & ... & x_{1m} \\
x_{21} & x_{22} & ... & x_{2m} \\
... & ... & ... & ...
\end{vmatrix}$,  $β = \begin{vmatrix} 
β_1 \\
β_2 \\
... \\
β_m
\end{vmatrix}$,  $Xβ = \begin{vmatrix} 
x_{11}β_1 + x_{12}β_2 +...+ x_{1m}β_m \\
x_{21}β_1 + x_{22}β_2 +...+ x_{2m}β_m \\  
... \\
x_{n1}β_1 + x_{n2}β_2 +...+ x_{nm}β_m \\
\end{vmatrix}$


## 이를 위한 gradient 벡터

- Gradient는 기울기 변화량의 최대(혹은 최소)가 되는 지점!
- 식 정리
    
    먼저 잔차의 gradient는 다음과 같다.
    
    $$
    ∇_β||y-Xβ||_2=(∂_{β_1}||y-Xβ||_2,...,∂_{β_d}||y-Xβ||_2)
    $$
    
    여기서 $∂_{β_k}||y-Xβ||_2$는 다음과 같이 쓸 수 있다.
    
    - 여기서의 $L_2$ norm은 기존 형태와 약간 다르다.
        - 선형회귀에서의 **cost function** : **잔차 제곱 합의 평균 (즉, MSE; Mean Squared Error)**
            
            $$
            \frac{1}{n}\sum_{i=1}^n(y_i-\sum_{j=1}^d X_{ij}β_j)^2
            $$
            
        - 이에 root를 씌운 것이 현재의 $L_2$ norm이다.
    
    $$
    ∂_{β_k}||y-Xβ||_2 = ∂_{β_k} \{   \frac{1}{n} \sum^n_{i=1} \{ y_i-\sum^d_{j=1}X_{ij}β_j \}^2 \}^{1/2}
    $$
    
    이제 본격적으로 chain rule을 이용해서 편미분해보자.
    
    ![2.jpg](/assets/images/posts/AI_Math/gradient_descent_2/2.jpg){: width="400"}
    
    현재까지 한 부분을 정리하면 다음과 같다.
    
    $$
    ∂_{β_k}||y-Xβ||_2 = -\frac{X^T_{·k}(y-Xβ)}{n||y-Xβ||_2}
    $$
    
    따라서 잔차의 gradient는 다음과 같이 표현할 수 있다.
    
    $$
    ∇_β||y-Xβ||_2
    =(∂_{β_1}||y-Xβ||_2,...,∂_{β_d}||y-Xβ||_2)\\
    =(-\frac{X^T_{·1}(y-Xβ)}{n||y-Xβ||_2},...,-\frac{X^T_{·d}(y-Xβ)}{n||y-Xβ||_2})\\
    =-X^T\frac{(y-Xβ)}{n||y-Xβ||_2}
    $$
    

## 드디어 Gradient Descent!

$$
β^{(t+1)}←β^{(t)}-λ∇_β||y-Xβ^{(t)}||
$$

방금 구한 gradient를 넣으면 다음과 같다.

$$
β^{(t+1)}←β^{(t)}+\frac{λ}{n}\frac{X^T(y-Xβ^{(t)})}{||y-Xβ^{(t)}||_2}
$$

만약 잔차가 아닌 **잔차 제곱**에 관해서 정리를 했다면 다음과 같이 더 간단하게 표현할 수 있다.

$$
β^{(t+1)}←β^{(t)}+\frac{2λ}{n}X^T(y-Xβ^{(t)})
$$



# 경사하강법 기반 선형회귀 알고리즘

```python
"""
<input>
X : 입력변수
y : 정답
lr : learning rate
T : 학습횟수

<output>
beta : 선형회귀 계수
"""

for t in range(T):
	error = y - (X @ beta)
	grad = - transpose(X) @ error
	beta = beta - lr * grad
```


# 경사하강법의 한계점

- diffirentiable & convex → convergent!
    
    e.g.) linear regression
    
- **non-linear** regression (실제는 이런 경우가 더 많다.) → **convergent...?**



# 확률적 경사하강법 (Stochastic Gradient Descent)

- 모든 데이터 $(X,y)$가 아닌 mini batch $(X_{(b)},y_{(b)})$만큼 사용해서 업데이트하는 방식

$$
β^{(t+1)}←β^{(t)}+\frac{2λ}{b}X^T_{(b)}(y_{(b)}-X_{(b)}β^{(t)})
$$

- 연산량이 $b/n$로 감소한다.

## Mini Batch

![2.png](/assets/images/posts/AI_Math/gradient_descent_2/3.png){: width="400"}

- 확률적으로 선택하므로 objective function 모양이 변하게 된다.
- 일반 gradient descent보다 ML 학습에 더 효율적이다.