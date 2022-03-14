---
title: "[ Basic AI Math ]  Statistics"
date: 2022-01-21
excerpt: "통계학에 대해 가볍게 알아봅시다."
categories: 
    - AI_Math
toc: true
toc_sticky: true
---



## 모수란 무엇인가?

### 모수적(parametric) 방법론

데이터가 **특정 확률분포**를 따른다고 **선험적으로 가정**한 후 그 분포를 결정하는 모수(parameter)를 추정하는 방법

### 비모수적(non-parametric) 방법론

특정 확률분포를 가정하지 않고 **데이터에 따라** 모델의 구조 및 모수의 개수가 **유연하게 바뀌는** 경우

## 확률분포 가정하기

Histogram을 보고 판단한다.

### 베르누이 분포

데이터가 2개의 값(0,1)만 가지는 경우

### 카테고리 분포

데이터가 n개의 이산적인 값을 가지는 경우

### 베타 분포

데이터가 [0,1] 사이에서 값을 가지는 경우

### 감마 분포, 로그 정규분포

데이터가 0 이상의 값을 가지는 경우

### 정규분포, 라플라스분포

데이터가 $\mathbb{R}$ 전체에서 값을 가지는 경우




## 모수 추정하기 - 정규분포

### 용어

- 모평균 $μ$
- 모분산 $σ^2$
- 표본평균 $\overline{X}$
- 표본분산 $S^2$

### 관계식

$$
\overline{X}=\frac{1}{N}\sum_{i=1}^N X_i
\\ \, \\
\mathbb{E}[\overline{X}]=μ
$$

$$
S^2=\frac{1}{N-1}\sum_{i=1}^N (X_i-\overline{X})^2
\\ \, \\
\mathbb{E}[S^2]=σ^2
$$

- 표본분산을 구할 때 $N-1$로 나누는 이유?
    - 표본의 분산은 모집단의 분산을 underestimate하여 ‘표본의 분산 < 모집단의 분산’과 같은 상태가 되기 때문에 이를 보정해주기 위해서 표본분산의 분모를 작게 만들어 전체 표본분산을 크게 만들었다.
    - 샘플 분산에서는 **degrees of freedom가 n-1**이기 때문이다.
        - degree of freedom : 표본 수 - (사용한) 통계량 수
    - 표본분산의 기대치가 수학적으로 정확하게 모분산이므로 n 대신 n-1로 나누어 준다.

### 중심극한정리 (Central Limit Thm.)

- 표본평균$\overline{X}$의 분포는 $N$이 커질수록 정규분포 $\mathscr{N}(μ,σ^2/N)$를 따른다.
- 모집단의 분포가 정규분포를 따르지 않아도 성립한다.

## Maximum Likelihood Estimation (MLE; 최대우도법)

$$
\hat{\theta}_{MLE}=\underset{\theta}{argmax}L(\theta;\mathbf{x})=\underset{\theta}{argmax}P(\mathbf{x}|\theta)
$$

- **모수적인(parametrix)** 데이터 밀도 추정 방법
- 파라미터 $\theta=(\theta_{1},...,\theta_{m})$로 구성된 어떤 확률밀도함수 $P(\mathbf{x}|\theta)$에서 관측된 표본 데이터 집합을 $x=\{x_1,...,x_n\}$이라 할 때, 이 표본들에서 파라미터 $\theta=(\theta_{1},...,\theta_{m})$를 추정하는 방법
- $L(\theta;\mathbf{x})$ : 가능도 함수
- 데이터 집합 $\mathbf{X}$가 독립추출일 경우 **log likelihood**로 변경해서 생각할 수 있다.
    
    $$
    L(\theta;\mathbf{X})=\prod_{i=1}^nP(\mathbf{x}_i|\theta) \, → \, logL(\theta;\mathbf{X})=\sum_{i=1}^nlogP(\mathbf{x}_i|\theta)
    $$
    
    - 여기서의 $log$는 자연로그 $ln$을 의미한다.
    - 로그 가능도를 사용하면 연산량을 $O(n^2)$에서 $O(n)$으로 감소시킬 수 있다.

### 가능도(Likelihood)와 가능도함수

- 가능도(우도) : **가정된 분포**에서 **주어진 데이터**가 나올 가능성 (특정 확률값을 가질 가능성)
- 가능도 ≠ 확률
    
    e.g.) 동전 던지기를 10번 해서 앞면이 4번 나왔다. 
    
    - 확률 : 앞면이 나올 확률은 0.4이다. 즉, 확률은 실험 결과를 집계한 것.
    - 가능도 : 동전 던지기가 **이항분포**를 따른다고 **가정하고** 앞면이 나올 확률에 따라 **10번 중 앞면이 4번 나올 가능성**
- 가능도(Likelihood) 함수 : 가능도 값을 계산하는 함수

### 예제 - 정규분포

- 정규분포를 따르는 확률변수 $X$로부터 독립표본 $\{x_1,...,x_n \}$을 얻었다.
- 계산하기
    
    정규분포에서의 모수는 모평균 $\mu$, 모분산 $\sigma^2$이므로 다음과 같이 적을 수 있다.
    
    $$
    \hat{\theta}_{MLE}=\underset{\theta}{argmax}L(\theta;\mathbf{x})=\underset{μ,\sigma^2}{argmax}P(\mathbf{X}|μ,\sigma^2)
    $$
    
    여기서 log likelihood를 구해보자. 정규분포의 확률밀도함수(Probability Density Function; PDF)는 **Gaussian Function**이므로 다음과 같이 표현할 수 있다.
    
    $$
    logL(\theta;\mathbf{X})=\sum_{i=1}^nlogP(\mathbf{x}_i|\theta)=\sum_{i=1}^nlog\frac{1}{ \sqrt{2\pi\sigma^2}}e^{-\frac{|x_i-μ|^2}{2\sigma^2}}
    $$
    
    이를 좀 더 간단하게 정리하면 다음과 같다. 여기서의 $log$는 $ln$임을 기억하자.
    
    $$
    \sum_{i=1}^nlog\frac{1}{ \sqrt{2\pi\sigma^2}}e^{-\frac{|x_i-μ|^2}{2\sigma^2}}
    \\ \, \\
    \sum_{i=1}^n\{ log\frac{1}{\sqrt{2\pi\sigma^2}} + log \,e^{-\frac{|x_i-\mu|^2}{2\sigma^2}} \}
    \\ \, \\
    \sum_{i=1}^n \{ -\frac{1}{2}log(2\pi\sigma^2) -\frac{|x_i-\mu|^2}{2\sigma^2}log\,e \}
    \\ \, \\
    =-\frac{n}{2}log2\pi\sigma^2-\sum_{i=1}^n\frac{|x_i-μ|^2}{2\sigma^2}
    $$
    
    이제 likelihood가 최대가 될 수 있도록 하는 파라미터($\mu, \sigma^2$)를 찾아보자. 
    
    $$
    logL(\theta;\mathbf{X})=-\frac{n}{2}log2\pi\sigma^2-\sum_{i=1}^n\frac{|x_i-μ|^2}{2\sigma^2}
    $$
    
    어떤 함수의 최댓값을 구할 때 **극값**을 이용하는 경우가 많다. 따라서 양변을 편미분하자.
    
    - 모평균 $\mu$에 대해 계산
    
    $$
    \frac{\partial}{\partial\mu}logL(\theta;\mathbf{X})=0
    \\ \, \\
    LHS=\frac{\partial}{\partial\mu}(-\frac{n}{2}log2\pi\sigma^2-\sum_{i=1}^n\frac{|x_i-μ|^2}{2\sigma^2})
    =-\sum_{i=1}^n\frac{x_i-\mu}{\sigma^2}
    \\ \, \\
    → -\sum_{i=1}^n\frac{x_i-\mu}{\sigma^2}=0\, ···①
    $$
    
    - 모분산 $\sigma^2$에 대해 계산
    
    $$
    \frac{\partial}{\partial\sigma}logL(\theta;\mathbf{X})=0
    \\ \, \\
    LHS=\frac{\partial}{\partial\sigma}(-\frac{n}{2}log2\pi\sigma^2-\sum_{i=1}^n\frac{|x_i-μ|^2}{2\sigma^2})
    \\ \, \\
    \frac{\partial}{\partial\sigma}(-\frac{n}{2}log2\pi\sigma^2)=-\frac{n}{2}\frac{1}{(2\pi\sigma^2)\,ln\,e}\,4\pi\sigma=-\frac{n}{\sigma}
    \\ \, \\
    \frac{\partial}{\partial\sigma}(-\sum_{i=1}^n\frac{|x_i-μ|^2}{2\sigma^2})=-\sum_{i=1}^n\frac{|x_i-\mu|^2}{2}(-2\frac{1}{\sigma^3})=\frac{1}{\sigma^3}\sum_{i=1}^n|x_i-\mu|^2
    \\ \, \\
    → -\frac{n}{\sigma}+\frac{1}{\sigma^3}\sum_{i=1}^n|x_i-\mu|^2 =0\,···②
    $$
    
    위에서 구한 두 식이 성립되는 $\mu$, $\sigma^2$을 구하면 된다.
    
    $$
    ① -\sum_{i=1}^n\frac{x_i-\mu}{\sigma^2}=0 → \sum_{i=1}^n(x_i-\mu)=0 → \sum_{i=1}^n x_i=n\,\mu
    \\ \, \\
    ∴\hat{\mu}_{MLE} = \frac{1}{n}\sum_{i=1}^n x_i=\overline{X}
    $$
    
    $$
    ② -\frac{n}{\sigma}+\frac{1}{\sigma^3}\sum_{i=1}^n|x_i-\mu|^2 =0 → \sum_{i=1}^n|x_i-\mu|^2=n\,\sigma^2
    \\ \, \\
    ∴ \hat{\sigma^2}_{MLE}=\frac{1}{n}\sum_{i=1}^n (x_i-\mu)^2 = \frac{1}{n}\sum_{i=1}^n (x_i-\overline{X})^2
    $$
    

### 예제 - 카테고리 분포

## 딥러닝에서 최대우도법

### 분류문제

- weight $\theta=(\operatorname{W}^{(1)},...,\operatorname{W}^{(L)})$
- softmax 벡터 : MLE 이용해서 categorical 분포의 모수 $(p_1,...,p_k)$를 모델링

## 확률분포의 거리

- Loss function은 이 둘 간의 거리를 통해 MLE를 유도한다.
    - **모델이 학습**하는 확률분포 $P(\mathbf{x})$
    - 데이터에서 **관찰**되는 확률분포 $Q(\mathbf{x})$
- 거리를 계산할 때 사용하는 함수들
    - Total Variance Distance (TV)
    - Wasserstein Distance
    - **Kullback-Leibler Divergence (KL)**

### Kullback-Leibler Divergence

- 정의
    - 이산확률변수
        
        $$
        \mathbb{KL}(P\|Q)=\sum_{\mathbf{x}∈}P(\mathbf{x})log(\frac{P(\mathbf{x})}{Q(\mathbf{x})})
        
        $$
        
    - 연속확률변수
        
        $$
        \mathbb{KL}(P\|Q)=\int_{s}P(\mathbf{x})log(\frac{P(\mathbf{x})}{Q(\mathbf{x})})d\mathbf{x}
        
        $$