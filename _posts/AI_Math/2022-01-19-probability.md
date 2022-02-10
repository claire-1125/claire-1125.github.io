---
title: "[ Basic AI Math ]  Probability"
date: 2022-01-19
excerpt: "probability에 대해 가볍게 알아봅시다."
categories: 
    - AI Math
toc: true
toc_sticky: true
---



## 데이터의 확률분포

### 기본 용어

- $(x,y)\; ⁓\; \mathscr{D}$
    - 데이터 (**확률변수**) : $(x,y)∈\mathscr{X}×\mathscr{Y}$
    - 데이터 공간의 **확률분포** : $\mathscr{D}$
- $P(x.y)$는 $\mathscr{D}$를 모델링한다.
    - $P(x)$ : 입력 $x$에 대한 **‘주변’확률분포**
        - 이 자체로는 y에 대한 정보를 알 수 없다.
    - $P(x,y)$ : **결합분포**
- $P(x|y)$ : 조건부 확률분포


### 이산확률변수 & 연속확률변수

확률분포 $\mathscr{D}$에 따라 두 가지로 나뉜다.

- **이산확률변수**
    - 확률변수가 가질 수 있는 경우의 수 이용
        
        $$P(X∈A) = \sum_{x∈A} P(X=x)$$
        
    
- **연속확률변수**
    - 데이터 공간에 정의된 확률변수의 밀도(density) 이용
        
        $$P(X∈A) = \int_A P(x)dx$$
        



## 조건부확률과 기계학습

데이터에서 추출된 패턴 $\phi$을 기반으로 확률을 해석한다.

### 분류 문제

- Logistic Regression에서의 $softmax(W\phi + b)$
    
    - 데이터의 특징패턴 $\phi$과 weight $W$을 통해 $P(y|x)$ 계산
    

### 회귀 문제

- 조건부기댓값 $\mathbb{E}[y|x]$을 추정한다.

$$\mathbb{E}_{y⁓P(y|x)}[y|x] = \int_yyP(y|x)dy$$

- 기대값 (Expected Value)
    
    - 통계량 중 하나로 이를 통해 또 다른 통계량을 계산할 수 있다.
    
    - 연속확률분포
        
        $$\mathbb{E}_{x⁓P(x)}[f(x)]=\int_{\mathcal{X}}f(x)P(x)dx$$
        
    - 이산확률분포
        
        $$\mathbb{E}_{x⁓P(x)}[f(x)]=\sum_{x∈\mathcal{X}}f(x)P(x)$$
        

### 딥러닝

- MLP를 이용해서 특징패턴 $\phi$을 추출



## Monte Carlo Sampling

$$
\mathbb{E}_{x⁓P(x)}[f(x)] ≈ \frac{1}{N}\sum^N_{i=1}f(x^{(i)})
$$

- **확률분포를 모르는 상황**에서 기댓값을 계산할 때 사용하는 방식
- $x^{(i)}$는 **i.i.d.**를 따른다.
    - i.i.d. (independent & identically distribution; 독립항등분포)
        
        - 확률변수가 상호독립적이며 모두 동일한 확률분포를 가지는 상황
        
- 확률분포 유형(이산형, 연속형)에 상관없이 사용 가능하다.
- 기대값이 **ramdom하게 뽑은 샘플 N개의 평균**과 비슷하다.
- **i.i.d.**가 보장되면 대수의 법칙에 의해 **수렴을 보장**한다.
    - 대수의 법칙
        
        - 사건을 무한히 반복할 때 일정한 사건이 일어나는 비율은 횟수를 거듭하면 할수록 일정한 값에 가까워지는 법칙
        
- 예제
    
    ![Untitled](/assets/images/posts/AI_Math/probability/1.png){:width="400"}
    
    $$
    \frac{1}{2}\int_{-1}^1 e^{-x^{2}}dx ≈ \frac{1}{N}\sum_{i=1}^Nf(x^{(i)})
    $$
    
    - 단, $x^{(i)}⁓U(-1,1)$