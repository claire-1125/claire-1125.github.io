---
title: "[ 딥러닝 ]  Minimax Algorithm"
date: 2022-01-10
excerpt: "GAN의 상위개념인 Minimax Algorithm에 대해 알아봅시다."
categories: 
    - Deep Learning
toc: true
toc_sticky: true
---


## Background Knowledge

### Intelligent (Rational) Agent란 무엇인가?

![1.png](/assets/images/posts/deep_learning/minimax/1.png){: width="400"}

- agent가 하는 일
    - perceive its environment through sensor (주변 환경 인지)
    - act upon that environment through actuators (적절한 action 취함)


### ‘Rational’한 Agent

1. 충분한 양의 information을 gather해서
2. 1을 가지고 계속 학습해서 agent ftn.을 개선시킴으로써
3. 궁극적으로 performance measure를 maximize하도록 하는 agent
- agent는 omniscient (전지전능)하지 않다! (rationality ≠ perfection)
- agent는 performance measure를 **maximize할 것으로 예상되는** action을 취한다.


### 특성에 따른 task environment의 구분  

agent는 environment와의 상호작용을 통해 action을 취하므로 환경이 어떠한 특성을 지녔는지 파악하는 것은 매우 중요하다.

- 현재 상황 : **‘competitive’ ‘multi’**-agent
    
    본인의 performance를 maximize시키도록 **상대방의 performance를 minimize**시키는 방식
    
    c.f.) 용어 설명
    
    - multi-agent : agent가 여러 개인 상황 e.g.) adversarial search
    - competitive multi-agent : zero-sum
    - cooperative multi-agent : 서로 협력
    



## Minimax Algorithm

![2.png](/assets/images/posts/deep_learning/minimax/2.png){: width="400"}

MAX 기준으로 tree 그림 (MAX 이기는 경우에 utility가 +1)

### 필요한 요소
- $s_0$ : initial state
- PLAYER(s) : 현재 state에서 action 취할 수 있는 사람 i.e.) MAX or MIN
- ACTIONS(s) : 현재 state에서 가능한 action들
- RESULT(s,a) : state에서 action 취한 결과
- TERMINAL_TEST(s) : 지금 state가 terminal (게임 끝)인가?
- UTILITY(s,p) : state $s$, player $p$일 때 utility ftn.값.
    - 이기면 +1, 지면 0, 비기면 +1/2


### Optimal Decisions in Games : Minimax Algo.
- optimal decision : 상대방이 (그 선택권 안에서) 최선의 선택을 한다고 가정

    ![3.jpg](/assets/images/posts/deep_learning/minimax/3.jpg){: width="400"}
