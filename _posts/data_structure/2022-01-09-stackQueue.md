---
title: "[ 데이터구조 ]  Stack, Queue"
date: 2022-01-09
excerpt: "stack과 queue에 대해 알아봅시다."
categories: 
    - Data Structure
toc: true
toc_sticky: true
---

## 스택 (Stack)
![1.jpg](/assets/images/posts/data_structure/stackQueue/1.jpg){: width="400"}


- **LIFO (last-in first-out)** : 가장 늦게 들어간 얘가 가장 먼저 나간다. e.g.) 돌탑쌓기
- 스택의 추상 데이터 타입 (Stack ADT)
    - 데이터 : 유한한 길이를 가지는 순서 리스트
    - 연산 (진짜 중요한 기능만 넣음.)
        - **push** : top에서 새로운 원소 삽입
        - **pop** : top에서 원소 빼내며 삭제


## 큐 (Queue)
![2.jpg](/assets/images/posts/data_structure/stackQueue/2.jpg){: width="400"}


- **FIFO (first-in first-out)** : 뒤로 들어와서 앞으로 나간다. e.g.) 한줄서기
- 큐의 추상 데이터 타입 (Queue ADT)
    - 데이터 : 유한한 길이를 가지는 순서 리스트
    - 연산 (진짜 중요한 기능만 넣음.)
        - **enqueue** : rear에서 새로운 원소 삽입
        - **dequeue** : front에서 원소 빼내며 삭제

### 우선순위 큐 (priority queue)

- **가장 높은** 우선순위부터 **삭제**하는 큐
- 삽입할 때는 임의의 순서대로 진행.
- **들어올 땐 마음대로지만 나갈 땐 아니란다.**

### 덱 (Deque)
![3.jpg](/assets/images/posts/data_structure/stackQueue/3.jpg){: width="400"}

- double ended queue
- 큐의 **front와 rear**에서 모두 **삽입·삭제**가 가능한 큐
- 파이썬 collections 모듈 내 deque
    
    ```python
    from collections import deque
    
    deq = deque()
    
    # Add element to the start
    deq.appendleft(10)
    
    # Add element to the end
    deq.append(0)
    
    # Pop element from the start
    deq.popleft()
    
    # Pop element from the end
    deq.pop()
    ```