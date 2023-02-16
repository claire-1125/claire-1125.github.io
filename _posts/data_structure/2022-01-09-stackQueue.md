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

### 스택 (Stack)
![1.jpg](/assets/images/posts/data_structure/stackQueue/1.jpg){: width="400"}


- **LIFO (last-in first-out)** : 가장 늦게 들어간 얘가 가장 먼저 나간다. e.g.) 돌탑쌓기
- 스택의 추상 데이터 타입 (Stack ADT)
    - 데이터 : 유한한 길이를 가지는 순서 리스트
    - 필수 연산
        - **push** : top에서 새로운 원소 삽입
        - **pop** : top에서 원소 빼내며 삭제
- **Python의 list**를 이용하여 구현할 수 있다.
- 구현예제
    
    ```python
    stack = []
    
    # 삽입(5) - 삽입(2) - 삽입(3) - 삽입(7) - 삭제() - 삽입(1) - 삽입(4) - 삭제()
    stack.append(5)
    stack.append(2)
    stack.append(3)
    stack.append(7)
    stack.pop()
    stack.append(1)
    stack.append(4)
    stack.pop()
    
    print(stack) # 최하단 원소부터 출력
    print(stack[::-1]) # 최상단 원소부터 출력
    ```

### 재귀 함수 (recursive function)

![recursive.jpg](/assets/images/posts/data_structure/stackQueue/recursive.jpg){: width="400"}

- 재귀함수 사용시 꼭 **종료조건**을 설정해야 한다! (그렇지 않으면 무한 호출)
- 예시 : 팩토리얼, 유클리드 호제법
- **스택** 구현 시 **재귀함수**를 이용하기도 한다.



## 큐 (Queue)

### 큐 (Queue)
![2.jpg](/assets/images/posts/data_structure/stackQueue/2.jpg){: width="400"}


- **FIFO (first-in first-out)** : 뒤로 들어와서 앞으로 나간다. e.g.) 한줄서기
- 큐의 추상 데이터 타입 (Queue ADT)
    - 데이터 : 유한한 길이를 가지는 순서 리스트
    - 필수 연산
        - **enqueue** : rear에서 새로운 원소 삽입
        - **dequeue** : front에서 원소 빼내며 삭제
- **Python의 deque 라이브러리**를 이용하여 구현할 수 있다.
- 구현예제
    
    ```python
    from collections import deque 
    
    # 큐(Queue) 구현을 위해 deque 라이브러리 사용
    queue = deque()
    
    # 삽입(5) - 삽입(2) - 삽입(3) - 삽입(7) - 삭제() - 삽입(1) - 삽입(4) - 삭제()
    queue.append(5)
    queue.append(2)
    queue.append(3)
    queue.append(7)
    queue.popleft()
    queue.append(1)
    queue.append(4)
    queue.popleft()
    
    print(queue) # 먼저 들어온 순서대로 출력
    queue.reverse() # 다음 출력을 위해 역순으로 바꾸기
    print(queue) # 나중에 들어온 원소부터 출력
    ```


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