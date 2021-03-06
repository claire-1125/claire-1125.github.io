---
title: "[ 프로그래머스 ]  입국심사 (Python)"
date: 2022-02-16
excerpt: "이진탐색(binary search)을 이용해 가장 빨리 입국심사를 끝낼 수 있는 시간을 찾아봅시다."
categories: 
    - CodingTest
toc: true
toc_sticky: true
---

## 기본 정보

- Level: 3
- Problem Link: [입국심사](https://programmers.co.kr/learn/courses/30/lessons/43238)
- My Solution Link: [나의 풀이](https://github.com/claire-1125/AlgoStudy/blob/main/Programmers/BinarySearch/Immigration.py)
- Algorithm I used : 이진탐색(binary search)

<br/>

## 문제

- 입국심사를 기다리는 사람 n 명
- 각 심사관이 한 명을 검사하는데 걸리는 시간(분)이 담긴 배열 times
- 한 심사대에서는 동시에 한 명만 심사
- 더 빨리 끝나는 곳으로 가서 심사 받을 수 있다.
- 모든 사람이 심사 받는데 걸리는 최소 시간?

<br/>

## 알고리즘

- 이진탐색으로 푸는 이유?
    - parametric search
        - **최적화** 문제 → **결정** 문제(Yes/No)
        - 특정한 조건을 만족하는 가장 알맞은 값을 빠르게 찾는 문제를 풀 때, 탐색 범위를 좁혀가며 각 범위 내에서 조건 만족 여부를 확인하는 방식으로 값을 찾는다.
- (예제 기준) 최대시간 6*10=10분
    - [1,60] 범위로 이진탐색 진행
- middle이라는 숫자까지 7의 배수 개수, 10의 배수 개수?
    - middle = 30, 7의 배수 = [7,14,21,28], 10의 배수 = [10,20,30]
    - 7 혹은 10의 배수 = [7,10,14,20,21,28,30] → 인원수 길이에 해당하는 숫자가 최소?

<br/>

## 1차 시도

- $O(log \mathbf{n})$인 알고리즘을 어쩌다보니 $O(n^2)$으로 만들어 버렸다.
- 정확성 테스트 탈락

```python
def binary_search(left, right, times, n):
    while left <= right:
        middle = (left + right) // 2

        here = []
        for elem in times:
            here.extend([elem*i for i in range(1, (middle//elem)+1)])

        here = sorted(here)

        if n <= len(here):
            right = middle - 1
        else:
            left = middle + 1

    return right

def solution(n, times):
    return binary_search(1, n*max(times), times, n)

print(solution(6,[7,10]))
```

<br/>

## 2차 시도

- binary_search 함수를 굳이 정의하지 않아도 될 것 같아서 solution 내부에 구현
- here이라는 리스트에 굳이 담지 않고, 그냥 개수만 counting하면 된다.

```python
def solution(n, times):
    left, right = 1, n*max(times)

    while left <= right:
        middle = (left + right) // 2

        cnt = 0
        for elem in times:
            cnt += middle // elem

        if n <= cnt:
            right = middle - 1
        else:
            left = middle + 1

    return left
```