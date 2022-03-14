---
title: "[ 프로그래머스 ]  소수 찾기 (Python)"
date: 2022-02-09
excerpt: "완전탐색(brute force)을 이용해서 소수를 찾아봅시다."
categories: 
    - Coding_Test
toc: true
toc_sticky: true
---


## 기본 정보

- Level: 2
- Problem Link: [소수 찾기](https://programmers.co.kr/learn/courses/30/lessons/42839)
- My Solution Link: [나의 풀이](https://github.com/claire-1125/AlgoStudy/blob/main/Programmers/BruteForce/search_prime_number.py)
- Algorithm I used : 완전탐색(brute force)

<br/>

## 문제

- numbers : 각 종이 조각에 적힌 숫자가 적힌 문자열
    - 길이 : 1~7
- 11 == 011
- 한 개, 두 개, .., 전부 사용해서 조합 가능
- 종이 조각으로 만들 수 있는 prime number 개수?

<br/>

## 알고리즘

- numbers 속 숫자 나누기
    - 문자열 각각을 list에 저장
- **조합 가능한 모든 경우의 수** 구하기
    - from itertools import permutations 이용하기
- permutation의 결과는 다음과 같은 형태이므로 각 element를 숫자로 붙여줘야 한다.
    
    ```
    [('0', '1', '1'), ('1', '0', '1'), ('1', '1'), ('1',), ('1', '0'), ('0', '1'), ('0',), ('1', '1', '0')]
    → [0, 1, 101, 10, 11, 110]
    ```
    
- 조합 list의 각 element마다 소수 여부 판별하기
    - 소수 : 약수가 1과 자기 자신(n)뿐
        
        ```python
        def is_prime_number(n):
        	if n < 2:
        		return False
        	for i in range(2,n):
        		if n % i == 0:
        			return False
        	return True
        ```
        
<br/>

## 1차 시도

일단 풀 수 있을라나 해서 찔러봤는데... 알고리즘은 맞는 듯 하나 테스트 케이스 1도 안 맞는다...

```python
from itertools import permutations

def is_prime_number(n):
	for i in range(2,n):
		if n % i == 0:
			return False
	return True

def solution(numbers):
    answer = 0
    
    # 각 숫자 조각들 넣은 리스트
    num_list = [int(elem) for elem in numbers]
    
    temp_per_list = []
    per_list = []
    
    for i in range(1,len(num_list)+1):
        temp_per_list.append(list(permutations(num_list,i)))
        
    for elem in temp_per_list:
        for a in elem:
            per_list.append(a)
            
    temp_str = ""
    for elem in per_list:
        for i in elem:
            temp_str += str(i)
            per_num = int(temp_str)
    
    if is_prime_number(per_num):
        answer += 1

    return answer
```

<br/>

## 2차 시도

- 깔끔하게 정답!

```python
from itertools import permutations

def is_prime_number(n):
    if n < 2:
        return False

    for i in range(2, n):
        if n % i == 0:
            return False

    return True

def solution(numbers):
    # numbers 속 숫자 나누기
    numList = [num for num in numbers]

    wholePermu = list()
    # 조합 가능한 모든 경우의 수 구하기
    for a in range(1, len(numList) + 1):
        wholePermu.extend(list(permutations(numList, a)))

    # wholePermu = [('0',), ('1',), ('1',), ('0', '1'), ('0', '1'), ('1', '0'),
    #               ('1', '1'), ('1', '0'), ('1', '1'), ('0', '1', '1'), ('0', '1', '1'),
    #               ('1', '0', '1'), ('1', '1', '0'), ('1', '0', '1'), ('1', '1', '0')]

    # join 함수 : 리스트에 있는 요소들을 합쳐서 하나의 문자열로 바꾼다.
    for i in range(len(wholePermu)):
        wholePermu[i] = int(''.join([b for b in wholePermu[i]]))
    # wholePermu = [0,1,1,1,1,10,11,10,11,11,11,101,110,101,110]

    # 중복되는 조합을 제거하기 위해 중간에 잠시 set 타입으로 변경했다.
    wholePermu = list(set(wholePermu))
    # wholePermu = [0, 1, 101, 10, 11, 110]

    answer = list()

    # 조합 list의 각 element마다 소수 여부 판별하기
    for elem in wholePermu:
        if is_prime_number(elem):
            answer.append(elem)

    return len(answer)

# 실제 시작은 여기서부터
if __name__ == "__main__":
    numbers = "011"
    print(solution(numbers))
```