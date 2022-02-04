---
title: "[ 데이터구조 ]  Binary Search"
date: 2022-01-11
excerpt: "이진탐색에 대해 알아봅시다."
categories: 데이터구조
toc: true
toc_sticky: true
---


## 이진탐색 (이분탐색)

### 주어진 문제

- **서로 다른 n개**의 **정렬된** 수가 **list**에 저장되어 있을 때, **특정한 값**을 찾아라

### 필요한 변수

- **left** : 범위 중 가장 왼쪽 원소의 index
- **right** : 범위 중 가장 오른쪽 원소의 index
- **middle** : (left+right)/2 → **버림**해야 함.

### 알고리즘

```python
while 검사할 수가 남아있을 때까지:
	middle = (left + right) / 2

	if (찾는 수) < (middle이 가리키는 수):
		right = middle - 1

	elif (찾는 수) == (middle이 가리키는 수):
		return middle

	else:  # 찾는 수 > middle이 가리키는 수
		left = middle + 1
```

- 예시
    
    ![1.jpg](/assets/images/posts/data_structure/binary_search/1.jpg){:width=300}
    
- 시간복잡도 : $O(logN)$




## 파이썬 bisect 모듈

- 이진탐색을 구현한 모듈

### 모듈 내 대표적인 메소드

오름차순으로 **정렬된 리스트**에 **새로운 수를 삽입**하려 할 때...

- bisect_left(literable, value)
    - 어떠한 수의 **왼쪽**으로 삽입
    - 동일한 수가 이미 리스트에 존재한다면 → 기존 항목의 **왼쪽**
- bisect_right(literable, value)
    - 어떠한 수의 **오른쪽**으로 삽입
    - 동일한 수가 이미 리스트에 존재한다면 → 기존 항목의 **오른쪽**

두 메소드 모두 **새로 들어갈 수의 index를 반환**한다.

### 사용예시

```python
from bisect import bisect_left, bisect_right

nums = [0,1,2,3,4,5,6,7,8,9] 
n = 5 # 삽입할 수 

bisect_left(nums, n) # 좌측에 삽입한다. 결과: 5
bisect_right(nums, n) # 우측에 삽입한다. 결과: 6
```