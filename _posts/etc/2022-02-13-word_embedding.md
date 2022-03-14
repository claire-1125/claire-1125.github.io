---
title: "[ 이것저것 ]  Word Embedding"
date: 2022-02-13
excerpt: "word embedding에 대해 알아봅시다."
categories: 
    - Miscellaneous
toc: true
toc_sticky: true
---


## Word Vector

- 단어를 vector로 mapping한 것

### one-hot vector
- sparse representation : 해당 단어만 1이고 나머지 성분은 0
- 한계점
    - 다른 단어 간의 관계를 표현할 수 없다.
        - 동음이의어를 구분할 수 없다.
    - 계산 측면에서 비효율적이다.
        - 차원의 저주
        - 맥락상 의미를 파악할 수 없다.

<br/>

## Word Embedding Vector

- 단어를 dense(↔ sparse) vector로 표현하는 것
- embedding?
    - **discrete variable**을 **vector of continuous numbers**로 mapping시키는 것
- 예제)
    - 단어에 부여한 index : 2
    - one-hot vector : $\begin{bmatrix}
    0&0&1&0
    \end{bmatrix}$
    - weight matrix : $\begin{bmatrix}
    5.1&-0.3&0.1&3.2&1.3
    \\ -1&7.5&-2&3.4&0.5
    \\ 1.2&27&0.4&-1&2.8
    \\ 1&-5&2.1&0.2&11
    \end{bmatrix}$
    - embedding vector : one-hot vector·weight matrix = $\begin{bmatrix}
    1.2&27&0.4&-1&2.8
    \end{bmatrix}$