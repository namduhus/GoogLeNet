## Going deeper with convolutions
Author:
<p>Christian Szegedy [1], Wei Liu[2], Yangqing Jia[1], Pierre Sermanet[1], Scott Reed[3], Dragomir Anguelov[1], Dumitru Erhan[1], Vincent Vanhoucke[1], Andrew Rabinovich[1]</p>
<p>1.Google Inc</p>
<p>2.University of North Carolina, Chapel Hill</p>
<p>3.University of Michigan</p>

## Abstract
![image](https://github.com/user-attachments/assets/68c798e7-26ed-448b-b89c-098cad3fabec)
<p>이 논문은 Code Name "Inception"이라는 deep CNN 아키텍처를 제안한다.</p>
이 아키텍처에서 제일 중요한 달성 사항은 네트워크 내의 계산해야할 리소스들을 더 나은 방법으로 사용했다는 점이다. 또한 이 논문에서 언급되는 GoogLeNet으로 ILSVRC14에서 1등을 했다.

## Introduction
2012 ~ 2015 CNN 분야는 급속도로 발전해 왔다. 이러한 발전은 하드웨어 발전이나 더 많은 데이터의 집합 덕분이기보다는 새로운 아이디어, 알고리즘, 발전한 네트워크 아키텍처 덕분이라고 한다.
GoogLeNet에서 주목할 만한 점은 본 논문에서는 전력과 메모리 사용을 효율적으로 설계하여 모바일이나 임베디드 환경에 적용시킬 수 있게 한 것이다.

이 논문에서 deep은 두 가지 의미를 가진다고 한다.
첫번째는 Inception 모듈이라는 새로운 형태의 구성을 제안하고, 두번째는 매우 깊은 network를 의미한다.

## Motivation and High Level Considerations
DNN의 성능을 높이는 가장 단순하면서 정확한 방법은 depth, the number of levels, width, the number of units를 늘리는 것이라고한다. 그러나 이러한 접근은 두 결점이 존재한다고 한다.
첫번째는 사이즈를 키운다는것은 Overfitting을 야기하고 두번째는 데이터의 수를 늘리면 해결될 문제이지만 비용측면에서 다소 무리가 있다.
두번째는 균등하게 증가된 network는 컴퓨팅 자원을 더 많이 잡아먹는다.

두 가지 문제를 해결하는 근본적인 방법은 fully connected에서 sparsely connected 구조로 변경하는 것이다.

![image](https://github.com/user-attachments/assets/fdecf836-e91e-405b-9657-5e4997064650)

## Architectural Details
<p>Inception의 주요 아이디어는 convolutional vision network에서 최적의 local sparse 구조를 어떻게 하면 현재 사용 가능한 dense component로 구성할지에서 기반을 했고</p>
<p>본 논문의 Inception에서는 편의를 위해 filter size를 1x1, 3x3, 5x5로 제한했으며 Pooling이 CNN의 성공에 있어 필수 요소기 때문에,</p>
<p>1x1, 3x3, 5x5 conv layer에 이어 pooling도 추가했다고 한다.3✕3, 5✕5 비율의 convolution이 높은 계층으로 갈 수록 증가하는데 5✕5 convolution의 수가 많지 않더라도</p> 
<p>이 convolution은 수많은 필터를 갖는 convolution 위에 추가하기엔 상당히 연상량이 높아서 안좋다.</p>
<p>또한 Pooling 계층의 출력을 convolution 계층의 출력과 합치는 것은 단계별로 출력의 개수를 늘릴 수 밖에 없기에 이 아키텍처는 최적의 sparse 구조에 매우 비효율적이라고 한다.</p>

![image](https://github.com/user-attachments/assets/b83817c0-d4ed-4f12-874e-40b94ac55fe6)
![image](https://github.com/user-attachments/assets/017eb968-e50b-4491-8755-16f2917a7b9e)

### Inception Module
![image](https://github.com/user-attachments/assets/4a9b080c-dc90-4f5c-b50d-0a8fb19960f5)
![image](https://github.com/user-attachments/assets/e8498dc4-f0e8-4610-8d83-1400cbf51647)

<p>새로운 버전의 Inception 모듈은 1x1 conv를 제외한 각각의 conv layer 앞에, pooling layer에는 뒤에 1x1 conv를 적용시켜 dimension reduction 효과를 보게 했다.</p>
<p>메모리 효율 문제로 모델의 초반엔 기존의 CNN의 방식을 따르고 이후엔 Inception 모듈을 쌓아 사용했고,</p> 
<p>그 결과 3x3과 5x5와 같이 큰 patch에서의 conv 연산을 진행기 전에 연산량 감소가 이루어졌고, 여러크기의 patch에서 얻은 출력으로 인해 여러 스케일에서 동시의 특징을 추출할 수 있게 되었다.</p>

## Average Pooling
<p>본 논문에서는 추가적으로 학습 파라미터 수도 줄이고 Overfitting을 방지할 수 있는 방법을 제안한다.</p>
<p>네트워크 마지막 부분에 Fully Connected 연결 대신 Average Pooling을 사용하는 방법이다.</p>

![image](https://github.com/user-attachments/assets/f498a126-4795-4fc8-a030-f37e0ad6d9a1)

![image](https://github.com/user-attachments/assets/87124ae8-c5b8-4e62-b9af-3ab09fa71a5a)

## GoogLeNet
![image](https://github.com/user-attachments/assets/dd3e00a2-b191-49d2-84f5-0bbecbc38cc3)


