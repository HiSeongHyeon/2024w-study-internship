# 2024_Winter_공학연구인턴십
목적 : [1] J. Frankle and M. Carbin, “The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks,” Mar. 04, 2019, arXiv: arXiv:1803.03635. doi: 10.48550/arXiv.1803.03635.

위 논문을 바탕으로, 기계 결함 진단 데이터셋인 UOS, CWRU 데이터셋에서 WDCNN 기반의 Bearing Machine Fault Diagnosis Model에서도 Winning Ticket이 존재하는지 Pruning 기법을 활용하여 확인하고자 함.

본 프로젝트의 실험 구성은 '4. 실험 구성' 참조


## 1. 프로젝트 개요
* **기간**: 2025.01 - 2025.02
* **참여자**: 최성현
* **주요 기술**: WDCNN, CUDA, MODEL COMPRESSION(PRUNING)
  
---

## 2. 개발 환경 

* **OS**: Window 11
* **Language**: Python 3.13.1
* **Dependencies**:
    * conda env create -f environment.yml
* **Hardware**: NVIDIA RTX 4060

---

## 3. 폴더 구조 
```text
├── CWRU                   # CWRU 데이터셋에 대한 실험 진행
├── Dataset                # 데이터셋 (CWRU, UOS)
├── UOS                    # UOS 데이터셋에 대한 실험 진행
├── environment.yml        
└── README.md
```
---

## 4. 실험 구성

### 실험 1.
- 대상이 다른 Pruning 방식(Unstructured, Structured)의 차이가 Model의 Accuracy와 Inference Time에 미치는 영향 조사
##### Dataset
 - CWRU Dataset
 - UOS Dataset
##### Model
 - 1D-CNN (WDCNN)
##### 실험 절차
1. 성능을 최대한 높인 Best Model 설계
2. Model에 One-Shot Pruning 적용 후 Fine-Tuning
    - Pruning Ratio : [0%, 20%, 40%, 60%, 80%]
    2.1 Unstructured Pruning
    2.2 Structured Pruning
### 실험 2.
- Iterative Pruning Strategy, 초기 Weight 값(실험 1의 Best Model's Weights, Random Weights)을 다르게 하여 Winning Ticket 발견하기
    - Winning Ticket 원본 논문 : [1] J. Frankle and M. Carbin, “The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks,” Mar. 04, 2019, arXiv: arXiv:1803.03635. doi: 10.48550/arXiv.1803.03635.
##### Dataset
 - CWRU Dataset
 - UOS Dataset
##### Model
 - 1D-CNN (WDCNN)
##### Background Knowledge
 - Iterative Pruning
    - Iterative Pruning Ratio : 20%
 - Iterative Pruning Strategy
    - Strategy 1
        - Step 1) Randomly initialized neural network (W~0~)
        - Step 2) Train that model
        - Step 3) prune s% of the parameters ('s' is a pruning ratio)
        - Step 4) Initialization (W~0~)
        - Step 5) Repeat 'Step 2 ~ 4'
    - Strategy 2
        - Step 1) Randomly initialized neural network (W~0~)
        - Step 2) Train that model
        - Step 3) prune s% of the parameters ('s' is a pruning ratio)
        - Step 4) repeat 'step 2 ~ 3'
##### 실험 절차
- Approach 1. 실험 1에서 도출된 Best Model's Weight 값으로 WDCNN 모델 불러온 다음
    - 1.1 Strategy 1 적용 (Initialization)
    - 1.2 Strategy 2 적용 (No Initialization)
- Approach 2. Weight가 Randomly Initialized된 WDCNN 모델 불러온 다음
    - 2.1 Strategy 1 적용 (Initialization)
    - 2.2 Strategy 2 적용 (No Initialization)
- - -



