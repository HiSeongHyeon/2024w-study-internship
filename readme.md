# 2024_Winter_공학연구인턴십
목적 : 

## 1. 프로젝트 개요
* **기간**: 2025.01 - 2025.02
* **참여자**: 최성현
* **주요 기술**: WDCNN, CUDA, MODEL COMPRESSION(PRUNING)
  
---

## 2. 개발 환경 

* **OS**: Window 11
* **Language**: Python ??
* **Dependencies**:
    * (예: `pip install -r requirements.txt` 또는 `conda env create -f environment.yaml`)
* **Hardware**: NVIDIA RTX 4060

---

## 3. 폴더 구조 
```text
├── src/            # 소스 코드
├── data/           # 데이터셋 
├── docs/           # 관련 문서
├── weights/        # 학습된 모델 가중치 
└── README.md
```

## 4. 실행 방법 (예시, Optional)
1. 데이터 전처리: ```python data/preprocess.py```
2. 모델 학습: ```python src/main.py --config config.yaml```
3. 결과 시각화: ```python src/visualize.py```

---

## 5. 브랜치 구조 (예시, Optional)

* **main**: 메인 브랜치

---
## 6. 실험 구성



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
    - Winning Ticket 원본 논문 : 
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

