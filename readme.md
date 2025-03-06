# 2024_Winter_공학연구인턴십
- - - 
## 실험 1.
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
- - - 
## 실험 2.
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