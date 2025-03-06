import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
from sklearn.metrics import classification_report, confusion_matrix
from Model import WDCNN
import pickle
import random
from IPython.display import display

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 다중 GPU 사용 시 필요
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ============================================================
# 🔹 파라미터 저장 경로 정의
# ============================================================
parameter_dir = r"C:\Users\ChoiSeongHyeon\Desktop\WinningT\Winning_Ticket_Project\CWRU\Parameters"
original_path = os.path.join(parameter_dir, "Best Model")
experiment_1_path = os.path.join(parameter_dir, "Experiment 1")
experiment_2_path = os.path.join(parameter_dir, "Experiment 2")
unstructured_path = os.path.join(experiment_1_path, "Unstructured")
structured_path = os.path.join(experiment_1_path, "Structured")

# Experiment 1
unstructured_path_100 = os.path.join(unstructured_path, "100 Epochs")
unstructured_path_1000 = os.path.join(unstructured_path, "1000 Epochs")
structured_path_100 = os.path.join(structured_path, "100 Epochs")
structured_path_1000 = os.path.join(structured_path, "1000 Epochs")

# Experiment 2
approach_1_path = os.path.join(experiment_2_path, "Approach 1")
approach_2_path = os.path.join(experiment_2_path, "Approach 2")

initialization_approach_1 = os.path.join(approach_1_path, "Initialization")
No_initialization_approach_1 = os.path.join(approach_1_path, "No Initialization")

initialization_approach_2 = os.path.join(approach_2_path, "Initialization")
No_initialization_approach_2 = os.path.join(approach_2_path, "No Initialization")


unstructured_experiment_1_100_results_path = os.path.join(unstructured_path_100, "unstructured_experiment_results.pkl")
unstructured_experiment_1_1000_results_path = os.path.join(unstructured_path_1000, "unstructured_experiment_results.pkl")
structured_experiment_1_100_results_path = os.path.join(structured_path_100, "structured_experiment_results.pkl")
structured_experiment_1_1000_results_path = os.path.join(structured_path_1000, "structured_experiment_results.pkl")


unstructured_results_path = unstructured_experiment_1_100_results_path
structured_results_path = structured_experiment_1_100_results_path

# 생성할 폴더 및 폴더 생성 함수
directories = [
    parameter_dir, original_path, experiment_1_path, experiment_2_path,
    unstructured_path, structured_path,
    unstructured_path_100, unstructured_path_1000,
    structured_path_100, structured_path_1000,
    approach_1_path, approach_2_path,
    initialization_approach_1, No_initialization_approach_1,
    initialization_approach_2, No_initialization_approach_2
]

def create_directories(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1. UOS 데이터셋 관련 함수들
# ============================================================
class CWRUDataset(Dataset):
    """
    CWRU 데이터셋 클래스  
    각 CSV 파일은 4,096 길이의 1차원 신호 데이터로 구성되며,  
    파일명에 "Train", "Validation", "Test" 문자열이 포함되어 데이터를 구분합니다.
    """
    def __init__(self, dataset_type):
        base_path = r"C:\Users\ChoiSeongHyeon\Desktop\WinningT\Winning_Ticket_Project\Dataset\CWRU"
        fault_types = {'N': 0, 'IR': 1, 'OR@06': 2, 'B': 3}

        self.data, self.labels = [], []

        for fault, label in fault_types.items():
            sample_path = os.path.join(base_path, fault, "Samples")
            if not os.path.exists(sample_path):
                print(f"Warning: Path does not exist - {sample_path}")
                continue

            for file in os.listdir(sample_path):
                if file.endswith('.csv') and dataset_type in file:
                    file_path = os.path.join(sample_path, file)
                    data = pd.read_csv(file_path, header=None).values.flatten()
                    # Data Sample Length
                    if len(data) == 4096:
                        self.data.append(data)
                        self.labels.append(label)

        # Convert to Pytorch Tensors
        self.data = torch.tensor(self.data, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

        # Validation & Test 데이터는 재현을 위해 한 번만만 섞음
        if dataset_type in ["Validation", "Test"]:
            np.random.seed(42)
            indices = np.random.permutation(len(self.data))
            self.data = self.data[indices]
            self.labels = self.labels[indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
# ============================================================
# 1.5. 기존 원본 모델 학습 및 평가 함수들
# ============================================================
def train_original_model_with_batch_sizes(train_dataset, val_dataset, batch_sizes=[256], num_epochs=500, learning_rate=0.01, patience=200):
    """Original Model을 다양한 배치 크기로 학습하고 best_overall_model.pth에 저장"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_overall_acc = 0.0
    best_batch_size = None
    best_model_path = os.path.join(original_path, "best_overall_model.pth")
    best_initial_weights_path = os.path.join(original_path, "best_initial_weights.pth")

    for batch_size in batch_sizes:
        print(f"\n🔹 Training Original Model with Batch Size: {batch_size}")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=lambda _: np.random.seed(42))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 모델의 초기값이 할당
        model = WDCNN().to(device)

        # 🔹 **현재 배치 크기의 초기 가중치를 저장**
        init_weights_path = os.path.join(original_path, f"initial_weights_bs{batch_size}.pth")
        torch.save(model.state_dict(), init_weights_path)
        print(f"✅ Initial weights saved for batch {batch_size}!")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)  

        best_val_acc = 0.0
        early_stop_counter = 0

        for epoch in range(num_epochs):
            model.train()
            train_loss, train_correct = 0.0, 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_correct += (outputs.argmax(dim=1) == labels).sum().item()

            train_acc = train_correct / len(train_loader.dataset)

            # 🔹 Validation
            model.eval()
            val_loss, val_correct = 0.0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_correct += (outputs.argmax(dim=1) == labels).sum().item()

            val_acc = val_correct / len(val_loader.dataset)
            scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}: Batch {batch_size}, LR: {current_lr:.6f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

            # Save best model for current batch size
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                early_stop_counter = 0
                torch.save(model.state_dict(), os.path.join(original_path, f"best_model_bs{batch_size}.pth"))
                print(f"✅ Best model saved (Batch {batch_size}) with val_acc {best_val_acc:.4f}")
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break

        # 🔹 Save best overall model and corresponding initial weights
        if best_val_acc > best_overall_acc:
            best_overall_acc = best_val_acc
            best_batch_size = batch_size
            torch.save(model.state_dict(), best_model_path)
            best_initial_weights_path = init_weights_path  # 🔹 최고 성능 모델의 초기 가중치 업데이트
            print(f"\n🏆 New Best Overall Model Saved! Val Acc: {best_overall_acc:.4f} (Batch {best_batch_size})")

    print(f"\n✅ Training completed! Best model: {best_model_path} from batch size {best_batch_size}")
    print(f"🎯 Best Initial Weights Saved: {best_initial_weights_path}")

# ============================================================
# 🔹 Classification Report 및 Confusion Matrix
# ============================================================
def evaluate_classification(model, test_loader, device, title="Model Evaluation"):
    """Test Dataset에서 Classification Report 및 Confusion Matrix 출력"""
    model.to(device)
    model.eval()
    
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    print(f"\n📌 {title} - Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['N', 'IR', 'OR@06', 'B'], digits = 6))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['N', 'IR', 'OR@06', 'B'], yticklabels=['N', 'IR', 'OR@06', 'B'])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{title} - Confusion Matrix")
    plt.show()

# ============================================================
# 🔹 Pruning 후 저장 함수
# ============================================================
def remove_pruning_and_save_unstructured(model, save_path):
    """Unstructured Pruning된 모델에서 Pruning Mask를 제거하고 저장"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
            prune.remove(module, "weight")  # ✅ Unstructured Pruning 제거
    torch.save(model.state_dict(), save_path)  # ✅ 가중치 저장 (Pruning Mask 제거됨)

def remove_pruning_and_save_structured(model, save_path):
    """Structured Pruning된 모델에서 Pruning Mask를 제거하고 저장"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d):  # ✅ Structured Pruning된 Conv1d Layer만 적용
            prune.remove(module, "weight")  # ✅ Pruning Mask 제거
    torch.save(model.state_dict(), save_path)  # ✅ Pruning Mask 없이 가중치만 저장


# ============================================================
# 🔹 성능평가 함수
# ============================================================
def measure_inference_time(model, test_loader, device, warmup_batches=5, repeat_times=5, print_batches=True):
    """
    모델의 Inference Time을 ms 단위로 측정합니다.
    처음 warmup_batches 만큼은 워밍업만 하고, 나머지 배치에 대해 측정합니다.
    repeat_times만큼 반복하여 평균을 반환합니다.
    """
    model.eval()
    is_cuda = device.type == 'cuda'
    total_batches = len(test_loader)

    # 최초 1회만 출력
    if print_batches:
        print(f"Total Batches in Test Loader: {total_batches}")

    eval_batches = max(0, total_batches - warmup_batches)

    if print_batches:
        print(f"Batches Used for Evaluation after Warm-up: {eval_batches}")

    inference_times = []

    for _ in range(repeat_times):
        total_time = 0.0
        num_runs = 0

        with torch.no_grad():
            for i, (inputs, _) in enumerate(test_loader):
                inputs = inputs.to(device)

                # Warm-up 단계
                if i < warmup_batches:
                    _ = model(inputs)
                    if is_cuda:
                        torch.cuda.synchronize()
                    continue

                # 이후 배치부터 측정
                if is_cuda:
                    torch.cuda.synchronize()
                start_time = time.perf_counter()

                _ = model(inputs)

                if is_cuda:
                    torch.cuda.synchronize()
                end_time = time.perf_counter()

                inference_time_ms = (end_time - start_time) * 1000.0

                total_time += inference_time_ms
                num_runs += 1

        avg_time = total_time / num_runs if num_runs > 0 else 0.0
        inference_times.append(avg_time)

    return sum(inference_times) / len(inference_times)

def evaluate_test_performance_unstructured_avg(original_model, test_loader, device, experiment_results, model_save_path, repeat_times=5):
    """
    Original 및 Unstructured Pruned Model의 Test Accuracy, Inference Time(ms 단위)을 평가 (N회 반복 평균 적용)
    """
    test_accuracies = []

    # ✅ Original Model 성능 평가
    original_model.to(device)
    original_model.eval()

    total_params = count_total_parameters(original_model)
    test_correct = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = original_model(inputs)
            test_correct += (outputs.argmax(dim=1) == labels).sum().item()

    original_test_acc = test_correct / len(test_loader.dataset)

    # Inference Time N번 반복 -> 평균
    original_inference_times = []
    for _ in range(repeat_times):
        inference_time = measure_inference_time(original_model, test_loader, device, warmup_batches=5)
        original_inference_times.append(inference_time)

    original_inference_time_avg = sum(original_inference_times) / repeat_times

    test_accuracies.append({
        "Pruning Ratio": 0.0,
        "Number of Non-Zero Params": total_params,
        "Inference Time (ms)": original_inference_time_avg,
        "Test Accuracy": original_test_acc
    })

    # 🔹 Unstructured Pruned Models 성능 평가
    for pruning_ratio in experiment_results.keys():
        model_path = os.path.join(model_save_path, f"best_unstructured_{int(pruning_ratio*100)}.pth")

        if not os.path.exists(model_path):
            print(f"❌ Model file not found for Pruning Ratio: {pruning_ratio}")
            continue

        # 🔹 Best Pruned Model 로드
        model = WDCNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        nonzero_params = count_nonzero_parameters(model)

        # 🔹 Test Accuracy 측정
        model.eval()
        test_correct = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                test_correct += (outputs.argmax(dim=1) == labels).sum().item()

        test_acc = test_correct / len(test_loader.dataset)

        # Inference Time N번 반복 -> 평균
        pruned_inference_times = []
        for _ in range(repeat_times):
            inference_time = measure_inference_time(model, test_loader, device, warmup_batches=5)
            pruned_inference_times.append(inference_time)

        inference_time_avg = sum(pruned_inference_times) / repeat_times

        # 🔹 결과 저장
        test_accuracies.append({
            "Pruning Ratio": pruning_ratio,
            "Number of Non-Zero Params": nonzero_params,
            "Inference Time (ms)": inference_time_avg,
            "Test Accuracy": test_acc
        })

    return pd.DataFrame(test_accuracies)

def evaluate_test_performance_structured_avg(original_model, test_loader, device, experiment_results, model_save_path, repeat_times=5):
    """
    Original 및 Structured Pruned Model의 Test Accuracy, Inference Time(ms 단위)을 평가 (N회 반복 평균 적용)
    """
    test_accuracies = []

    # ✅ Original Model 성능 평가
    original_model.to(device)
    original_model.eval()

    total_params = count_total_parameters(original_model)
    test_correct = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = original_model(inputs)
            test_correct += (outputs.argmax(dim=1) == labels).sum().item()

    original_test_acc = test_correct / len(test_loader.dataset)

    # Inference Time N번 반복 -> 평균
    original_inference_times = []
    for _ in range(repeat_times):
        inference_time = measure_inference_time(original_model, test_loader, device, warmup_batches=5)
        original_inference_times.append(inference_time)

    original_inference_time_avg = sum(original_inference_times) / repeat_times

    test_accuracies.append({
        "Pruning Ratio": 0.0,
        "Number of Non-Zero Params": total_params,
        "Inference Time (ms)": original_inference_time_avg,
        "Test Accuracy": original_test_acc
    })

    # 🔹 Structured Pruned Models 성능 평가
    for pruning_ratio in experiment_results.keys():
        model_path = os.path.join(model_save_path, f"best_structured_finetuned_{int(pruning_ratio * 100)}.pth")

        if not os.path.exists(model_path):
            print(f"❌ Model file not found for Pruning Ratio: {pruning_ratio}")
            continue

        # 🔹 Best Pruned Model 로드
        model = WDCNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        nonzero_params = count_nonzero_parameters(model)

        # 🔹 Test Accuracy 측정
        model.eval()
        test_correct = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                test_correct += (outputs.argmax(dim=1) == labels).sum().item()

        test_acc = test_correct / len(test_loader.dataset)

        # Inference Time N번 반복 -> 평균
        pruned_inference_times = []
        for _ in range(repeat_times):
            inference_time = measure_inference_time(model, test_loader, device, warmup_batches=5)
            pruned_inference_times.append(inference_time)

        inference_time_avg = sum(pruned_inference_times) / repeat_times

        # 🔹 결과 저장
        test_accuracies.append({
            "Pruning Ratio": pruning_ratio,
            "Number of Non-Zero Params": nonzero_params,
            "Inference Time (ms)": inference_time_avg,
            "Test Accuracy": test_acc
        })

    return pd.DataFrame(test_accuracies)

# ============================================================
# 🔹 Unstructured Pruning Fine-Tuning 수행 및 저장 (훈련 1번만 수행)
# ============================================================
def apply_sparse_training(model, amount):
    """Unstructured Pruning을 모델에 적용"""
    model = deepcopy(model)
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
    return model

def fine_tune_unstructured(pruning_amounts, train_loader, val_loader, num_epochs=100, save_results=True):
    """Unstructured Pruning Fine-Tuning 수행 후 결과를 저장"""

    # ✅ 기존 학습 결과가 있다면 불러오기 (불필요한 학습 방지)
    if os.path.exists(unstructured_results_path):
        with open(unstructured_results_path, "rb") as f:
            results = pickle.load(f)
        print("\n✅ 기존 Fine-Tuning 결과를 불러왔습니다. 학습을 건너뜁니다.")
        return results
    
    # ✅ 새롭게 Fine-Tuning 실행
    results = {}

    for pruning_amount in pruning_amounts:
        print(f"\n🔹 Fine-tuning with Unstructured Pruning Amount: {pruning_amount}")

        model = WDCNN().to(device)
        model.load_state_dict(torch.load(original_path + "/best_overall_model.pth", map_location=device))
        pruned_model = apply_sparse_training(model, amount=pruning_amount)
        pruned_model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(pruned_model.parameters(), lr=0.00001)

        best_val_acc = 0.0
        best_pruned_model = None
        val_accuracies = []

        for epoch in range(num_epochs):
            pruned_model.train()
            train_correct = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = pruned_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_correct += (outputs.argmax(dim=1) == labels).sum().item()

            train_acc = train_correct / len(train_loader.dataset)

            # Validation
            pruned_model.eval()
            val_correct = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = pruned_model(inputs)
                    val_correct += (outputs.argmax(dim=1) == labels).sum().item()

            val_acc = val_correct / len(val_loader.dataset)
            val_accuracies.append(val_acc)

            # 실시간 학습 진행 출력
            print(f"Epoch {epoch+1}/{num_epochs} - Pruning {pruning_amount:.1f}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_pruned_model = deepcopy(pruned_model)

        # ✅ 최적 모델 저장
        model_save_path = os.path.join(unstructured_path, f"best_unstructured_{int(pruning_amount*100)}.pth")
        remove_pruning_and_save_unstructured(best_pruned_model, model_save_path)

        results[pruning_amount] = {
            "Best Validation Accuracy": best_val_acc,
            "Validation Accuracy per Epoch": val_accuracies
        }

    # ✅ Fine-Tuning 결과 저장
    if save_results:
        with open(unstructured_experiment_1_100_results_path, "wb") as f:
            pickle.dump(results, f)
        print("\n✅ Fine-Tuning Results Saved!")

    return results

def load_unstructured_results():
    """저장된 Unstructured Pruning 결과 불러오기"""
    if os.path.exists(unstructured_results_path):
        with open(unstructured_results_path, "rb") as f:
            results = pickle.load(f)
        print("\n✅ Loaded Saved Unstructured Pruning Results!")
        return results
    else:
        print("\n❌ No Saved Results Found. Need to Fine-Tune First!")
        return None

def plot_unstructured_pruning_results(experiment_results, num_epochs, val_loader, include_original=True):
    """Pruning된 모델들의 Validation Accuracy 변화를 그래프로 출력 (Original Model 포함 가능)"""
    plt.figure(figsize=(10, 6))

    # ✅ Original Model Validation Accuracy 추가
    if include_original:
        original_model = WDCNN().to(device)
        original_model.load_state_dict(torch.load(original_path + "/best_overall_model.pth", map_location=device))
        original_model.eval()
        
        val_accuracies_original = []
        with torch.no_grad():
            for epoch in range(1, num_epochs + 1):
                val_correct = 0
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = original_model(inputs)
                    val_correct += (outputs.argmax(dim=1) == labels).sum().item()

                val_acc = val_correct / len(val_loader.dataset)
                val_accuracies_original.append(val_acc)

        # ✅ Original Model Plot (Pruning 0%)
        plt.plot(range(1, num_epochs+1), val_accuracies_original, label="Original Model (Pruning Ratio 0%)", linestyle="--", color="black")

    # ✅ Pruned Models Plot
    for pruning_ratio, result in experiment_results.items():
        val_accuracies = result["Validation Accuracy per Epoch"]
        if len(val_accuracies) > 0:
            plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label=f"Pruning Ratio {pruning_ratio*100:.0f}%")

    plt.xlabel("Training Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy of WDCNN \nwith Different Unstructured Pruning Ratios")
    plt.legend()
    plt.grid(True)
    plt.show()

def evaluate_test_performance_unstructured(original_model, test_loader, device, experiment_results, model_save_path):
    """
    Original 및 Unstructured Pruned Model의 Test Accuracy, Inference Time(ms 단위) 평가 (테이블 출력)
    """
    test_accuracies = []

    # ✅ Original Model 성능 평가
    original_model.to(device)
    original_model.eval()

    total_params = count_total_parameters(original_model)
    test_correct = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = original_model(inputs)
            test_correct += (outputs.argmax(dim=1) == labels).sum().item()

    original_test_acc = test_correct / len(test_loader.dataset)
    original_inference_time = measure_inference_time(original_model, test_loader, device, warmup_batches=5)

    test_accuracies.append({
        "Pruning Ratio": 0.0,
        "Number of Non-Zero Params": total_params,
        "Inference Time (ms)": original_inference_time,
        "Test Accuracy": original_test_acc
    })

    # 🔹 Unstructured Pruned Models 성능 평가
    for pruning_ratio in experiment_results.keys():
        model_path = os.path.join(model_save_path, f"best_unstructured_{int(pruning_ratio*100)}.pth")  # ✅ Unstructured Pruning 모델 불러오기

        if not os.path.exists(model_path):
            print(f"❌ Model file not found for Pruning Ratio: {pruning_ratio}")
            continue

        # 🔹 Best Pruned Model 로드
        model = WDCNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        nonzero_params = count_nonzero_parameters(model)

        # 🔹 Test Accuracy 측정
        model.eval()
        test_correct = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                test_correct += (outputs.argmax(dim=1) == labels).sum().item()

        test_acc = test_correct / len(test_loader.dataset)

        # 🔹 Inference Time 측정 (ms, Warm-up 적용)
        inference_time = measure_inference_time(model, test_loader, device, warmup_batches=5)

        # 🔹 결과 저장
        test_accuracies.append({
            "Pruning Ratio": pruning_ratio,
            "Number of Non-Zero Params": nonzero_params,
            "Inference Time (ms)": inference_time,
            "Test Accuracy": test_acc
        })

    return pd.DataFrame(test_accuracies)  # ✅ 최종 결과 테이블 반환


# ============================================================
# 🔹 Structured Pruning Fine-Tuning 수행 및 저장 (훈련 1번만 수행)
# ============================================================
def apply_structured_pruning(model, amount):
    model = deepcopy(model)  # ✅ 모델 복사 후 Pruning 적용
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d):  # ✅ Linear Layer 제외
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
    return model  # ✅ Pruned된 모델 반환

def calculate_actual_sparsity(model):
    """실제 Structured Pruning Ratio 계산 (비-제로 가중치 비율)"""
    total_weights = 0
    zero_weights = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d):
            total_weights += module.weight.nelement()
            zero_weights += (module.weight == 0).sum().item()
    return zero_weights / total_weights if total_weights > 0 else 0.0

def fine_tune_structured(pruning_amounts, train_loader, val_loader, num_epochs=100, save_results=True):
    """Structured Pruning Fine-Tuning을 수행하고 결과를 저장"""
    
    # ✅ 기존 학습 결과가 있다면 불러오기 (불필요한 학습 방지)
    if os.path.exists(structured_results_path):
        with open(structured_results_path, "rb") as f:
            results = pickle.load(f)
        print("\n✅ 기존 Structured Fine-Tuning 결과를 불러왔습니다. 학습을 건너뜁니다.")
        return results

    # ✅ 새롭게 Fine-Tuning 실행
    results = {}

    for pruning_amount in pruning_amounts:
        print(f"\n🔹 Fine-tuning with Structured Pruning Amount: {pruning_amount}")

        model = WDCNN().to(device)
        model.load_state_dict(torch.load(original_path + "/best_overall_model.pth", map_location=device))
        pruned_model = apply_structured_pruning(model, amount=pruning_amount)  # ✅ Pruned 모델 반환
        pruned_model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(pruned_model.parameters(), lr=0.00001)

        best_val_acc = 0.0
        best_pruned_model = None
        val_accuracies = []

        for epoch in range(num_epochs):
            pruned_model.train()
            train_correct = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = pruned_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_correct += (outputs.argmax(dim=1) == labels).sum().item()

            train_acc = train_correct / len(train_loader.dataset)

            # Validation
            pruned_model.eval()
            val_correct = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = pruned_model(inputs)
                    val_correct += (outputs.argmax(dim=1) == labels).sum().item()

            val_acc = val_correct / len(val_loader.dataset)
            val_accuracies.append(val_acc)

            print(f"Epoch {epoch+1}/{num_epochs} - Pruning {pruning_amount:.1f}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_pruned_model = deepcopy(pruned_model)

        # ✅ 최적 모델 저장 (Pruning Mask 제거 후 저장)
        model_save_path = os.path.join(structured_path, f"best_structured_finetuned_{int(pruning_amount*100)}.pth")
        remove_pruning_and_save_structured(best_pruned_model, model_save_path)

        results[pruning_amount] = {
            "Pruning Ratio": pruning_amount,
            "Best Validation Accuracy": best_val_acc,
            "Validation Accuracy per Epoch": val_accuracies
        }

    # ✅ Fine-Tuning 결과 저장
    if save_results:
        with open(structured_results_path, "wb") as f:
            pickle.dump(results, f)
        print("\n✅ Structured Fine-Tuning Results Saved!")

    return results

def plot_structured_pruning_results(experiment_results, num_epochs, val_loader, include_original=False):
    """Structured Pruning된 모델들의 Validation Accuracy 변화를 그래프로 출력"""

    plt.figure(figsize=(10, 6))

    # ✅ Original Model Plot (Pruning 0%) 추가 (필요한 경우)
    if include_original:
        print("\n🔹 Calculating Original Model Validation Accuracy for Graph...")
        original_model = WDCNN().to(device)
        original_model.load_state_dict(torch.load(original_path + "/best_overall_model.pth", map_location=device))
        original_model.eval()

        val_accuracies_original = []
        with torch.no_grad():
            for epoch in range(1, num_epochs + 1):  
                val_correct = 0
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = original_model(inputs)
                    val_correct += (outputs.argmax(dim=1) == labels).sum().item()

                val_acc = val_correct / len(val_loader.dataset)
                val_accuracies_original.append(val_acc)

        plt.plot(range(1, num_epochs + 1), val_accuracies_original, 
                 label="Original Model (Pruning Ratio 0%)", linestyle="--", color="black")

    # ✅ Structured Pruned Models Plot
    for pruning_ratio, result in experiment_results.items():
        val_accuracies = result["Validation Accuracy per Epoch"]
        if len(val_accuracies) > 0:
            plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, 
                     label=f"Structured Pruning {pruning_ratio*100:.0f}%")

    plt.xlabel("Training Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy of WDCNN \nwith Different Structured Pruning Ratios")
    plt.legend()
    plt.grid(True)
    plt.show()


def load_structured_results():
    """저장된 Structured Pruning 결과 불러오기"""
    if os.path.exists(structured_results_path):
        with open(structured_results_path, "rb") as f:
            results = pickle.load(f)
        print("\n✅ Loaded Saved Structured Pruning Results!")
        return results
    else:
        print("\n❌ No Saved Results Found. Need to Fine-Tune First!")
        return None

def evaluate_test_performance_structured(original_model, test_loader, device, experiment_results, model_save_path):
    """
    Original 및 Structured Pruned Model의 Test Accuracy, Inference Time(ms 단위)을 평가 (테이블 출력)
    """
    test_accuracies = []

    # ✅ Original Model 성능 평가
    original_model.to(device)
    original_model.eval()

    total_params = count_total_parameters(original_model)
    test_correct = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = original_model(inputs)
            test_correct += (outputs.argmax(dim=1) == labels).sum().item()

    original_test_acc = test_correct / len(test_loader.dataset)
    original_inference_time = measure_inference_time(original_model, test_loader, device, warmup_batches=5)

    test_accuracies.append({
        "Pruning Ratio": 0.0,
        "Number of Non-Zero Params": total_params,
        "Inference Time (ms)": original_inference_time,
        "Test Accuracy": original_test_acc
    })

    # 🔹 Structured Pruned Models 성능 평가
    for pruning_ratio in experiment_results.keys():
        model_path = os.path.join(model_save_path, f"best_structured_finetuned_{int(pruning_ratio * 100)}.pth")  # ✅ Structured Pruning 모델 불러오기

        if not os.path.exists(model_path):
            print(f"❌ Model file not found for Pruning Ratio: {pruning_ratio}")
            continue

        # 🔹 Best Pruned Model 로드
        model = WDCNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        nonzero_params = count_nonzero_parameters(model)

        # 🔹 Test Accuracy 측정
        model.eval()
        test_correct = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                test_correct += (outputs.argmax(dim=1) == labels).sum().item()

        test_acc = test_correct / len(test_loader.dataset)

        # 🔹 Inference Time 측정 (ms, Warm-up 적용)
        inference_time = measure_inference_time(model, test_loader, device, warmup_batches=5)

        # 🔹 결과 저장
        test_accuracies.append({
            "Pruning Ratio": pruning_ratio,
            "Number of Non-Zero Params": nonzero_params,
            "Inference Time (ms)": inference_time,
            "Test Accuracy": test_acc
        })

    return pd.DataFrame(test_accuracies)


# 🔹 전체 파라미터 수 계산 함수
def count_total_parameters(model):
    """모델의 전체 파라미터 개수 계산"""
    return sum(p.numel() for p in model.parameters())

# 🔹 비-제로 가중치 개수 계산 함수
def count_nonzero_parameters(model):
    """Pruning된 모델의 비-제로 가중치 개수 계산"""
    return sum((p != 0).sum().item() for p in model.parameters() if p.requires_grad)