import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold
from tabulate import tabulate
from typing import Dict, Any, List, Optional

# Custom imports
from audio_processing import prepare_training_data
from gmm_manager import train_gmms, classify_sample
from decorators import safe_execution


def calculate_accuracy(gmm_models: Dict, test_data: Dict[str, np.ndarray], n_samples: int = 5):
    """Calculates classification accuracy on a test set using random sampling."""
    correct = 0
    tested = 0
    predictions = []
    true_labels = []

    for true_label, mfcc_matrix in test_data.items():
        # Validate data
        if mfcc_matrix is None or not isinstance(mfcc_matrix, np.ndarray) or mfcc_matrix.shape[0] < n_samples:
            continue

        # Randomly select 'n_samples' frames to test
        # Note: In real scenarios, we usually test whole files, but here we follow original logic
        # of testing random frames/segments.
        total_frames = mfcc_matrix.shape[0]
        # Ensure we don't sample more than available
        n_to_sample = min(n_samples, total_frames)
        
        indices = np.random.choice(total_frames, n_to_sample, replace=False)

        for idx in indices:
            # Slice to keep 2D shape (1, n_features)
            sample_mfcc = mfcc_matrix[idx:idx+1, :]
            
            predicted, _ = classify_sample(gmm_models, sample_mfcc, show_table=False)

            predictions.append(predicted)
            true_labels.append(true_label)

            if str(predicted) == str(true_label):
                correct += 1
            tested += 1
        
    accuracy = (correct / tested * 100) if tested > 0 else 0.0
    return accuracy, predictions, true_labels


def perform_cross_validation(raw_dataset: Dict, k_folds: int = 5, n_samples: int = 3):
    """
    Performs K-Fold Cross Validation.
    
    Args:
        raw_dataset: The dictionary returned by load_train_files_and_determine_mfcc
                     Format: {speaker_id: [{MFCC, num, filename}, ...]}
    """
    print(f"\nRozpoczynam {k_folds}-krotną walidację krzyżową...")
    
    speakers = list(raw_dataset.keys())
    
    if len(speakers) < k_folds:
        print(f"Zbyt mało mówców ({len(speakers)}) na {k_folds} foldów.")
        return None

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_accuracies = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(speakers), 1):
        print(f"\n--- Fold {fold}/{k_folds} ---")
        
        # 1. Split Speakers
        train_speakers = [speakers[i] for i in train_idx]
        test_speakers = [speakers[i] for i in test_idx]
        
        # 2. Filter Dataset
        train_raw = {k: raw_dataset[k] for k in train_speakers}
        test_raw = {k: raw_dataset[k] for k in test_speakers}
        
        # 3. Prepare Data (Concatenate MFCCs)
        # Note: We suppress the table print in loop to avoid clutter
        train_ready = prepare_training_data(train_raw, show_table=False)
        test_ready = prepare_training_data(test_raw, show_table=False)
        
        # 4. Train
        models = train_gmms(train_ready)
        
        # 5. Test
        acc, _, _ = calculate_accuracy(models, test_ready, n_samples=n_samples)
        fold_accuracies.append(acc)
        print(f"  -> Skuteczność fold {fold}: {acc:.2f}%")
    
    # Statistics
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    
    print(f"\n=== WYNIKI WALIDACJI KRZYŻOWEJ ===")
    print(f"Średnia skuteczność:     {mean_acc:.2f}%")
    print(f"Odchylenie standardowe:  {std_acc:.2f}%")
    print(f"Wyniki per fold:         {[f'{acc:.2f}%' for acc in fold_accuracies]}")
    
    return {
        'fold_accuracies': fold_accuracies,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc
    }


@safe_execution("Błąd podczas ewaluacji systemu")
def evaluate_system(gmm_models: Dict, test_dataset: Dict[str, np.ndarray], n_samples: int = 5) -> Optional[Dict[str, Any]]:
    """
    Performs a comprehensive evaluation of the system with detailed metrics.
    """
    print("Przeprowadzam szczegółową ewaluację systemu...")
    
    all_predictions = []
    all_true_labels = []
    
    for true_label, mfcc_matrix in test_dataset.items():
        if mfcc_matrix is None or not isinstance(mfcc_matrix, np.ndarray) or mfcc_matrix.shape[0] == 0:
            continue
            
        # Sample random frames
        n_test = min(n_samples, mfcc_matrix.shape[0])
        idx = np.random.choice(mfcc_matrix.shape[0], n_test, replace=False)
        
        for i in idx:
            sample_mfcc = mfcc_matrix[i:i+1, :]
            predicted, _ = classify_sample(gmm_models, sample_mfcc, show_table=False)
            
            all_true_labels.append(str(true_label))
            all_predictions.append(str(predicted))
    
    if not all_true_labels:
        print("Brak danych do ewaluacji!")
        return None
    
    # Calculate Metrics
    metrics = {
        'overall_accuracy': accuracy_score(all_true_labels, all_predictions) * 100,
        'precision': precision_score(all_true_labels, all_predictions, average='weighted', zero_division=0),
        'recall': recall_score(all_true_labels, all_predictions, average='weighted', zero_division=0),
        'f1_score': f1_score(all_true_labels, all_predictions, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(all_true_labels, all_predictions),
        'report': classification_report(all_true_labels, all_predictions, output_dict=True, zero_division=0),
        'total_tested': len(all_true_labels),
        'classes': sorted(list(set(all_true_labels)))
    }

    return metrics


def display_confusion_matrix(cm: np.ndarray, labels: List[str]):
    """Visualizes the confusion matrix using Tabulate."""
    headers = [f"Pred {label}" for label in labels]
    rows = []
    
    for i, true_label in enumerate(labels):
        # Handle cases where confusion matrix might be smaller if some classes weren't predicted
        if i < len(cm): 
            row_data = [str(cm[i][j]) for j in range(len(cm[i]))]
            rows.append([f"True {true_label}"] + row_data)
    
    print("\n=== MACIERZ POMYŁEK ===")
    print(tabulate(rows, headers=headers, tablefmt="grid"))


def print_evaluation_report(metrics: Dict):
    """Prints a user-friendly summary of the evaluation."""
    if not metrics:
        return

    print(f"\n=== WYNIKI EWALUACJI (n={metrics['total_tested']}) ===")
    print(f"Dokładność (Accuracy):  {metrics['overall_accuracy']:.2f}%")
    print(f"Precyzja (Precision):   {metrics['precision']:.4f}")
    print(f"Czułość (Recall):       {metrics['recall']:.4f}")
    print(f"F1 Score:               {metrics['f1_score']:.4f}")

    display_confusion_matrix(metrics['confusion_matrix'], metrics['classes'])
