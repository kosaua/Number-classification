import sys
from typing import Tuple, Dict, Any

# Custom imports
from config import (
    PROCESSED_DATA_FILE, DEFAULT_MODEL_FILENAME, FINAL_MODEL_FILENAME,
    BEST_MFCC_PARAMS_FILE, BEST_GMM_PARAMS_FILE
)
from utils import load_pickle, save_pickle, load_params_from_csv
from audio_processing import (
    load_mfcc_params, load_gmm_params,
    load_train_files_and_determine_mfcc, 
    prepare_training_data, 
    split_train_test_by_speaker,
    validate_data_quality
)
from gmm_manager import train_gmms, save_models, load_models
from evaluation import calculate_accuracy, perform_cross_validation, evaluate_system, print_evaluation_report
from optimization import optimize_parameters_full

# --- Helper to load params ---
def _load_best_or_default_params() -> Tuple[Dict, Dict]:
    """Loads best parameters if available, otherwise defaults."""
    mfcc = load_params_from_csv(BEST_MFCC_PARAMS_FILE)
    gmm = load_params_from_csv(BEST_GMM_PARAMS_FILE)
    
    if mfcc and gmm:
        print(f"-> Wykryto zoptymalizowane parametry (MFCC n={mfcc.get('n_mfcc')}, GMM k={gmm.get('num_components')})")
        return mfcc, gmm
    
    print("-> Brak zapisanych najlepszych parametrów. Używam domyślnych.")
    return load_mfcc_params(), load_gmm_params()

# --- Stages ---

def prepare_data_stage():
    """Stage 1: Initial data preparation."""
    print("\n=== ETAP 1: PRZYGOTOWANIE DANYCH ===")
    mfcc_params = load_mfcc_params()
    
    full_dataset = load_train_files_and_determine_mfcc(mfcc_params, show_table=True)
    if not full_dataset:
        return

    validate_data_quality(full_dataset)
    save_pickle(full_dataset, PROCESSED_DATA_FILE)

def quick_prototype_stage():
    """Stage 2a: Quick prototype using current processed data."""
    print("\n=== ETAP 2a: SZYBKI PROTOTYP ===")
    dataset = load_pickle(PROCESSED_DATA_FILE)
    if not dataset:
        print("! Najpierw przygotuj dane (Opcja 1).")
        return

    train_spk, test_spk = split_train_test_by_speaker(dataset, test_fraction=0.2, seed=42)
    
    # Zmienione przypisanie: train_ready to dane sklejone, test_samples_ready to lista próbek
    train_ready, test_samples_ready = prepare_training_data({k: dataset[k] for k in train_spk}, show_table=True)
    _, test_samples = prepare_training_data({k: dataset[k] for k in test_spk}, show_table=False)
    
    print("Trenowanie...")
    models = train_gmms(train_ready) 
    
    # Użycie test_samples w calculate_accuracy
    acc, _, _ = calculate_accuracy(models, test_samples)
    print(f"\n>> Skuteczność prototypu: {acc:.2f}%")
    save_models(models, "prototype_models.pkl")

def optimize_parameters_stage():
    """Stage 2b: Full parameter optimization."""
    # Optimization needs to reload raw files multiple times, so we don't load the pickle here.
    # The optimization module handles data loading internally.
    optimize_parameters_full()

def cross_validation_stage():
    """Stage 2c: Cross-validation."""
    print("\n=== ETAP 2c: WALIDACJA KRZYŻOWA ===")
    dataset = load_pickle(PROCESSED_DATA_FILE)
    if not dataset:
        print("! Najpierw przygotuj dane (Opcja 1).")
        return

    perform_cross_validation(dataset, k_folds=5)

def train_final_model_stage():
    """Stage 3: Train final model using best parameters."""
    print("\n=== ETAP 3: TRENING KOŃCOWY ===")
    
    # 1. Load Params
    mfcc_params, gmm_params = _load_best_or_default_params()
    
    # 2. Re-process data with these params
    print("Przetwarzanie danych źródłowych z wybranymi parametrami...")
    final_dataset = load_train_files_and_determine_mfcc(mfcc_params, show_table=False)
    
    if not final_dataset:
        print("Błąd przetwarzania danych.")
        return

    # 3. Prepare full training set (tylko pierwsza część krotki - sklejone dane)
    train_data_prepared, _ = prepare_training_data(final_dataset, show_table=True)
    
    # 4. Train
    print("Trenowanie modelu końcowego...")
    final_models = train_gmms(train_data_prepared, gmm_params)
    
    save_models(final_models, FINAL_MODEL_FILENAME)
    print(f"Zapisano model do: {FINAL_MODEL_FILENAME}")

def evaluate_system_stage():
    """Stage 4: Evaluation of the final model."""
    print("\n=== ETAP 4: EWALUACJA SYSTEMU ===")
    
    # 1. Load Model
    final_model = load_models(FINAL_MODEL_FILENAME)
    if not final_model:
        print("! Najpierw wytrenuj model końcowy (Opcja 3).")
        return

    # 2. Load Params
    mfcc_params, _ = _load_best_or_default_params()
    
    # 3. Load Data
    print("Wczytywanie danych do testów...")
    dataset = load_train_files_and_determine_mfcc(mfcc_params, show_table=False)
    
    if not dataset:
        return

    # 4. Split (30% for test)
    _, test_speakers = split_train_test_by_speaker(dataset, test_fraction=0.3, seed=42)
    test_data = {k: dataset[k] for k in test_speakers}
    
    # Tylko druga część krotki - próbki do testowania
    _, test_samples_ready = prepare_training_data(test_data, show_table=False)
    
    # 5. Evaluate (usunięto n_samples)
    metrics = evaluate_system(final_model, test_samples_ready)
    print_evaluation_report(metrics)