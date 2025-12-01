import time
import itertools
from tabulate import tabulate
from typing import Dict, List, Any

# Custom imports
from config import BEST_MFCC_PARAMS_FILE, BEST_GMM_PARAMS_FILE
from audio_processing import load_train_files_and_determine_mfcc, prepare_training_data, split_train_test_by_speaker
from gmm_manager import train_gmms
from evaluation import calculate_accuracy
from utils import save_params_to_csv

def _generate_param_combinations():
    """Generates all combinations of parameters to test."""
    # Define ranges
    n_mfcc_vals = [13, 16, 20]
    n_fft_vals = [512, 1024]
    win_len_vals = [400, 512]
    hop_len_vals = [160, 200]
    n_mels_vals = [26, 32]
    delta_opts = [(False, False), (True, False), (True, True)]
    gmm_comp_vals = [4, 8, 12]
    gmm_cov_vals = ['diag', 'tied']

    # Cartesian product of all lists
    return list(itertools.product(
        n_mfcc_vals, n_fft_vals, win_len_vals, hop_len_vals, n_mels_vals, 
        delta_opts, gmm_comp_vals, gmm_cov_vals
    ))

def optimize_parameters_full():
    """Comprehensive Grid Search for MFCC and GMM parameters."""
    print("=== ROZPOCZĘCIE OPTYMALIZACJI PARAMETRÓW ===")
    print("UWAGA: Proces może zająć dużo czasu...")

    combinations = _generate_param_combinations()
    total_combinations = len(combinations)
    
    results = []
    start_time = time.time()

    for i, (n_mfcc, n_fft, win_len, hop_len, n_mels, (delta, delta2), gmm_comp, gmm_cov) in enumerate(combinations, 1):
        
        # Constraint check
        if win_len > n_fft:
            continue

        # Prepare Params
        mfcc_params = {
            'n_mfcc': n_mfcc, 'n_fft': n_fft, 'win_length': win_len,
            'hop_length': hop_len, 'n_mels': n_mels,
            'count_delta': delta, 'count_delta_delta': delta2
        }

        # Progress Info
        elapsed = time.time() - start_time
        avg_time = elapsed / i
        remaining = avg_time * (total_combinations - i)
        
        print(f"\nKombinacja {i}/{total_combinations} | Pozostało: ~{remaining/60:.1f} min")
        print(f"MFCC: {mfcc_params} | GMM: comp={gmm_comp}, cov={gmm_cov}")

        try:
            # 1. Load Data (Expensive Step)
            # Note: In a production system, we would cache MFCCs, but n_fft changes the base features
            dataset = load_train_files_and_determine_mfcc(mfcc_params, show_table=False)
            if not dataset: continue

            # 2. Split
            train_spk, test_spk = split_train_test_by_speaker(dataset, test_fraction=0.2, seed=42)
            if not train_spk or not test_spk: continue

            train_data = prepare_training_data({k: dataset[k] for k in train_spk}, show_table=False)
            test_data = prepare_training_data({k: dataset[k] for k in test_spk}, show_table=False)

            # 3. Train
            gmm_params = {
                'num_components': gmm_comp, 'cov_type': gmm_cov,
                'max_iter': 200, 'random_state': 42
            }
            models = train_gmms(train_data, gmm_params)

            # 4. Evaluate
            accuracy, _, _ = calculate_accuracy(models, test_data, n_samples=3)
            print(f" -> Wynik: {accuracy:.2f}%")

            results.append({
                "mfcc_params": mfcc_params,
                "gmm_params": gmm_params,
                "test_accuracy": accuracy
            })

        except Exception as e:
            print(f" -> Błąd w iteracji: {e}")

    return _process_optimization_results(results)

def _process_optimization_results(results: List[Dict]):
    """Analyzes and saves the best results from optimization."""
    if not results:
        print("Nie znaleziono żadnych wyników!")
        return None

    # Sort by accuracy descending
    results.sort(key=lambda x: x["test_accuracy"], reverse=True)
    best = results[0]

    print(f"\n=== NAJLEPSZE WYNIKI (Acc: {best['test_accuracy']:.2f}%) ===")
    
    # Save to CSV
    save_params_to_csv(best['mfcc_params'], BEST_MFCC_PARAMS_FILE)
    save_params_to_csv(best['gmm_params'], BEST_GMM_PARAMS_FILE)

    # Display Top 10
    table_data = []
    for rank, res in enumerate(results[:10], 1):
        m = res['mfcc_params']
        g = res['gmm_params']
        table_data.append({
            "Rank": rank,
            "Acc %": f"{res['test_accuracy']:.2f}",
            "MFCC": m['n_mfcc'],
            "FFT": m['n_fft'],
            "Delta": "✓" if m['count_delta'] else "-",
            "GMM Comp": g['num_components'],
            "GMM Cov": g['cov_type']
        })
    
    print(tabulate(table_data, headers="keys", tablefmt="grid"))
    return best