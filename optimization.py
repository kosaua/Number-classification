import time
import itertools
from tabulate import tabulate
from typing import Dict, List, Any
import concurrent.futures
import os 

# Custom imports
from config import BEST_MFCC_PARAMS_FILE, BEST_GMM_PARAMS_FILE, OPTIMIZATION_RESULTS_FILE
from audio_processing import load_train_files_and_determine_mfcc, prepare_training_data, split_train_test_by_speaker
from gmm_manager import train_gmms
from evaluation import calculate_accuracy
from utils import save_params_to_csv, save_list_to_csv

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

def _worker_optimize_single(combination: tuple):
    """Helper function that processes a single combination"""    
    n_mfcc, n_fft, win_len, hop_len, n_mels, (delta, delta2), gmm_comp, gmm_cov = combination

    mfcc_params = {
        'n_mfcc': n_mfcc, 'n_fft': n_fft, 'win_length': win_len,
        'hop_length': hop_len, 'n_mels': n_mels,
        'count_delta': delta, 'count_delta_delta': delta2
    }
    gmm_params = {
        'num_components': gmm_comp, 'cov_type': gmm_cov,
        'max_iter': 200, 'random_state': 42
    }
    
    print(f"\nSTART MFCC: n_mfcc={n_mfcc}, n_fft={n_fft} | GMM: comp={gmm_comp}, cov={gmm_cov}")

    try:
        dataset = load_train_files_and_determine_mfcc(mfcc_params, show_table=False)
        if not dataset: return None

        train_spk, test_spk = split_train_test_by_speaker(dataset, test_fraction=0.2, seed=42)
        if not train_spk or not test_spk: return None

        train_data, _ = prepare_training_data({k: dataset[k] for k in train_spk}, show_table=False)
        _, test_samples = prepare_training_data({k: dataset[k] for k in test_spk}, show_table=False)

        models = train_gmms(train_data, gmm_params)

        accuracy, _, _ = calculate_accuracy(models, test_samples) 
        
        print(f"DONE MFCC: n_mfcc={n_mfcc}, n_fft={n_fft} | GMM: comp={gmm_comp}, cov={gmm_cov} -> Wynik: {accuracy:.2f}%")

        return {
            "mfcc_params": mfcc_params,
            "gmm_params": gmm_params,
            "test_accuracy": accuracy
        }

    except Exception as e:
        print(f" -> Błąd w iteracji (dla kombinacji {combination}): {e}")
        return None

def optimize_parameters_full():
    """Comprehensive Grid Search for MFCC and GMM parameters using parallel processing."""
    
    print("=== ROZPOCZĘCIE OPTYMALIZACJI PARAMETRÓW (Wielordzeniowo) ===")
    print("UWAGA: Proces może zająć dużo czasu, ale zostanie przyspieszony dzięki przetwarzaniu równoległemu.")

    combinations = _generate_param_combinations()
    total_combinations = len(combinations)
    
    results = []
    start_time = time.time()
    
    # Using ProcessPoolExecutor to run tasks in parallel
    # max_workers=None will use all available CPU cores
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        
        futures = {executor.submit(_worker_optimize_single, comb): comb for comb in combinations}
        
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            if result:
                results.append(result)
            
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining_tasks = total_combinations - i
            remaining = avg_time * remaining_tasks
            
            print(f"\nPOSTĘP: {i}/{total_combinations} wykonano. Pozostało: ~{remaining/60:.1f} min. Czas miniony: {elapsed/60:.1f} min.")
            
    return _process_optimization_results(results)

def _process_optimization_results(results: List[Dict]):
    """Analyzes results, saves to CSV, and updates best params."""
    if not results:
        print("Nie znaleziono żadnych wyników!")
        return None

    results.sort(key=lambda x: x["test_accuracy"], reverse=True)
    best_result = results[0]

    # Save and print
    flat_data = [
        _flatten_result(res, rank=i+1) 
        for i, res in enumerate(results)
    ]
    save_list_to_csv(flat_data, OPTIMIZATION_RESULTS_FILE)
    _save_best_config(best_result)
    _print_leaderboard(results[:10])
    
    return best_result

def _flatten_result(result: Dict, rank: int) -> Dict[str, Any]:
    """Helper: Flattens nested parameter dicts into a single-level dict for CSV."""
    m = result['mfcc_params']
    g = result['gmm_params']
    
    return {
        'rank': rank,
        'accuracy': f"{result['test_accuracy']:.2f}",
        # MFCC
        'n_mfcc': m['n_mfcc'],
        'n_fft': m['n_fft'],
        'win_len': m['win_length'],
        'hop_len': m['hop_length'],
        'n_mels': m['n_mels'],
        'delta': m['count_delta'],
        'delta2': m['count_delta_delta'],
        # GMM
        'gmm_comp': g['num_components'],
        'gmm_cov': g['cov_type']
    }

def _save_best_config(best_result: Dict):
    """Helper: Saves the best configuration to separate config files."""
    print(f"\n=== NAJLEPSZY WYNIK (Acc: {best_result['test_accuracy']:.2f}%) ===")
    save_params_to_csv(best_result['mfcc_params'], BEST_MFCC_PARAMS_FILE)
    save_params_to_csv(best_result['gmm_params'], BEST_GMM_PARAMS_FILE)

def _print_leaderboard(top_results: List[Dict]):
    """Helper: Displays the top results table."""
    table_data = []
    for rank, res in enumerate(top_results, 1):
        m = res['mfcc_params']
        g = res['gmm_params']
        table_data.append({
            "Rank": rank,
            "Acc": f"{res['test_accuracy']:.2f}%",
            "MFCC": f"n={m['n_mfcc']}, fft={m['n_fft']}",
            "Deltas": f"{'Yes' if m['count_delta'] else 'No'}",
            "GMM": f"k={g['num_components']} ({g['cov_type']})"
        })
    
    print("\nTOP 10 WYNIKÓW:")
    print(tabulate(table_data, headers="keys", tablefmt="simple_grid"))
    print(f"Pełna historia: {OPTIMIZATION_RESULTS_FILE}")