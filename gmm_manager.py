import numpy as np
from sklearn.mixture import GaussianMixture
from tabulate import tabulate
from typing import Dict, Tuple, Any, Optional, List

from decorators import safe_execution
from audio_processing import load_gmm_params
from utils import save_pickle, load_pickle
from config import DEFAULT_MODEL_FILENAME

@safe_execution("Błąd podczas treningu GMM")
def train_gmms(training_dict: Dict[str, np.ndarray], gmm_params: Optional[Dict] = None) -> Dict[str, GaussianMixture]:
    """Trains GMM models for each digit provided in the training dictionary."""
    
    if gmm_params is None:
        gmm_params = load_gmm_params()
    
    gmm_models = {}
    print(f"Rozpoczynam trening GMM (komponenty: {gmm_params['num_components']})...")

    for digit, mfcc in training_dict.items():
        # Validation checks
        if mfcc is None or not isinstance(mfcc, np.ndarray) or len(mfcc) == 0:
            print(f"  [!] Brak danych dla cyfry {digit}, pomijam.")
            continue
            
        print(f"  -> Trenowanie modelu dla cyfry {digit}...")
        
        gmm = GaussianMixture(
            n_components=gmm_params['num_components'],
            covariance_type=gmm_params['cov_type'],
            max_iter=gmm_params['max_iter'],
            random_state=gmm_params['random_state'],
            verbose=0
        )

        gmm.fit(mfcc)
        gmm_models[digit] = gmm
        
    print(f"Zakończono trening. Utworzono modele dla: {list(gmm_models.keys())}")
    return gmm_models

def classify_sample(gmm_models: Dict[str, GaussianMixture], mfcc: np.ndarray, show_table: bool = False) -> Tuple[Any, Dict[str, float]]:
    """Classifies a single MFCC sample against all GMM models."""
    scores = {}
    
    for digit, model in gmm_models.items():
        try:
            # score() returns log-likelihood
            scores[digit] = model.score(mfcc)
        except Exception as e:
            # We catch specific model errors here without breaking the loop
            scores[digit] = -np.inf

    # Find the digit with the highest score
    if not scores:
        return None, {}

    predicted = max(scores, key=scores.get)

    if show_table:
        _display_classification_scores(predicted, scores)

    return predicted, scores

def _display_classification_scores(predicted, scores):
    """Helper to visualize classification scores."""
    print("\nOceny modeli GMM:")
    print(f"Przewidywana liczba: {predicted}")
    
    # Sort by key (digit) for clean display
    ordered = sorted(scores.items(), key=lambda x: str(x[0]))
    
    # Transpose for horizontal table
    table_data = [{str(k): f"{v:.2f}" for k, v in ordered}]
    print(tabulate(table_data, headers="keys", tablefmt="grid"))

def save_models(gmm_models: Dict, filename: str = DEFAULT_MODEL_FILENAME) -> bool:
    return save_pickle(gmm_models, filename)

def load_models(filename: str = DEFAULT_MODEL_FILENAME) -> Optional[Dict]:
    models = load_pickle(filename)
    if models:
        print(f"Dostępne modele dla cyfr: {list(models.keys())}")
    return models

def generate_classification_results_for_eval(gmm_models: Dict[str, GaussianMixture], samples: List[Dict]) -> List[List[Any]]:
    """
    Przeprowadza klasyfikację dla listy próbek i przygotowuje dane pod eval_2025.
    Zwraca listę list: [[filename, predicted_label, max_score], ...]
    """
    results_rows = []
    
    print(f"Rozpoczynam klasyfikację {len(samples)} plików dla ewaluacji...")
    
    for sample in samples:
        filename = sample.get("filename")
        mfcc = sample.get("MFCC")
        
        # Pomijamy błędne próbki
        if filename is None or mfcc is None:
            continue
            
        # Używamy istniejącej funkcji classify_sample
        predicted_digit, scores = classify_sample(gmm_models, mfcc, show_table=False)
        
        if predicted_digit is not None:
            # Pobieramy wynik (log-likelihood) dla zwycięskiej cyfry
            max_score = scores[predicted_digit]
            results_rows.append([filename, predicted_digit, max_score])
        else:
            # W przypadku błędu klasyfikacji (np. puste dane)
            results_rows.append([filename, -1, -9999.0])
            
    return results_rows