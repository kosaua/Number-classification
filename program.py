from ntpath import join
import sys
import librosa
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from collections import defaultdict
from tabulate import tabulate
from os import listdir
from os.path import isfile
import csv
import os
import pickle
import random
import time
import io
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

DEFAULT_MFCC_PARAMS = {
    'n_mfcc': 13,
    'n_fft': 512,
    'win_length': 400,
    'hop_length': 160,
    'n_mels': 26,
    'count_delta': True,
    'count_delta_delta': False
}

DEFAULT_GMM_PARAMS = {
    'n_components': 8,
    'covariance_type': 'diag',
    'max_iter': 200,
    'random_state': 42
}


def get_mfcc(x, fs, n_mfcc, n_fft, win_length, hop_length, n_mels, count_delta, count_delta_delta):
    """Oblicza współczynniki MFCC z opcjonalnymi deltami i delta-deltami"""
    try:
        mfcc = librosa.feature.mfcc(
            y=x, sr=fs, n_mfcc=n_mfcc, n_fft=n_fft,
            win_length=win_length, hop_length=hop_length, n_mels=n_mels
        ).T

        first_feature_id = 1
        mfcc = mfcc[:, first_feature_id:]
        
        features = [mfcc]
        
        if count_delta:
            delta = librosa.feature.delta(mfcc, axis=0)
            features.append(delta)
            
        if count_delta_delta:
            delta_delta = librosa.feature.delta(mfcc, axis=0, order=2)
            features.append(delta_delta)
        
        if len(features) > 1:
            mfcc = np.concatenate(features, axis=1)

    except Exception as e:
        print(f"Błąd podczas obliczania MFCC: {e}")
        return None

    return mfcc


def load_mfcc_params():
    """Wczytuje parametry MFCC z pliku lub używa domyślnych"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mfcc_params_path = os.path.join(base_dir, "mfcc_params.csv")
    
    def parse_value(v):
        v = v.strip()
        if v.lower() in ["true", "false"]:
            return v.lower() == "true"
        try:
            return int(v)
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return v

    if os.path.exists(mfcc_params_path):
        try:
            with open(mfcc_params_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                params = {k: parse_value(v) for k, v in next(reader).items()}
            
            print("Parametry MFCC (wczytane z pliku):")
            table_data = [{"parametr": k, "wartość": v} for k, v in params.items()]
            print(tabulate(table_data, headers="keys", tablefmt="grid"))
            return params
        except Exception as e:
            print(f"Błąd podczas wczytywania parametrów MFCC z pliku: {e}")
    
    print("Używam domyślnych parametrów MFCC:")
    params = DEFAULT_MFCC_PARAMS.copy()
    table_data = [{"parametr": k, "wartość": v} for k, v in params.items()]
    print(tabulate(table_data, headers="keys", tablefmt="grid"))
    return params


def load_gmm_params():
    """Wczytuje parametry GMM z pliku lub używa domyślnych"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    gmm_params_path = os.path.join(base_dir, "gmm_params.csv")
    
    def parse_value(v):
        v = v.strip()
        if v.lower() in ["true", "false"]:
            return v.lower() == "true"
        try:
            return int(v)
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return v

    if os.path.exists(gmm_params_path):
        try:
            with open(gmm_params_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                params = {k: parse_value(v) for k, v in next(reader).items()}
            
            print("Parametry GMM (wczytane z pliku):")
            table_data = [{"parametr": k, "wartość": v} for k, v in params.items()]
            print(tabulate(table_data, headers="keys", tablefmt="grid"))
            return params
        except Exception as e:
            print(f"Błąd podczas wczytywania parametrów GMM z pliku: {e}")
    
    print("Używam domyślnych parametrów GMM:")
    params = DEFAULT_GMM_PARAMS.copy()
    table_data = [{"parametr": k, "wartość": v} for k, v in params.items()]
    print(tabulate(table_data, headers="keys", tablefmt="grid"))
    return params


def get_drive():
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    else:
        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return build('drive', 'v3', credentials=creds)


def download_data(folder_id, target="downloaded"):
    service = get_drive()

    results = service.files().list(
        q=f"'{folder_id}' in parents", fields="files(id, name, mimeType, size)", pageSize=1000
        ).execute()
    
    files = results.get('files', [])
    downloaded = 0

    os.makedirs(target, exist_ok=True)
    for f in files:
        if f.get('mimeType') != 'application/vnd.google-apps.folder':
            request = service.files().get_media(fileId=f['id'])
            with io.FileIO(os.path.join(target, f['name']), 'wb') as file:
                downloader = MediaIoBaseDownload(file, request)
                while not downloader.next_chunk()[1]:
                    pass
    
            downloaded += 1
    print(f"\nPobrano {downloaded}/{len(files)} plików")
    return target


def load_train_files_and_determine_mfcc(mfcc_params, showTable=False):
    """Wczytuje pliki audio i oblicza dla nich MFCC"""

    folder_id = "1WQVB4mqdNBSvpa1SZ8EbUc5eJ--e1t6y"
    train_data_dir = download_data(folder_id)

    if not os.path.exists(train_data_dir):
        print(f"Brak folderu {train_data_dir}!")
        return None

    short_wavpaths = [f for f in listdir(train_data_dir) if isfile(join(train_data_dir, f)) and f.lower().endswith('.wav')]

    if not short_wavpaths:
        print("Nie znaleziono plików WAV w folderze train_data!")
        return None

    wavpaths = [join(train_data_dir, f) for f in short_wavpaths]
    wavpaths.sort()
    short_wavpaths.sort()

    sound_data = defaultdict(list)

    print(f"Znaleziono {len(wavpaths)} plików audio.")

    for i, f in enumerate(wavpaths):
        file_name = short_wavpaths[i]
        try:
            x, fs = librosa.load(f, sr=16000, mono=True)
            mfcc = get_mfcc(x, fs, **mfcc_params)
        except Exception as e:
            print(f"Błąd przetwarzania pliku {f}: {e}")
            mfcc = None

        sound_data[file_name[0:2]].append({
            "MFCC": mfcc,
            "num": file_name[6],
            "filename": file_name
        })
    
    print(f"Wczytano pliki i wyznaczono z nich macierze MFCC.")

    if showTable:
        rows = []
        for spk, items in sound_data.items():
            for d in items:
                row = {
                    "speaker": spk,
                    "num": d["num"],
                    "filename": d["filename"]
                }

                if isinstance(d.get("MFCC"), np.ndarray):
                    row["MFCC"] = f"shape={d['MFCC'].shape}"
                else:
                    row["MFCC"] = "None"

                rows.append(row)

        print(tabulate(rows, headers="keys", tablefmt="grid"))
    
    return sound_data


def prepare_training_data(sound_data, showTable=False):
    """Przygotowuje dane do treningu GMM"""
    training_data = defaultdict(list)

    for speaker, samples in sound_data.items():
        for s in samples:
            mfcc = s.get("MFCC")
            label = s.get("num")

            if mfcc is None or not isinstance(mfcc, np.ndarray):
                continue

            training_data[label].append(mfcc)

    for label in training_data:
        if training_data[label]:
            training_data[label] = np.vstack(training_data[label])

    if showTable:
        print("Przygotowane dane do treningu GMM:")
        summary = []
        for k, v in training_data.items():
            if isinstance(v, np.ndarray):
                summary.append({"cyfra": k, "liczba ramek": v.shape[0], "liczba cech": v.shape[1]})
            else:
                summary.append({"cyfra": k, "liczba ramek": 0, "liczba cech": 0})
        print(tabulate(summary, headers="keys", tablefmt="grid"))

    print("Przygotowano dane do treningu GMM")
    return training_data


def save_processed_dataset(dataset, filename="dataset_processed.pkl"):
    """Zapisuje przetworzony dataset do pliku"""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Dataset zapisano do pliku: {filename}")
        return True
    except Exception as e:
        print(f"Błąd podczas zapisywania datasetu: {e}")
        return False


def load_processed_dataset(filename="dataset_processed.pkl"):
    """Wczytuje przetworzony dataset z pliku"""
    try:
        with open(filename, 'rb') as f:
            dataset = pickle.load(f)
        print(f"Dataset wczytano z pliku: {filename}")
        return dataset
    except Exception as e:
        print(f"Błąd podczas wczytywania datasetu: {e}")
        return None


def validate_data_quality(dataset):
    """Sprawdza jakość danych i generuje raport"""
    stats = {
        'total_samples': 0,
        'samples_per_digit': defaultdict(int),
        'samples_per_speaker': defaultdict(int),
        'missing_mfcc': 0,
        'avg_frames_per_digit': defaultdict(float),
        'digits_with_data': 0,
        'speakers_with_data': 0
    }
    
    for speaker, samples in dataset.items():
        speaker_samples = 0
        stats['samples_per_speaker'][speaker] = 0
        
        for sample in samples:
            if isinstance(sample, dict) and sample.get("MFCC") is not None and isinstance(sample.get("MFCC"), np.ndarray):
                digit = sample.get("num")
                mfcc_data = sample["MFCC"]
                
                stats['samples_per_digit'][digit] += 1
                stats['total_samples'] += 1
                speaker_samples += 1
                
                if mfcc_data.shape[0] > 0:
                    stats['avg_frames_per_digit'][digit] = mfcc_data.shape[0]
            else:
                stats['missing_mfcc'] += 1
        
        stats['samples_per_speaker'][speaker] = speaker_samples
        if speaker_samples > 0:
            stats['speakers_with_data'] += 1
    
    stats['digits_with_data'] = len([d for d, count in stats['samples_per_digit'].items() if count > 0])
    
    print("\n=== RAPORT JAKOŚCI DANYCH ===")
    print(f"Łączna liczba próbek: {stats['total_samples']}")
    print(f"Cyfry z danymi: {stats['digits_with_data']}/10")
    print(f"Mówcy z danymi: {stats['speakers_with_data']}/{len(dataset)}")
    print(f"Błędne próbki MFCC: {stats['missing_mfcc']}")
    
    print("\nPRÓBKI PER CYFRA:")
    for digit in sorted(stats['samples_per_digit'].keys()):
        count = stats['samples_per_digit'][digit]
        avg_frames = stats['avg_frames_per_digit'].get(digit, 0)
        print(f"  Cyfra {digit}: {count} próbek, średnio {avg_frames:.1f} ramek")
    
    print("\nPRÓBKI PER MÓWCA:")
    for speaker in sorted(stats['samples_per_speaker'].keys()):
        count = stats['samples_per_speaker'][speaker]
        print(f"  Mówca {speaker}: {count} próbek")
    
    return stats


def split_train_test_by_speaker(sound_data, test_fraction=0.2, seed=None):
    """Dzieli dane na zbiór treningowy i testowy według mówców"""
    if seed is not None:
        random.seed(seed)

    speakers = list(sound_data.keys())
    num_test = max(1, int(len(speakers) * test_fraction))
    test_speakers = random.sample(speakers, num_test)
    train_speakers = [s for s in speakers if s not in test_speakers]

    print(f"Mówcy do treningu: {train_speakers}")
    print(f"Mówcy do testu: {test_speakers}")

    return train_speakers, test_speakers


def train_gmms(training_dict, gmm_params=None):
    """Trenuje modele GMM dla każdej cyfry"""
    if gmm_params is None:
        gmm_params = load_gmm_params()
    
    gmm_models = {}
    
    print(f"Trenowanie GMM z parametrami: {gmm_params}")

    for digit, mfcc in training_dict.items():
        if mfcc is None or not isinstance(mfcc, np.ndarray) or len(mfcc) == 0:
            print(f"Brak danych dla cyfry {digit}, pomijam...")
            continue
            
        # print(f"Trenowanie modelu dla cyfry {digit}...")
        gmm = GaussianMixture(
            n_components=gmm_params['n_components'],
            covariance_type=gmm_params['covariance_type'],
            max_iter=gmm_params.get('max_iter', 200),
            random_state=gmm_params.get('random_state',42),
            verbose=0)
        

        gmm.fit(mfcc)
        gmm_models[digit] = gmm
        # print(f"  Wytrenowano model dla cyfry: {digit}")

    print("Trenowanie wszystkich modeli zakończone.")
    return gmm_models


def classifier(gmm_models, mfcc, showTable=False):
    """Klasyfikuje próbkę MFCC za pomocą modeli GMM"""
    scores = {}
    for number, model in gmm_models.items():
        try:
            score = model.score(mfcc)
            scores[number] = score
        except Exception as e:
            print(f"Model {number} zgłosił błąd: {e}")
            scores[number] = -np.inf

    predicted = max(scores, key=scores.get)

    if showTable:
        print("\nOceny modeli GMM dla podanej próbki MFCC:")
        print('Przewidywana liczba:', predicted)
        ordered = sorted(scores.items(), key=lambda x: x[0])
        score_table_transposed = [{str(number): f"{score:.2f}" for number, score in ordered}]
        print(tabulate(score_table_transposed, headers="keys", tablefmt="grid"))

    return predicted, scores


def calculate_accuracy(gmm_models, test_data, n_samples=5):
    """Oblicza dokładność klasyfikacji na zbiorze testowym"""
    correct = 0
    tested = 0
    all_predictions = []
    all_true_labels = []

    for true_label, mfcc_matrix in test_data.items():
        if mfcc_matrix is None or not isinstance(mfcc_matrix, np.ndarray) or len(mfcc_matrix) < n_samples:
            continue

        idx = np.random.choice(mfcc_matrix.shape[0], n_samples, replace=False)

        for i in idx:
            sample_mfcc = mfcc_matrix[i:i+1, :] 
            predicted, scores = classifier(gmm_models, sample_mfcc, showTable=False)

            all_predictions.append(predicted)
            all_true_labels.append(true_label)

            if predicted == true_label:
                correct += 1
            tested += 1
        
    accuracy = correct / tested * 100 if tested > 0 else 0
    return accuracy, all_predictions, all_true_labels


def save_gmm_models(gmm_models, filename="gmm_models.pkl"):
    """Zapisuje modele GMM do pliku"""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(gmm_models, f)
        print(f"Modele GMM zapisano do pliku: {filename}")
        return True
    except Exception as e:
        print(f"Błąd podczas zapisywania modeli GMM: {e}")
        return False


def load_gmm_models(filename="gmm_models.pkl"):
    """Wczytuje modele GMM z pliku"""
    try:
        with open(filename, 'rb') as f:
            gmm_models = pickle.load(f)
        print(f"Modele GMM wczytano z pliku: {filename}")
        print(f"Wczytano modele dla cyfr: {list(gmm_models.keys())}")
        return gmm_models
    except Exception as e:
        print(f"Błąd podczas wczytywania modeli GMM: {e}")
        return None


def save_best_parameters(best_mfcc_params, best_gmm_params):
    """Zapisuje najlepsze parametry do plików CSV"""
    try:
        with open("best_mfcc_params.csv", "w", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=best_mfcc_params.keys())
            writer.writeheader()
            writer.writerow(best_mfcc_params)
        
        with open("best_gmm_params.csv", "w", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=best_gmm_params.keys())
            writer.writeheader()
            writer.writerow(best_gmm_params)
        
        print("Najlepsze parametry zapisano do plików:")
        print("  - best_mfcc_params.csv")
        print("  - best_gmm_params.csv")
        return True
    except Exception as e:
        print(f"Błąd podczas zapisywania parametrów: {e}")
        return False


def load_best_parameters():
    """Wczytuje najlepsze parametry z plików"""
    best_mfcc = load_mfcc_params()
    best_gmm = load_gmm_params()
    return best_mfcc, best_gmm


def cross_validation(dataset, k_folds=5, n_samples=3):
    """Przeprowadza k-fold cross validation"""
    print(f"Rozpoczynam {k_folds}-krotną walidację krzyżową...")
    
    speakers = list(dataset.keys())
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_accuracies = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(speakers), 1):
        print(f"\n--- Fold {fold}/{k_folds} ---")
        
        train_speakers = [speakers[i] for i in train_idx]
        test_speakers = [speakers[i] for i in test_idx]
        
        print(f"Mówcy treningowi: {train_speakers}")
        print(f"Mówcy testowi: {test_speakers}")
        
        # Przygotowanie danych
        train_data_dict = {k: dataset[k] for k in train_speakers}
        test_data_dict = {k: dataset[k] for k in test_speakers}
        
        train_data = prepare_training_data(train_data_dict, False)
        test_data = prepare_training_data(test_data_dict, False)
        
        # Trening modeli
        models = train_gmms(train_data)
        
        # Testowanie
        accuracy, _, _ = calculate_accuracy(models, test_data, n_samples=n_samples)
        fold_accuracies.append(accuracy)
        print(f"Skuteczność fold {fold}: {accuracy:.2f}%")
    
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    
    print(f"\n=== WYNIKI WALIDACJI KRZYŻOWEJ ===")
    print(f"Średnia skuteczność: {mean_accuracy:.2f}%")
    print(f"Odchylenie standardowe: {std_accuracy:.2f}%")
    print(f"Wyniki per fold: {[f'{acc:.2f}%' for acc in fold_accuracies]}")
    
    return {
        'fold_accuracies': fold_accuracies,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy
    }


def evaluate_system(gmm_models, test_dataset, n_samples=5):
    """Kompleksowa ewaluacja systemu z różnymi metrykami"""
    print("Przeprowadzam ewaluację systemu...")
    
    all_predictions = []
    all_true_labels = []
    
    for true_label, mfcc_matrix in test_dataset.items():
        if mfcc_matrix is None or not isinstance(mfcc_matrix, np.ndarray) or len(mfcc_matrix) == 0:
            continue
            
        n_test = min(n_samples, len(mfcc_matrix))
        idx = np.random.choice(mfcc_matrix.shape[0], n_test, replace=False)
        
        for i in idx:
            sample_mfcc = mfcc_matrix[i:i+1, :]
            predicted, _ = classifier(gmm_models, sample_mfcc, showTable=False)
            all_true_labels.append(true_label)
            all_predictions.append(predicted)
    
    if not all_true_labels:
        print("Brak danych do ewaluacji!")
        return None
    
    accuracy = accuracy_score(all_true_labels, all_predictions) * 100
    precision = precision_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
    
    report = classification_report(all_true_labels, all_predictions, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_true_labels, all_predictions)
    
    per_class_metrics = {}
    for digit in sorted(set(all_true_labels)):
        if digit in report:
            per_class_metrics[digit] = {
                'precision': report[digit]['precision'],
                'recall': report[digit]['recall'],
                'f1_score': report[digit]['f1-score'],
                'support': report[digit]['support']
            }
    
    return {
        'overall_accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'per_class_metrics': per_class_metrics,
        'total_tested': len(all_true_labels)
    }


def print_confusion_matrix(cm, labels=None):
    """Wyświetla macierz pomyłek w ładnej formie"""
    if labels is None:
        labels = [str(i) for i in range(10)]
    
    headers = [f"Pred {label}" for label in labels]
    rows = []
    
    for i, true_label in enumerate(labels):
        row = [f"True {true_label}"] + [str(cm[i][j]) for j in range(len(labels))]
        rows.append(row)
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))


def optimize_parameters_full(dataset):
    """Kompleksowa optymalizacja parametrów MFCC i GMM"""
    print("=== ROZPOCZĘCIE OPTYMALIZACJI PARAMETRÓW ===")
    
    n_mfcc_values = [13, 16, 20]
    n_fft_values = [512, 1024]
    win_length_values = [400, 512]
    hop_length_values = [160, 200]
    n_mels_values = [26, 32]
    
    delta_combinations = [
        (False, False),  
        (True, False),   
        (True, True)     
    ]
    
    gmm_components = [4, 8, 12]
    gmm_cov_types = ['diag', 'tied']
    
    results = []
    combination_count = 0
    total_combinations = (len(n_mfcc_values) * len(n_fft_values) * len(win_length_values) * 
                         len(hop_length_values) * len(n_mels_values) * len(delta_combinations) *
                         len(gmm_components) * len(gmm_cov_types))
    
    start_time = time.time()
    
    for n_mfcc in n_mfcc_values:
        for n_fft in n_fft_values:
            for win_length in win_length_values:
                for hop_length in hop_length_values:
                    for n_mels in n_mels_values:
                        for delta_combo in delta_combinations:
                            
                            if win_length > n_fft:
                                continue
                                
                            mfcc_params = {
                                'n_mfcc': n_mfcc,
                                'n_fft': n_fft,
                                'win_length': win_length,
                                'hop_length': hop_length,
                                'n_mels': n_mels,
                                'count_delta': delta_combo[0],
                                'count_delta_delta': delta_combo[1]
                            }
                            
                            try:
                                print(f"\nPrzetwarzanie MFCC: {mfcc_params}")
                                processed_data = load_train_files_and_determine_mfcc(mfcc_params, False)
                                
                                if processed_data is None or len(processed_data) == 0:
                                    continue
                                
                                train_speakers, test_speakers = split_train_test_by_speaker(processed_data, test_fraction=0.2, seed=42)
                                
                                if len(train_speakers) == 0 or len(test_speakers) == 0:
                                    continue
                                
                                train_data = prepare_training_data({k: processed_data[k] for k in train_speakers}, False)
                                test_data = prepare_training_data({k: processed_data[k] for k in test_speakers}, False)
                                
                                if len(train_data) == 0 or len(test_data) == 0:
                                    continue
                                
                                for n_components in gmm_components:
                                    for cov_type in gmm_cov_types:
                                        
                                        combination_count += 1
                                        elapsed_time = time.time() - start_time
                                        estimated_total = (elapsed_time / combination_count) * total_combinations
                                        remaining_time = estimated_total - elapsed_time
                                        
                                        print(f"Kombinacja {combination_count}/{total_combinations}")
                                        print(f"MFCC({n_mfcc},{n_fft},{win_length},{hop_length},{n_mels}) "
                                              f"GMM({n_components},{cov_type})")
                                        print(f"Pozostały czas: {remaining_time/60:.1f} minut")
                                        
                                        try:
                                            gmm_params = {
                                                'n_components': n_components,
                                                'covariance_type': cov_type,
                                                'max_iter': 200,
                                                'random_state': 42
                                            }
                                            
                                            models = train_gmms(train_data, gmm_params)
                                            accuracy, _, _ = calculate_accuracy(models, test_data, n_samples=3)
                                            

                                            result = {
                                                "mfcc_params": mfcc_params.copy(),
                                                "gmm_params": gmm_params.copy(),
                                                "test_accuracy": accuracy,
                                                "total_features": next(iter(train_data.values())).shape[1] if train_data else 0
                                            }
                                            
                                            results.append(result)
                                            print(f"  Wynik: {accuracy:.2f}%")
                                            
                                        except Exception as e:
                                            print(f"  Błąd: {e}")
                                            continue
                                            
                            except Exception as e:
                                print(f"Błąd przetwarzania: {e}")
                                continue
    
    if results:
        results.sort(key=lambda x: x["test_accuracy"], reverse=True)
        best_result = results[0]
        
        print(f"\n=== NAJLEPSZE PARAMETRY ===")
        print(f"Skuteczność: {best_result['test_accuracy']:.2f}%")
        print(f"Parametry MFCC: {best_result['mfcc_params']}")
        print(f"Parametry GMM: {best_result['gmm_params']}")
        
        save_best_parameters(best_result['mfcc_params'], best_result['gmm_params'])
        
        print(f"\n=== TOP 10 KOMBINACJI ===")
        top_results = results[:10]
        summary_table = []
        
        for i, result in enumerate(top_results, 1):
            mfcc = result["mfcc_params"]
            gmm = result["gmm_params"]
            
            summary_table.append({
                "Rank": i,
                "n_mfcc": mfcc["n_mfcc"],
                "n_fft": mfcc["n_fft"],
                "win_len": mfcc["win_length"],
                "hop_len": mfcc["hop_length"],
                "n_mels": mfcc["n_mels"],
                "delta": "✓" if mfcc["count_delta"] else "✗",
                "delta2": "✓" if mfcc["count_delta_delta"] else "✗",
                "GMM_comp": gmm["n_components"],
                "GMM_cov": gmm["covariance_type"],
                "Accuracy%": f"{result['test_accuracy']:.2f}"
            })
        
        print(tabulate(summary_table, headers="keys", tablefmt="grid"))
        
        return best_result
    else:
        print("Nie udało się znaleźć żadnych wyników!")
        return None


def prepare_data_stage():
    """Etap 1: Przygotowanie danych"""
    print("=== PRZYGOTOWANIE DANYCH ===")
    mfcc_params = load_mfcc_params()
    
    print("Wczytywanie danych i Ekstrakcja cech MFCC z datasetu...")
    full_dataset = load_train_files_and_determine_mfcc(mfcc_params, True)
    
    if full_dataset is None:
        print("Nie udało się wczytać danych!")
        return
    
    print("Walidacja jakości danych...")
    data_quality = validate_data_quality(full_dataset)
    
    save_processed_dataset(full_dataset, "dataset_processed.pkl")
    print("Dane przygotowane i zapisane.")


def quick_prototype():
    """Etap 2a: Szybki prototyp"""
    print("=== SZYBKI PROTOTYP ===")
    dataset = load_processed_dataset()
    if dataset is None:
        print("Najpierw przygotuj dane (opcja 1)!")
        return
    
    train_speakers, test_speakers = split_train_test_by_speaker(dataset, 0.2, seed=42)
    train_data = {k: dataset[k] for k in train_speakers}
    test_data = {k: dataset[k] for k in test_speakers}
    
    train_data_prepared = prepare_training_data(train_data, True)
    test_data_prepared = prepare_training_data(test_data, True)
    
    print("Rozpoczynam trening...")
    models = train_gmms(train_data_prepared)
    accuracy, _, _ = calculate_accuracy(models, test_data_prepared, n_samples=5)
    print(f"Skuteczność prototypu: {accuracy:.2f}%")
    
    save_gmm_models(models, "prototype_models.pkl")


def optimize_parameters_stage():
    """Etap 2b: Optymalizacja parametrów"""
    print("=== OPTYMALIZACJA PARAMETRÓW ===")
    dataset = load_processed_dataset()
    if dataset is None:
        print("Najpierw przygotuj dane (opcja 1)!")
        return
    
    best_result = optimize_parameters_full(dataset)
    if best_result:
        print("Optymalizacja zakończona pomyślnie!")
    else:
        print("Optymalizacja nie powiodła się!")


def cross_validation_stage():
    """Etap 2c: Testy krzyżowe"""
    print("=== TESTY KRZYŻOWE ===")
    dataset = load_processed_dataset()
    if dataset is None:
        print("Najpierw przygotuj dane (opcja 1)!")
        return
    
    cv_results = cross_validation(dataset, k_folds=5, n_samples=3)
    
    if cv_results:
        print("\n=== PODSUMOWANIE WALIDACJI KRZYŻOWEJ ===")
        print(f"Ostateczna średnia skuteczność: {cv_results['mean_accuracy']:.2f}%")
        print(f"Stabilność modelu: {cv_results['std_accuracy']:.2f}% (odchylenie)")


def train_final_model():
    """Etap 3: Trening końcowego modelu"""
    print("=== TRENING KOŃCOWEGO MODELU ===")
    dataset = load_processed_dataset()
    if dataset is None:
        print("Najpierw przygotuj dane (opcja 1)!")
        return
    
    best_mfcc, best_gmm = load_best_parameters()
    print(f"Używam parametrów: MFCC={best_mfcc}, GMM={best_gmm}")
    
    print("Przetwarzanie danych z optymalnymi parametrami MFCC...")
    final_dataset = load_train_files_and_determine_mfcc(best_mfcc, False)
    if final_dataset is None:
        print("Błąd przetwarzania danych!")
        return
        
    train_data_prepared = prepare_training_data(final_dataset, True)
    
    print("Trening końcowego modelu na pełnym datasetcie...")
    final_models = train_gmms(train_data_prepared, best_gmm)
    save_gmm_models(final_models, "final_classifier.pkl")
    print("Końcowy klasyfikator zapisany!")


def evaluate_system_stage():
    """Etap 4: Ewaluacja systemu"""
    print("=== EWALUACJA SYSTEMU ===")
    final_model = load_gmm_models("final_classifier.pkl")
    if final_model is None:
        print("Najpierw wytrenuj końcowy model (opcja 3)!")
        return
    
    dataset = load_processed_dataset()
    if dataset is None:
        print("Brak danych testowych!")
        return
    
    train_speakers, test_speakers = split_train_test_by_speaker(dataset, 0.2, seed=42)
    test_data = {k: dataset[k] for k in test_speakers}
    test_data_prepared = prepare_training_data(test_data, True)
    
    print("Przeprowadzam kompleksową ewaluację")
    evaluation_results = evaluate_system(final_model, test_data_prepared, n_samples=5)
    
    if evaluation_results:
        print("\n" + "="*50)
        print("RAPORT EWALUACJI SYSTEMU")
        print("="*50)
        print(f"Skuteczność całkowita: {evaluation_results['overall_accuracy']:.2f}%")
        print(f"Precyzja: {evaluation_results['precision']:.3f}")
        print(f"Czułość: {evaluation_results['recall']:.3f}")
        print(f"F1-score: {evaluation_results['f1_score']:.3f}")
        print(f"Przetestowanych próbek: {evaluation_results['total_tested']}")
        
        print("\nMACIERZ POMYŁEK:")
        labels = sorted(set([str(k) for k in test_data_prepared.keys()]))
        print_confusion_matrix(evaluation_results['confusion_matrix'], labels)
        
        print("\nMETRYKI PER CYFRA:")
        metrics_table = []
        for digit, metrics in sorted(evaluation_results['per_class_metrics'].items()):
            metrics_table.append({
                'Cyfra': digit,
                'Precyzja': f"{metrics['precision']:.3f}",
                'Czułość': f"{metrics['recall']:.3f}",
                'F1-score': f"{metrics['f1_score']:.3f}",
                'Próbki': metrics['support']
            })
        print(tabulate(metrics_table, headers="keys", tablefmt="grid"))
    else:
        print("Ewaluacja nie powiodła się!")


##############   MAIN FUNCTION    ##############
def main():
    while True:
        print("=== SYSTEM ROZPOZNAWANIA CYFR - ROZWÓJ ===\n")
        print("Wybierz etap rozwoju:")
        print("1. PRZYGOTOWANIE DANYCH")
        print("   - Ekstrakcja cech MFCC z datasetu")
        print("   - Walidacja jakości danych")
        print()
        print("2. EKSPERYMENTY I OPTYMALIZACJA")
        print("   a) Szybki prototyp (domyślne parametry)")
        print("   b) Optymalizacja parametrów MFCC i GMM")
        print("   c) Testy krzyżowe i walidacja")
        print()
        print("3. TRENING KOŃCOWEGO MODELU")
        print("   - Trening na pełnym datasetcie")
        print("   - Zapis ostatecznego klasyfikatora")
        print()
        print("4. EWALUACJA SYSTEMU")
        print("   - Testowanie końcowej skuteczności")
        print("   - Metryki jakości klasyfikacji")
        print()
        print("5. WYJŚCIE")

        try:
            answer = input("Twój wybór: ").strip()
            
            if answer == "1":
                prepare_data_stage()
                continue
                
            elif answer == "2":
                print("Wybierz tryb eksperymentów:")
                print("a) Szybki prototyp")
                print("b) Optymalizacja parametrów")  
                print("c) Testy krzyżowe")
                
                sub_choice = input("Twój wybór (a/b/c): ").lower()
                
                if sub_choice == "a":
                    quick_prototype()
                    
                elif sub_choice == "b":
                    optimize_parameters_stage()
                    
                elif sub_choice == "c":
                    cross_validation_stage()
                    
                else:
                    print("Nieprawidłowy wybór")
                    continue
                continue

            elif answer == "3":
                train_final_model()
                continue
                
            elif answer == "4":
                evaluate_system_stage()
                continue
                
            elif answer == "5":
                print("Zakończono program.")
                sys.exit(0)
                break
                
            else:
                print("Wybrano niepoprawną wartość.")
                continue

        except Exception as e:
            print(f"Wystąpił nieoczekiwany błąd: {e}")
            import traceback
            traceback.print_exc()


SCOPES = "https://www.googleapis.com/auth/drive.file"

if __name__ == "__main__":
    main()