import os
import random
import numpy as np
import librosa
from collections import defaultdict
from pathlib import Path
from tabulate import tabulate

# Custom imports
from config import DEFAULT_MFCC_PARAMS, DEFAULT_GMM_PARAMS
from decorators import safe_execution
from utils import load_params_from_csv, save_pickle, load_pickle

# --- Ekstrakcja cech MFCC ---

@safe_execution("Błąd podczas obliczania MFCC")
def get_mfcc(x, fs, n_mfcc, n_fft, win_length, hop_length, n_mels, count_delta, count_delta_delta):
    """Calculates MFCC coefficients with optional deltas."""
    
    mfcc = librosa.feature.mfcc(
        y=x, sr=fs, n_mfcc=n_mfcc, n_fft=n_fft,
        win_length=win_length, hop_length=hop_length, n_mels=n_mels
    ).T

    # Remove the first coefficient (energy) usually index 0
    mfcc = mfcc[:, 1:]
    
    feature_list = [mfcc]
    
    if count_delta:
        delta = librosa.feature.delta(mfcc, axis=0)
        feature_list.append(delta)
        
    if count_delta_delta:
        delta_delta = librosa.feature.delta(mfcc, axis=0, order=2)
        feature_list.append(delta_delta)
    
    # Concatenate features if we have more than just base MFCC
    if len(feature_list) > 1:
        return np.concatenate(feature_list, axis=1)
    
    return mfcc

# --- Wczytywanie parametrów ---

def _print_params(title: str, params: dict):
    """Helper to visualize parameters."""
    print(f"{title}:")
    table_data = [{"parametr": k, "wartość": v} for k, v in params.items()]
    print(tabulate(table_data, headers="keys", tablefmt="grid"))

def load_mfcc_params():
    """Loads MFCC params from CSV or defaults."""
    base_dir = Path(__file__).parent
    params = load_params_from_csv(base_dir / "mfcc_params.csv")
    
    if params:
        _print_params("Parametry MFCC (z pliku)", params)
        return params

    print("Używam domyślnych parametrów MFCC")
    _print_params("Parametry MFCC (domyślne)", DEFAULT_MFCC_PARAMS)
    return DEFAULT_MFCC_PARAMS.copy()

def load_gmm_params():
    """Loads GMM params from CSV or defaults."""
    base_dir = Path(__file__).parent
    params = load_params_from_csv(base_dir / "gmm_params.csv")
    
    if params:
        _print_params("Parametry GMM (z pliku)", params)
        return params

    print("Używam domyślnych parametrów GMM")
    _print_params("Parametry GMM (domyślne)", DEFAULT_GMM_PARAMS)
    return DEFAULT_GMM_PARAMS.copy()


# --- Wszytywanie danych ---

def load_train_files_and_determine_mfcc(mfcc_params, show_table=False):
    """Loads WAV files and computes MFCCs."""
    train_dir = Path("train_data")
    
    if not train_dir.exists():
        print(f"Brak folderu {train_dir}!")
        return None
        
    wav_files = sorted(list(train_dir.glob("*.wav")))
    
    if not wav_files:
        print("Nie znaleziono plików WAV w folderze train_data!")
        return None

    print(f"Znaleziono {len(wav_files)} plików audio.")
    sound_data = defaultdict(list)

    for file_path in wav_files:
        try:
            x, fs = librosa.load(file_path, sr=16000, mono=True)
            mfcc = get_mfcc(x, fs, **mfcc_params)
        except Exception as e:
            print(f"Błąd przetwarzania pliku {file_path.name}: {e}")
            mfcc = None

        # Logic assuming filename format 'XX..._Y...' where XX is speaker and Y is digit
        filename = file_path.name
        speaker_id = filename[0:2]
        digit = filename[6]

        sound_data[speaker_id].append({
            "MFCC": mfcc,
            "num": digit,
            "filename": filename
        })
    
    print("Wczytano pliki i wyznaczono z nich macierze MFCC.")

    if show_table:
        _display_mfcc_summary(sound_data)
    
    return sound_data

def _display_mfcc_summary(sound_data):
    """Helper to display the loading summary table."""
    rows = []
    for spk, items in sound_data.items():
        for d in items:
            mfcc_info = f"shape={d['MFCC'].shape}" if isinstance(d.get("MFCC"), np.ndarray) else "None"
            rows.append({
                "speaker": spk,
                "num": d["num"],
                "filename": d["filename"],
                "MFCC": mfcc_info
            })
    print(tabulate(rows, headers="keys", tablefmt="grid"))

def prepare_training_data(sound_data, show_table=False):
    """Converts dictionary of samples into concatenated arrays per digit."""
    training_data = defaultdict(list)

    for speaker, samples in sound_data.items():
        for s in samples:
            mfcc = s.get("MFCC")
            if isinstance(mfcc, np.ndarray):
                training_data[s["num"]].append(mfcc)

    # Stack arrays vertically
    final_data = {}
    summary = []
    
    for label, arrays in training_data.items():
        if arrays:
            stacked_data = np.vstack(arrays)
            final_data[label] = stacked_data
            summary.append({
                "cyfra": label, 
                "liczba ramek": stacked_data.shape[0], 
                "liczba cech": stacked_data.shape[1]
            })

    if show_table:
        print("Przygotowane dane do treningu GMM:")
        print(tabulate(summary, headers="keys", tablefmt="grid"))

    return final_data

def validate_data_quality(dataset):
    """Analyzes data quality statistics."""
    stats = {
        'total_samples': 0,
        'missing_mfcc': 0,
        'samples_per_digit': defaultdict(int),
        'total_frames_per_digit': defaultdict(int), # Changed to sum for correct avg calculation
        'samples_per_speaker': defaultdict(int)
    }

    for speaker, samples in dataset.items():
        for sample in samples:
            mfcc = sample.get("MFCC")
            
            if not isinstance(mfcc, np.ndarray):
                stats['missing_mfcc'] += 1
                continue

            digit = sample.get("num")
            frames = mfcc.shape[0]

            stats['total_samples'] += 1
            stats['samples_per_speaker'][speaker] += 1
            stats['samples_per_digit'][digit] += 1
            stats['total_frames_per_digit'][digit] += frames

    # Print Report
    digits_present = len(stats['samples_per_digit'])
    speakers_present = len(stats['samples_per_speaker'])
    
    print("\n=== RAPORT JAKOŚCI DANYCH ===")
    print(f"Łączna liczba próbek: {stats['total_samples']}")
    print(f"Cyfry z danymi: {digits_present}/10")
    print(f"Mówcy z danymi: {speakers_present}/{len(dataset)}")
    print(f"Błędne próbki MFCC: {stats['missing_mfcc']}")
    
    print("\nPRÓBKI PER CYFRA:")
    for digit in sorted(stats['samples_per_digit'].keys()):
        count = stats['samples_per_digit'][digit]
        total_frames = stats['total_frames_per_digit'][digit]
        avg_frames = total_frames / count if count > 0 else 0
        print(f"  Cyfra {digit}: {count} próbek, średnio {avg_frames:.1f} ramek")

    return stats


def split_train_test_by_speaker(sound_data, test_fraction=0.2, seed=None):
    """Splits speakers into train and test sets."""
    if seed is not None:
        random.seed(seed)

    speakers = list(sound_data.keys())
    if not speakers:
        print("Brak mówców do podziału!")
        return [], []

    num_test = max(1, int(len(speakers) * test_fraction))
    test_speakers = random.sample(speakers, num_test)
    train_speakers = [s for s in speakers if s not in test_speakers]

    print(f"Mówcy do treningu: {train_speakers}")
    print(f"Mówcy do testu: {test_speakers}")

    return train_speakers, test_speakers