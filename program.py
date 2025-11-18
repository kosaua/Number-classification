from ntpath import join
import sys
import librosa
import numpy as np
from sklearn.mixture import GaussianMixture
from collections import defaultdict
from tabulate import tabulate
from os import listdir
from os.path import isfile
import csv
import os
import pickle
import random


def get_mfcc(x, fs, n_mfcc, n_fft, win_length, hop_length, n_mels, count_delta, count_delta_delta):
    try:
        mfcc = librosa.feature.mfcc(
            y=x, sr=fs, n_mfcc=n_mfcc, n_fft=n_fft,
            win_length=win_length, hop_length=hop_length, n_mels=n_mels
        ).T

        counted_delta = 0

        first_feature_id = 1
        mfcc = mfcc[:, first_feature_id:] #DODANIE OBSŁUGI FIRST FEATURE W PÓŹNIEJSZYM ETAPIE
        if count_delta == True and count_delta_delta==False:
            deltas = librosa.feature.delta(mfcc.T).T
            mfcc = np.concatenate((mfcc, deltas), axis=1)
            # print("Obliczono delte")
            counted_delta = 1


        if count_delta == True and count_delta_delta==True:   #wartości delty-delty nie liczą się poprawnie
            delta_deltas = librosa.feature.delta(mfcc.T, order=2).T
            mfcc = np.concatenate((mfcc, delta_deltas), axis=1)
            # print("Obliczono delte-delte")
            counted_delta = 2     #DODANIE OBSŁUGI WYŚWIETLANIA INFO O DELTACH W PÓŹNIEJSZYM ETAPIE

    except Exception as e:
        print(f"Błąd podczas obliczania MFCC: {e}")
        return None

    return mfcc


def load_mfcc_params():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    editable_path = os.path.join(base_dir, "mfcc_params.csv")
    default_path = os.path.join(base_dir, "default_mfcc_params.csv")

    # Funkcja konwertująca wartości CSV
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

    # wybieramy które CSV otwieramy
    path = editable_path if os.path.exists(editable_path) else default_path

    # wczytujemy CSV jako słownik parametrów
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        params = {k: parse_value(v) for k, v in next(reader).items()}

    # wypisanie tabeli
    print("Parametry MFCC:")
    table_data = [{"parametr": k, "wartość": v} for k, v in params.items()]
    print(tabulate(table_data, headers="keys", tablefmt="grid"))
    print(f"Źródło parametrów: {os.path.basename(path)}")

    return params


def loadTrainFilesAndDetermineMFCC(mfcc_params, showTable):
    train_data_dir = "train_data"
    short_wavpaths = [f for f in listdir(train_data_dir) if isfile(join(train_data_dir, f)) and f.endswith('.wav')]

    wavpaths = [join(train_data_dir, f) for f in short_wavpaths]

    wavpaths.sort()
    short_wavpaths.sort()  # też posortuj dla spójności

    sound_data = defaultdict(list)

    print(f"Znaleziono {len(wavpaths)} plików audio.")

    for i, f in enumerate(wavpaths):
        file_name = short_wavpaths[i]  # odpowiadająca nazwa pliku bez ścieżki
        x, fs = librosa.load(f, sr=16000, mono=True)
        try:
            mfcc = get_mfcc(x, fs, **mfcc_params)
        except Exception as e:
            print(f"Błąd MFCC dla {f}: {e}")
            mfcc = None

        sound_data[file_name[0:2]].append({
            "MFCC": mfcc,
            "num": file_name[6],
            "filename": file_name
        })
    print(f"Wczytano pliki i wyznaczono z nich macierze MFCC.")

    # TABELA Z PODSUMOWANIEM
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
    
    # ZAPIS DO PLIKU PICKLE
    try:
        filename = "mfcc_data.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(sound_data, f)
        print(f"Dane MFCC zostały zapisane do pliku: {filename}")
    except Exception as e:
        print(f"Błąd podczas zapisywania danych do pliku: {e}")

    return sound_data
    


def prepare_training_data(sound_data, showTable):
    import numpy as np
    from collections import defaultdict

    training_data = defaultdict(list)

    for speaker, samples in sound_data.items():
        for s in samples:
            mfcc = s.get("MFCC")
            label = s.get("num")

            if mfcc is None or not isinstance(mfcc, np.ndarray):
                continue

            training_data[label].append(mfcc)

    # ŁĄCZMYMY DANE MFCC W MACIERZE DLA KOLEJNYCH CYFR
    for label in training_data:
        training_data[label] = np.vstack(training_data[label])

    # TABELA Z PODSUMOWANIEM
    if showTable==True:
      print("Przygotowane dane do treningu GMM:")
      summary = [
          {"cyfra": k, "liczba ramek": v.shape[0], "liczba cech": v.shape[1]}
          for k, v in training_data.items()
      ]
      from tabulate import tabulate
      print(tabulate(summary, headers="keys", tablefmt="grid"))

    print("Przygotowano dane do treningu GMM")
    return training_data


def loadMFFCData(show_table):
    """Wczytuje dane MFCC z pliku pickle"""
    try:
        # Szukanie plików .pkl w bieżącym katalogu
        pkl_files = [f for f in listdir(".") if isfile(f) and f.endswith('.pkl')]
        
        if not pkl_files:
            print("Nie znaleziono żadnych plików .pkl w bieżącym katalogu.")
            return None
        
        # Jeśli jest wiele plików, pozwól użytkownikowi wybrać
        if len(pkl_files) > 1:
            print("Znaleziono następujące pliki .pkl:")
            for i, filename in enumerate(pkl_files, 1):
                print(f"{i}. {filename}")
            
            try:
                choice = int(input("Wybierz numer pliku do wczytania: ")) - 1
                if choice < 0 or choice >= len(pkl_files):
                    print("Nieprawidłowy wybór.")
                    return None
                filename = pkl_files[choice]
            except ValueError:
                print("Nieprawidłowy wybór.")
                return None
        else:
            filename = pkl_files[0]
            print(f"Wczytuję plik: {filename}")

        # Wczytywanie danych
        with open(filename, 'rb') as f:
            sound_data = pickle.load(f)
        
        print(f"Pomyślnie wczytano dane MFCC z pliku {filename}")
             

        if show_table == True:
            # Wyświetlanie podsumowania wczytanych danych
            print("\nPodsumowanie wczytanych danych:")
            
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
        
    except FileNotFoundError:
        print(f"Plik {filename} nie został znaleziony.")
        return None
    except Exception as e:
        print(f"Błąd podczas wczytywania danych z pliku: {e}")
        return None
    


def split_train_test_by_speaker(sound_data, test_fraction=0.2, seed=None):

    if seed is not None:
        random.seed(seed)

    speakers = list(sound_data.keys())
    num_test = max(1, int(len(speakers) * test_fraction))
    test_speakers = random.sample(speakers, num_test)
    train_speakers = [s for s in speakers if s not in test_speakers]

    print(f"Mówcy do testu: {test_speakers}")

    return train_speakers, test_speakers



##############   MAIN FUNCTION    ##############
print("Wybierz opcję\n1.\twczytaj nowe nagrania do trenowania modelu\n2.\twczytaj wcześniej sparametryzowane pliki")
try:
    answer = int(input())
    if answer == 1:
        mfcc_params = load_mfcc_params()
        mfcc_data = loadTrainFilesAndDetermineMFCC(mfcc_params, False)
    elif answer == 2:
        mfcc_data = loadMFFCData(False)
        if mfcc_data is None:
            print("Nie udało się wczytać danych. Zakończono program.")
            sys.exit(0)
    else:
        print("Wybrano niepoprawną wartość.")
        sys.exit(0)

    # Kontynuuj przetwarzanie z mfcc_data
    if mfcc_data is not None:
        train_speakers, test_speakers = split_train_test_by_speaker(mfcc_data, test_fraction=0.2, seed=42)

        # Przygotowanie danych treningowych
        train_data = prepare_training_data({k: mfcc_data[k] for k in train_speakers}, showTable=True)

        # Przygotowanie danych testowych
        test_data = prepare_training_data({k: mfcc_data[k] for k in test_speakers}, showTable=True)
    else:
        print("Brak danych do przetworzenia.")
        
except ValueError:
    print("Nieprawidłowy wybór. Proszę wprowadzić liczbę 1 lub 2.")
    sys.exit(1)
except Exception as e:
    print(f"Wystąpił nieoczekiwany błąd: {e}")
    sys.exit(1)


# training_data = prepare_training_data(sound_data, True)


################ TRENING GMM ###################


def train_gmms(training_dict, num_components=8, cov_type='diag', max_iter=200):

    gmm_models = {}

    for digit, mfcc in training_dict.items():
        gmm = GaussianMixture(
            n_components=num_components,
            covariance_type=cov_type,
            max_iter=max_iter,
            verbose=1
        )

        gmm.fit(mfcc)
        gmm_models[digit] = gmm

    print("Trenowanie wszystkich modeli zakończone.")
    return gmm_models

models_dict = train_gmms(train_data, num_components=16)
