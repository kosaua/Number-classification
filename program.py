import librosa
import numpy as np
from collections import defaultdict
from tabulate import tabulate
from os import listdir
from os.path import isfile
import csv
import os


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


def loadTrainFilesAndMFCC(mfcc_params, showTable):


    wavpaths = [f for f in listdir(".") if isfile(f) and f.endswith('.wav')]
    print(wavpaths)
    wavpaths.sort()

    sound_data = defaultdict(list)

    print(f"Znaleziono {len(wavpaths)} plików audio.")

    for f in wavpaths:
        x, fs = librosa.load(f, sr=16000, mono=True)
        try:
            mfcc = get_mfcc(x, fs, **mfcc_params)
        except Exception as e:
            print(f"Błąd MFCC dla {f}: {e}")
            mfcc = None

        sound_data[f[0:2]].append({
            "MFCC": mfcc,
            "num": f[6],
            "filename": f
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

##############   MAIN FUNCTION    ##############

mfcc_params = load_mfcc_params()
sound_data = loadTrainFilesAndMFCC(mfcc_params, True);
training_data = prepare_training_data(sound_data, True)