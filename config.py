import os

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
    'num_components': 8,
    'cov_type': 'diag',
    'max_iter': 200,
    'random_state': 42
}

OUTPUT_DIR = "output"
CSV_DIR = os.path.join(OUTPUT_DIR, "csv_results")
PKL_DIR = os.path.join(OUTPUT_DIR, "pkl_models")

DEFAULT_MODEL_FILENAME = os.path.join(PKL_DIR, "gmm_models.pkl")
FINAL_MODEL_FILENAME = os.path.join(PKL_DIR, "final_classifier.pkl")
PROCESSED_DATA_FILE = os.path.join(PKL_DIR, "dataset_processed.pkl")
PROTOTYPE_FILE = os.path.join(PKL_DIR, "prototype_models.pkl")


BEST_MFCC_PARAMS_FILE = os.path.join(CSV_DIR, "best_mfcc_params.csv")
BEST_GMM_PARAMS_FILE = os.path.join(CSV_DIR, "best_gmm_params.csv")
OPTIMIZATION_RESULTS_FILE = os.path.join(CSV_DIR, "optimization_results.csv")

TRAIN_DATA_FOLDER_ID = "1WQVB4mqdNBSvpa1SZ8EbUc5eJ--e1t6y"
EVAL_FOLDER_ID = "1Ycuxo3aCw4yAXQBNPmf9cYcgNJRoTrGE"
EVAL_IA_FOLDER_ID = "1pivFMeM-_zlYEn_a8bTNv-wJfVM8LIDQ"