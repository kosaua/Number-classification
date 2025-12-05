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

DEFAULT_MODEL_FILENAME = "gmm_models.pkl"
FINAL_MODEL_FILENAME = "final_classifier.pkl"

PROCESSED_DATA_FILE = "dataset_processed.pkl"
BEST_MFCC_PARAMS_FILE = "best_mfcc_params.csv"
BEST_GMM_PARAMS_FILE = "best_gmm_params.csv"
OPTIMIZATION_RESULTS_FILE = "optimization_results.csv"

TRAIN_DATA_FOLDER_ID = "1WQVB4mqdNBSvpa1SZ8EbUc5eJ--e1t6y"
EVAL_FOLDER_ID = "1Ycuxo3aCw4yAXQBNPmf9cYcgNJRoTrGE"
EVAL_IA_FOLDER_ID = "1pivFMeM-_zlYEn_a8bTNv-wJfVM8LIDQ"