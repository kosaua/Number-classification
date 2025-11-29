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