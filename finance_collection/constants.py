# General time features
trd_days_year = 252
trd_days_biannual = 126
trd_days_quarter = 63
trd_days_month = 21
trd_days_week = 5
trd_days_day = 1
short_ma_window = (8, 24)
mid_ma_window = (16, 48)
long_ma_window = (32, 96)

# Financial values
HALFLIFE_WINSORISE = 252
BACKTEST_AVERAGE_BASIS_POINTS = [None, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# Statistical values
VOL_THRESHOLD = 5  # multiple to winsorise by
VOL_LOOKBACK = 60  # for ex-ante volatility
VOL_TARGET = 0.15  # 15% volatility target

# Hyperparameter
HP_MINIBATCH_SIZE = [64, 128, 256]
HP_LEARNING_RATE = [1e-4, 1e-3, 1e-2, 1e-1]
HP_HIDDEN_LAYER_SIZE = [5, 10, 20, 40, 80, 160]
HP_DROPOUT_RATE = [0.1, 0.2, 0.3, 0.4, 0.5]
HP_MAX_GRADIENT_NORM = [0.01, 1.0, 100.0]

# Data quality
MAX_NO_VOL_SEQ_LENGTH = 3  # maximum number of consecutive days with no volume
ASSET_DATA_QUALITY_THRESHOLD = 0.2  # threshold for data quality (percentage of max dates that can be removed before dismissing asset)
