import logging
import os

# Logging Configuration
class ColorFormatter(logging.Formatter):
    GREEN = "\033[92m"
    RESET = "\033[0m"

    def format(self, record):
        if record.levelno == logging.INFO:
            # Color the whole line green
            formatted_msg = super().format(record)
            return f"{self.GREEN}{formatted_msg}{self.RESET}"
        return super().format(record)

handler = logging.StreamHandler()
handler.setFormatter(ColorFormatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))

logging.basicConfig(
    level=logging.INFO,
    handlers=[handler]
)
logger = logging.getLogger(__name__)

# Directory Configurations
DATA_DIR = "data"
SUBMISSIONS_DIR = "submissions"
PLOTS_DIR = "plots"

# Dataset Constants
TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"

# Model Hyperparameters (Defaults)
LGBM_PARAMS = {
    "n_estimators": 800,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "random_state": 42,
    "importance_type": 'gain',
}

LOGREG_PARAMS = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "n_jobs": -1,
    "random_state": 42,
}

def setup_directories():
    """Ensures that necessary directories exist."""
    for directory in [SUBMISSIONS_DIR, PLOTS_DIR]:
        os.makedirs(directory, exist_ok=True)
