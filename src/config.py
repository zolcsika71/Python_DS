import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Any

# Logging Configuration
class ColorFormatter(logging.Formatter):
    GREEN = "\033[92m"
    RESET = "\033[0m"

    def format(self, record):
        if record.levelno == logging.INFO:
            formatted_msg = super().format(record)
            return f"{self.GREEN}{formatted_msg}{self.RESET}"
        return super().format(record)

def setup_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(handler)
    return logging.getLogger("src")

logger = setup_logging()

# Directory Configurations
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SUBMISSIONS_DIR = os.path.join(BASE_DIR, "submissions")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# Dataset Constants
TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"

@dataclass(frozen=True)
class PathConfig:
    data_dir: str = DATA_DIR
    submissions_dir: str = SUBMISSIONS_DIR
    plots_dir: str = PLOTS_DIR

@dataclass(frozen=True)
class ModelConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    target_col: str = TARGET_COL
    id_col: str = ID_COL
    lgbm_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 800,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "n_jobs": -1,
        "random_state": 42,
        "importance_type": 'gain',
    })
    
    logreg_params: Dict[str, Any] = field(default_factory=lambda: {
        "solver": "lbfgs",
        "max_iter": 1000,
        "n_jobs": -1,
        "random_state": 42,
    })

CONFIG = ModelConfig()

def setup_directories(config: ModelConfig = CONFIG):
    """Ensures that the necessary directories exist."""
    for directory in [config.paths.submissions_dir, config.paths.plots_dir]:
        os.makedirs(directory, exist_ok=True)
