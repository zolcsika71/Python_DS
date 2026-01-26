import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Any

# Logging Configuration
class ColorFormatter(logging.Formatter):
    """
    Custom logging formatter that applies ANSI color codes based on log level and content.
    
    Color Hierarchy (PEP 8 & Industry Aligned):
    - INFO: Green (Status updates, successful operations)
    - WARNING: Bold Yellow (Standard system alerts)
    - WARNING (Data Drift): Bold Blue (Domain-specific statistical alerts)
    - ERROR/CRITICAL: Bold Red (Execution-blocking failures)
    """
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    def format(self, record):
        """
        Formats the log record with colors. 
        Note: We use Bold for high-severity alerts to improve accessibility for color-blind users.
        """
        formatted_msg = super().format(record)
        if record.levelno == logging.INFO:
            return f"{self.GREEN}{formatted_msg}{self.RESET}"
        elif record.levelno == logging.WARNING:
            # Domain-specific logic: Data drift alerts are colored Blue to distinguish them
            # from standard system warnings, making them easier to scan in long logs.
            msg_lower = str(record.msg).lower()
            if "data drift" in msg_lower or str(record.msg).strip().startswith("- "):
                return f"{self.BOLD}{self.BLUE}{formatted_msg}{self.RESET}"
            # Standard warnings use Bold Yellow for high visibility
            return f"{self.BOLD}{self.YELLOW}{formatted_msg}{self.RESET}"
        elif record.levelno >= logging.ERROR:
            # Errors and Critical issues use Bold Red to signal immediate attention required
            return f"{self.BOLD}{self.RED}{formatted_msg}{self.RESET}"
        return formatted_msg

def setup_logging():
    """
    Initializes the root logger with our custom ColorFormatter.
    """
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
        "n_estimators": 2000,
        "learning_rate": 0.02,
        "num_leaves": 34,
        "colsample_bytree": 0.9497036,
        "subsample": 0.8715623,
        "max_depth": 8,
        "reg_alpha": 0.041545473,
        "reg_lambda": 0.0735294,
        "min_split_gain": 0.0222415,
        "min_child_weight": 39.3259775,
        "n_jobs": -1,
        "random_state": 42,
        "importance_type": 'gain',
        "verbosity": -1,
    })
    
    logreg_params: Dict[str, Any] = field(default_factory=lambda: {
        "solver": "lbfgs",
        "max_iter": 1000,
        "n_jobs": -1,
        "random_state": 42,
    })
    
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": -1,
        "random_state": 42,
        "eval_metric": "auc",
    })
    
    cat_params: Dict[str, Any] = field(default_factory=lambda: {
        "iterations": 1000,
        "learning_rate": 0.05,
        "depth": 6,
        "random_seed": 42,
        "verbose": False,
        "thread_count": -1,
    })

CONFIG = ModelConfig()

def setup_directories(config: ModelConfig = CONFIG):
    """Ensures that the necessary directories exist."""
    for directory in [config.paths.submissions_dir, config.paths.plots_dir]:
        os.makedirs(directory, exist_ok=True)
