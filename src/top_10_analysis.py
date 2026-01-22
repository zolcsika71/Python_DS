import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Add project root to path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CONFIG, logger
from src.orchestrator import analyze_top_10_targets

if __name__ == "__main__":
    import glob
    # Automatically find the latest submission
    submission_pattern = os.path.join(CONFIG.paths.submissions_dir, "submission_*.csv")
    submission_files = sorted(glob.glob(submission_pattern))
    
    if submission_files:
        latest_submission = submission_files[-1]
        logger.info(f"Analyzing latest submission: {os.path.basename(latest_submission)}")
        df = pd.read_csv(latest_submission)
        analyze_top_10_targets(df, CONFIG)
    else:
        logger.error("No submission files found in submissions directory.")
