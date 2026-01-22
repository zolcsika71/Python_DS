import pandas as pd
import numpy as np
import os
import sys

# Add project root to path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CONFIG, logger

def process_top_targets(input_file: str, output_file: str, top_n: int = 5):
    """
    Loads a CSV file, identifies the top N TARGET values, 
    sets them to 1.0, and saves the result.
    """
    logger.info(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    top_n_indices = df.nlargest(top_n, 'TARGET').index
    original_top_values = df.loc[top_n_indices, 'TARGET'].tolist()
    original_ids = df.loc[top_n_indices, 'SK_ID_CURR'].tolist()
    
    logger.info(f"Top {top_n} TARGET values identified: {original_top_values}")
    
    df.loc[top_n_indices, 'TARGET'] = 1.0
    
    logger.info(f"Saving processed data to {output_file}...")
    df.to_csv(output_file, index=False)
    
    summary = [
        f"ID {original_ids[i]}: Original TARGET {original_top_values[i]:.6f} -> New TARGET 1.0"
        for i in range(len(original_ids))
    ]
    
    return summary

if __name__ == "__main__":
    import glob
    # Automatically find the latest submission
    submission_pattern = os.path.join(CONFIG.paths.submissions_dir, "submission_*.csv")
    submission_files = sorted(glob.glob(submission_pattern))
    
    if submission_files:
        input_path = submission_files[-1]
        output_path = os.path.join(CONFIG.paths.submissions_dir, "processed_top_5_submission.csv")
        
        changes = process_top_targets(input_path, output_path)
        
        print(f"\nSummary of Modifications (from {os.path.basename(input_path)}):")
        for change in changes:
            print(f"- {change}")
        
        final_df = pd.read_csv(output_path)
        print(f"\nVerification - Top 5 values in {output_path}:")
        print(final_df.nlargest(5, 'TARGET')[['SK_ID_CURR', 'TARGET']])
    else:
        logger.error("No submission files found in submissions directory.")
