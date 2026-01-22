import os
import sys
import unittest
import tempfile
import shutil
import pandas as pd
import numpy as np

# Add project root to path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.orchestrator import analyze_top_10_targets
from src.config import ModelConfig, PathConfig

class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.submissions_dir = os.path.join(self.test_dir, "submissions")
        self.plots_dir = os.path.join(self.test_dir, "plots")
        os.makedirs(self.submissions_dir)
        os.makedirs(self.plots_dir)
        
        self.path_config = PathConfig(
            data_dir=os.path.join(self.test_dir, "data"),
            submissions_dir=self.submissions_dir,
            plots_dir=self.plots_dir
        )
        self.model_config = ModelConfig(paths=self.path_config)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_analyze_top_10_targets(self):
        # Create a mock submission dataframe
        data = {
            'SK_ID_CURR': range(100, 120),
            'TARGET': np.linspace(0, 0.95, 20)
        }
        df = pd.DataFrame(data)
        
        top_10 = analyze_top_10_targets(df, self.model_config)
        
        # Verify result size
        self.assertEqual(len(top_10), 10)
        
        # Verify files are saved
        self.assertTrue(os.path.exists(os.path.join(self.plots_dir, "top_10_targets_closest_to_1.png")))
        self.assertTrue(os.path.exists(os.path.join(self.submissions_dir, "top_10_closest_targets.csv")))
        
        # Verify the top values are indeed the largest (closest to 1)
        self.assertAlmostEqual(top_10['TARGET'].max(), 0.95)

if __name__ == "__main__":
    unittest.main()
