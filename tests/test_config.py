import os
import sys
import unittest
import tempfile
import shutil

# Add project root to path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import setup_directories, ModelConfig, PathConfig

class TestConfig(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for tests
        self.test_dir = tempfile.mkdtemp()
        self.submissions_dir = os.path.join(self.test_dir, "submissions")
        self.plots_dir = os.path.join(self.test_dir, "plots")
        
        # Create a custom config for testing
        self.path_config = PathConfig(
            data_dir=os.path.join(self.test_dir, "data"),
            submissions_dir=self.submissions_dir,
            plots_dir=self.plots_dir
        )
        self.model_config = ModelConfig(paths=self.path_config)

    def tearDown(self):
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)

    def test_setup_directories(self):
        # Ensure directories don't exist before setup
        self.assertFalse(os.path.exists(self.submissions_dir))
        self.assertFalse(os.path.exists(self.plots_dir))
        
        # Run setup
        setup_directories(self.model_config)
        
        # Ensure directories exist after setup
        self.assertTrue(os.path.exists(self.submissions_dir))
        self.assertTrue(os.path.exists(self.plots_dir))

    def test_config_defaults(self):
        config = ModelConfig()
        self.assertEqual(config.target_col, "TARGET")
        self.assertEqual(config.id_col, "SK_ID_CURR")
        self.assertIn("n_estimators", config.lgbm_params)
        self.assertIn("max_iter", config.logreg_params)

if __name__ == "__main__":
    unittest.main()
