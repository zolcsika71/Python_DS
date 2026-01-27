import os
import sys
import unittest
import tempfile
import shutil
import numpy as np
from sklearn.pipeline import Pipeline

# Add project root to path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.visualization import (
    plot_prediction_distribution,
    plot_roc_curve,
    plot_feature_importance,
    plot_train_test_distribution
)

class TestVisualization(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_plot_prediction_distribution(self):
        probes = np.random.rand(100)
        out_path = os.path.join(self.test_dir, "dist.png")
        plot_prediction_distribution(probes, out_path)
        self.assertTrue(os.path.exists(out_path))

    def test_plot_roc_curve(self):
        y_true = np.array([0, 0, 1, 1])
        y_probes = np.array([0.1, 0.4, 0.35, 0.8])
        out_path = os.path.join(self.test_dir, "roc.png")
        plot_roc_curve(y_true, y_probes, out_path)
        self.assertTrue(os.path.exists(out_path))

    def test_plot_train_test_distribution(self):
        train_proba = np.random.rand(100)
        test_proba = np.random.rand(100)
        out_path = os.path.join(self.test_dir, "train_test_dist.png")
        plot_train_test_distribution(train_proba, test_proba, out_path)
        self.assertTrue(os.path.exists(out_path))

    def test_plot_feature_importance(self):
        # Mock a pipeline and model
        class MockModel:
            def __init__(self):
                self.feature_importances_ = np.array([0.1, 0.2, 0.7])
                self.n_features_in_ = 3
        
        class MockPrep:
            @staticmethod
            def get_feature_names_out():
                return np.array(['feat1', 'feat2', 'feat3'])
        
        clf = Pipeline([
            ('prep', MockPrep()),
            ('model', MockModel())
        ])
        
        plot_feature_importance(clf, top_n=3, out_dir=self.test_dir)
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "feature_importance.png")))

if __name__ == "__main__":
    unittest.main()
