#%%
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
import unittest
import numpy as np
from strategy_class import NaiveSingleMonitorStrategy

#%%


# %%
class TestNaiveSingleMonitorStrategy(unittest.TestCase):
    def setUp(self):
        pass

    def test_fit_and_metrics(self):
        # Test fit method
        strategy = NaiveSingleMonitorStrategy(alpha=True)
        strategy.fit_method(
            in_sample_alpha_benign=np.array([1, 2, 3, 4, 5]),
            in_sample_alpha_attack=np.array([4, 5, 6, 7, 8]),
            in_sample_beta_benign=np.array([1, 2, 3, 4, 5]),
            in_sample_beta_attack=np.array([3, 4, 5, 6, 7]),
            Y=3,
            c_m=2,
            c_n=1,
            c_a=4
        )
        self.assertEqual(strategy.a_m, 5, "a_m should be 5")
        self.assertEqual(strategy.audit_fpr, 0.2, "audit_fpr should be 0.2")
        self.assertEqual(strategy.audit_budget_share, 0.2*4/3, "audit_budget_share")
        self.assertEqual(strategy.in_sample_safety, 0.8, "in_sample_safety should be 0.8")
        self.assertEqual(strategy.get_oos_safety(np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 0])),0.2, "oos safety should be 0.2")

        strategy.reset()
        self.assertEqual(strategy.a_m, None, "a_m should be None")
        strategy.alpha = False
        strategy.fit_method(
            in_sample_alpha_benign=np.array([1, 2, 3, 4, 5]),
            in_sample_alpha_attack=np.array([4, 5, 6, 7, 8]),
            in_sample_beta_benign=np.array([1, 2, 3, 4, 5]),
            in_sample_beta_attack=np.array([3, 4, 5, 6, 7]),
            Y=3,
            c_m=2,
            c_n=1,
            c_a=5
        )
        self.assertEqual(strategy.a_m, 4, "a_m should be 5")
        self.assertEqual(strategy.audit_fpr, 0.4, "audit_fpr should be 0.4")
        self.assertEqual(strategy.audit_budget_share, 0.4*5/3, "audit_budget_share")
        self.assertEqual(strategy.in_sample_safety, 0.8, "in_sample_safety should be 0.8")
        self.assertEqual(strategy.get_oos_safety(np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 0])),0.2, "oos safety should be 0.4")

def run_tests():
    """
    Runs all tests in the TestNewFrameworkWebapp test suite.
    Can be called from another script or used as a standalone test runner.
    """
    suite = unittest.TestLoader().loadTestsFromTestCase(TestNaiveSingleMonitorStrategy)
    unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == '__main__':
    run_tests()
