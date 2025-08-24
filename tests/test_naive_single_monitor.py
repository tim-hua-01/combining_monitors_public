import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
import unittest
import numpy as np
import single_monitor as sm

class TestNaiveSingleMonitor(unittest.TestCase):
    def setUp(self):
        pass

    def test_get_naive_single_monitor_policy(self):
        honest_scores = np.array([1, 2, 3, 4, 5])
        Y = 3
        c_a = 5
        c_m = 1
        naive_threshold = sm.get_naive_single_monitor_policy(honest_scores, Y, c_a, c_m)
        self.assertEqual(naive_threshold, 4)

def run_tests():
    """
    Runs all tests in the TestNewFrameworkWebapp test suite.
    Can be called from another script or used as a standalone test runner.
    """
    suite = unittest.TestLoader().loadTestsFromTestCase(TestNaiveSingleMonitor)
    unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == '__main__':
    run_tests() 
