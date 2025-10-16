#%%
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
import unittest
import numpy as np
import single_monitor as sms

#Some of these tests might not pass anymore but that should be ok
#%%
class TestSingleMonitorSolver(unittest.TestCase):
    def setUp(self):
        pass

    def test_single_monitor_solver(self):
        c_a = 2
        c_m = 0.5
        Y = 1.5
        attack_scores = np.array([0,3,6,7,8])
        honest_scores = np.array([0,1,2,3,5,10])
        obj, a_m, p_m, a_s = sms.solve_1mon_optimal_policy(c_a, c_m, Y, attack_scores, honest_scores, n_ptile_points=20)
        self.assertAlmostEqual(obj, 0.8, msg = "objective")
        self.assertTrue(a_m > 2 and a_m <= 3, msg = "a_m")
        self.assertAlmostEqual(p_m, 1, msg = "p_m")
        self.assertAlmostEqual(a_s, 0, msg = "a_s")
        self.assertTrue(c_a*a_s + p_m*(c_m + np.mean(honest_scores >= a_m)*c_a)<= 1.5, msg = "budget")
 
def run_tests():
    """
    Runs all tests in the TestNewFrameworkWebapp test suite.
    Can be called from another script or used as a standalone test runner.
    """
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSingleMonitorSolver)
    unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == '__main__':
    run_tests()