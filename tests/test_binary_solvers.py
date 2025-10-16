#%%
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
import unittest
import numpy as np
import binary_monitors as bm

#Some of these tests might not pass anymore but that should be ok
#%%
class TestBinarySolvers(unittest.TestCase):
    def setUp(self):
        self.alpha_attack_binary = np.array([1,1,1,1,0])
        self.beta_attack_binary = np.array( [0,0,0,1,1])
        self.alpha_benign_binary = np.array([0,0,0,1,0])
        self.beta_benign_binary = np.array([ 0,0,0,0,1])

    def test_calculate_prob_from_thresholds(self):
        
        probs = bm.calculate_prob_from_thresholds(
            attack_beta_binary=self.beta_attack_binary,
            attack_alpha_binary=self.alpha_attack_binary,
            honest_beta_binary=self.beta_benign_binary,
            honest_alpha_binary=self.alpha_benign_binary
        )
        self.assertAlmostEqual(probs["p_m_given_q"], 0.8, msg = "p_m_given_q")
        self.assertAlmostEqual(probs["p_m_given_not_q"], 0.2, msg = "p_m_given_not_q")
        self.assertAlmostEqual(probs["p_n_given_q"], 0.4, msg = "p_n_given_q")
        self.assertAlmostEqual(probs["p_n_given_not_q"], 0.2, msg = "p_n_given_not_q")
        self.assertAlmostEqual(probs["p_m1n0_given_q"], 0.6, msg = "p_m1n0_given_q")
        self.assertAlmostEqual(probs["p_m0n1_given_q"], 0.2, msg = "p_m0n1_given_q")
        self.assertAlmostEqual(probs["p_m0n0_given_q"], 0.0, msg = "p_m0n0_given_q")
        self.assertAlmostEqual(probs["p_m1n1_given_q"], 0.2, msg = "p_m1n1_given_q")
        self.assertAlmostEqual(probs["p_m1n0_given_not_q"], 0.2, msg = "p_m1n0_given_not_q")
        self.assertAlmostEqual(probs["p_m0n1_given_not_q"], 0.2, msg = "p_m0n1_given_not_q")
        self.assertAlmostEqual(probs["p_m0n0_given_not_q"], 0.6, msg = "p_m0n0_given_not_q")
        self.assertAlmostEqual(probs["p_m1n1_given_not_q"], 0.0, msg = "p_m1n1_given_not_q")

    def test_calculate_prob_attack_only(self):
        probs = bm.calculate_prob_attack_only(
            attack_beta_binary=self.beta_attack_binary,
            attack_alpha_binary=self.alpha_attack_binary
        )
        self.assertAlmostEqual(probs["p_m_given_q"], 0.8, msg = "p_m_given_q_attack_only")
        self.assertAlmostEqual(probs["p_n_given_q"], 0.4, msg = "p_n_given_q_attack_only")
        self.assertAlmostEqual(probs["p_m1n0_given_q"], 0.6, msg = "p_m1n0_given_q_attack_only")
        self.assertAlmostEqual(probs["p_m0n1_given_q"], 0.2, msg = "p_m0n1_given_q_attack_only")
        self.assertAlmostEqual(probs["p_m0n0_given_q"], 0.0, msg = "p_m0n0_given_q_attack_only")
        self.assertAlmostEqual(probs["p_m1n1_given_q"], 0.2, msg = "p_m1n1_given_q_attack_only")
        
        

    def test_oos_safety_binary(self):
        oos_safety = bm.oos_safety_binary(
            alpha_cutoff=0.5,
            beta_cutoff=0.5,
            x=np.array([0,0,0,0,0,0.5,0,0,1,0,1,0,0]),
            oos_alpha_attack=self.alpha_attack_binary,
            oos_beta_attack=self.beta_attack_binary,
        )
        self.assertAlmostEqual(oos_safety, 0.8, msg = "oos_safety")


def run_tests():
    """
    Runs all tests in the TestNewFrameworkWebapp test suite.
    Can be called from another script or used as a standalone test runner.
    """
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBinarySolvers)
    unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == '__main__':
    run_tests()