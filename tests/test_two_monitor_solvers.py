import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
import unittest
import numpy as np
import two_monitors as tm

# Some of these tests might not pass anymore but that should be ok


class TestTwoMonitor(unittest.TestCase):
    def setUp(self):
        pass

    def test_p_a_given_q(self):
        attack_a = np.array([4.0, 5, 6, 7])
        attack_b = np.array([2.0, 3, 4, 5])

        def simple_fquad(a, b):
            return a

        x = np.array([0, 2, 3, 4, 4, -2, -2])
        catch_rate = tm.p_a_given_q(x, attack_a, attack_b, simple_fquad)
        self.assertAlmostEqual(catch_rate, 0.5, msg="p_N no defer")

        x = np.array([1, 6.1, 6.01, 4, 4, -2, -2])
        catch_rate = tm.p_a_given_q(x, attack_a, attack_b, simple_fquad)
        self.assertAlmostEqual(catch_rate, 0.25, msg="p_M no defer")

        attack_a = np.array([4, 0, 0, 7])
        attack_b = np.array([2, 3, 4, 5])

        x = np.array([0, 6.1, 6.01, 5, 4, 1, 1])
        catch_rate = tm.p_a_given_q(x, attack_a, attack_b, simple_fquad)
        self.assertAlmostEqual(catch_rate, 0.25, msg="p_N defer")

        attack_a = np.array([4, 0, -2, 7])
        attack_b = np.array([2, 3, 4, 5])

        def simple_fquad(a, b):
            return a + b

        x = np.array([0, 6.1, 6.01, 5, 3, 2, 3])
        catch_rate = tm.p_a_given_q(x, attack_a, attack_b, simple_fquad)
        self.assertAlmostEqual(catch_rate, 0.5, msg="p_N defer2")

        x = np.array([1, 6, 0, 5, 3, 3, 4])
        catch_rate = tm.p_a_given_q(x, attack_a, attack_b, simple_fquad)
        self.assertAlmostEqual(catch_rate, 0.75, msg="p_M defer")

    def test_p_a_given_not_q(self):
        benign_a = np.array([1.0, 2, 3, 4])
        benign_b = np.array([1.0, 1, 2, 3])

        def simple_fquad(a, b):
            return a + b

        x = np.array([1, 4, 4, 2, 3, -2, -2])
        false_positive_rate = tm.p_a_given_not_q(x, benign_a, benign_b, simple_fquad)
        self.assertAlmostEqual(false_positive_rate, 0.25, msg="p_M no defer")

        x = np.array([1, 4, 2, 2, 3, 5, -2])
        false_positive_rate = tm.p_a_given_not_q(x, benign_a, benign_b, simple_fquad)
        self.assertAlmostEqual(false_positive_rate, 0.5, msg="p_M w defer")

        def new_fquad(a, b):
            return a + b / 2

        false_positive_rate = tm.p_a_given_not_q(x, benign_a, benign_b, new_fquad)
        self.assertAlmostEqual(false_positive_rate, 0.25, msg="p_M w defer2")

        x = np.array([0, 4, 4, 2, 1.1, -2, -2])
        false_positive_rate = tm.p_a_given_not_q(x, benign_a, benign_b, simple_fquad)
        self.assertAlmostEqual(false_positive_rate, 0.5, msg="p_N no defer")

        x = np.array([0, 4, 4, 3, 1, -2, 4])
        false_positive_rate = tm.p_a_given_not_q(x, benign_a, benign_b, simple_fquad)
        self.assertAlmostEqual(false_positive_rate, 0.5, msg="p_N w defer")

    def test_monitor_cost(self):
        benign_a = np.array([1.0, 2, 3, 4])
        benign_b = np.array([1.0, 1, 2, 3])
        x = np.array([0, 4, 4, 2, 1.1, -2, -2])
        cost = tm.monitor_cost(x, benign_a, benign_b, 2, 1)
        self.assertAlmostEqual(cost, 1, msg="monitor cost pn only")

        x = np.array([0, 4, 4, 2, 1, -2, -2])
        cost = tm.monitor_cost(x, benign_a, benign_b, 2, 1)
        self.assertAlmostEqual(cost, 1 + 2 * 0.5, msg="monitor cost pn then pm only")

    def test_2nd_rd_cutoff_point(self):
        benign_a_filtered = np.array([1.1, 2.2, 3.3, 4.4])
        benign_b_filtered = np.array([1, 1, 2, 3])
        f_mn = lambda a, b: a + b
        cutoff = tm.get_2ndrd_cutoff_point(
            benign_a_filtered, benign_b_filtered, f_mn, 0.5
        )
        self.assertAlmostEqual(cutoff, 5.3, msg="cutoff point")

        cutoff = tm.get_2ndrd_cutoff_point(
            benign_a_filtered, benign_b_filtered, f_mn, 0.25
        )
        self.assertEqual(cutoff, 7.4, msg="cutoff point")

        cutoff = tm.get_2ndrd_cutoff_point(
            benign_a_filtered, benign_b_filtered, f_mn, 0.24
        )
        self.assertAlmostEqual(cutoff, 7.4 + 1e-6, msg="cutoff point")

    def test_get_x_with_fin_audit_thresh(self):
        benign_a = np.array([11, 12, 13, 14, 15, 16])
        benign_b = np.array([1, 2, 3, 4, 5, 6])
        f_mn = lambda a, b: a + b
        x = np.array([0, 0, 0, 5, 3, 0, 0])
        x = tm.get_x_with_fin_audit_thresh(
            x, benign_a, benign_b, f_mn, c_a=2, c_m=3, c_n=1, Y=3, N_first=True
        )
        # Y = auditupfront*p_upfront + p_two(second_monitor_cost + p_audit_in_thing*audit_cost) + c_n
        # (Y - c_n - auditupfront*p_upfront)/p_two = second_monitor_cost + p_audit_in_thing*audit_cost

        self.assertAlmostEqual(x[-1], 18, msg="get x with fin audit thresh")
        self.assertTrue(
            tm.tot_cost(
                x,
                c_m=3,
                c_n=1,
                c_a=2,
                alpha_benign_scores=benign_a,
                beta_benign_scores=benign_b,
                f_mn=f_mn,
            )
            <= 3,
            msg="get x with fin audit thresh cost check",
        )

        benign_a = np.array([1, 2, 3])
        benign_b = np.array([0.1, 0.2, 0.3])
        x = np.array([0, 0, 0, 0.3, 0.3, 0, 0])
        x = tm.get_x_with_fin_audit_thresh(
            x, benign_a, benign_b, f_mn, c_a=2, c_m=3, c_n=1, Y=3, N_first=True
        )
        self.assertAlmostEqual(x[-1], np.inf, msg="no defer")
        self.assertTrue(
            tm.tot_cost(
                x,
                c_m=3,
                c_n=1,
                c_a=2,
                alpha_benign_scores=benign_a,
                beta_benign_scores=benign_b,
                f_mn=f_mn,
            )
            <= 3,
            msg="no defer cost check",
        )

        benign_a = np.array([1, 2, 3])
        benign_b = np.array([0.1, 0.2, 0.3])
        x = np.array([0, 0, 0, 1.0, 0, 0, 0])
        x = tm.get_x_with_fin_audit_thresh(
            x, benign_a, benign_b, f_mn, c_a=2, c_m=3, c_n=1, Y=500, N_first=True
        )
        self.assertAlmostEqual(x[-1], 1.1, msg="unlimited budget")
        self.assertTrue(
            tm.tot_cost(
                x,
                c_m=3,
                c_n=1,
                c_a=2,
                alpha_benign_scores=benign_a,
                beta_benign_scores=benign_b,
                f_mn=f_mn,
            )
            <= 500,
            msg="unlimited budget cost check",
        )

        benign_a = np.array([1, 2, 3])
        benign_b = np.array([0.1, 0.2, 0.3])
        x = np.array([0, 0, 0, 1.0, 0, 0, 0])
        x = tm.get_x_with_fin_audit_thresh(
            x, benign_a, benign_b, f_mn, c_a=2, c_m=3, c_n=1, Y=4, N_first=True
        )
        self.assertAlmostEqual(x[-1], 3.3 + 1e-6, msg="no budget")
        self.assertTrue(
            tm.tot_cost(
                x,
                c_m=3,
                c_n=1,
                c_a=2,
                alpha_benign_scores=benign_a,
                beta_benign_scores=benign_b,
                f_mn=f_mn,
            )
            <= 4,
            msg="no budget cost check",
        )

    def test_get_x_with_fin_audit_thresh_more(self):
        benign_a = np.array([1, 2, 3, 4, 5, 6])
        benign_b = np.array([10, 20, 30, 40, 50, 60])
        f_mn = lambda a, b: a + b / 2
        x = np.array([0, 0, 0, 40, 10, 0, 0])
        x = tm.get_x_with_fin_audit_thresh(
            x,
            benign_a,
            benign_b,
            f_mn,
            c_a=4,
            c_m=3,
            c_n=2,
            Y=5.5 + 4 / 3 - 0.0001,
            N_first=True,
        )

        self.assertAlmostEqual(x[-1], 18, msg="get x with fin audit thresh")
        self.assertTrue(
            tm.tot_cost(
                x,
                c_m=3,
                c_n=2,
                c_a=4,
                alpha_benign_scores=benign_a,
                beta_benign_scores=benign_b,
                f_mn=f_mn,
            )
            <= 5.5 + 4 / 3 - 0.0001,
            msg="get x with fin audit thresh",
        )
        benign_a = np.array([1, 2, 3, 4, 5, 6])
        benign_b = np.array([10, 20, 30, 40, 50, 60])
        f_mn = lambda a, b: a * b
        x = np.array([0, 0, 0, 50.0, 40, 0, 0])
        x = tm.get_x_with_fin_audit_thresh(
            x, benign_a, benign_b, f_mn, c_a=4, c_m=3, c_n=2, Y=4, N_first=True
        )
        # Y = auditupfront*p_upfront + p_two(second_monitor_cost + p_audit_in_thing*audit_cost) + c_n
        # (Y - c_n - auditupfront*p_upfront)/p_two = second_monitor_cost + p_audit_in_thing*audit_cost

        self.assertEqual(x[-1], 160 + 1e-6, msg="get x with fin audit thresh")
        self.assertTrue(
            tm.tot_cost(
                x,
                c_m=3,
                c_n=2,
                c_a=4,
                alpha_benign_scores=benign_a,
                beta_benign_scores=benign_b,
                f_mn=f_mn,
            )
            <= 4,
            msg="get x with fin audit thresh",
        )

        benign_b = np.array([1, 2, 3, 4, 5, 6])
        benign_a = np.array([10, 20, 30, 40, 50, 60])
        f_mn = lambda a, b: a / 2 + b
        x = np.array([1,40, 10, 0, 0, 0, 0])
        x = tm.get_x_with_fin_audit_thresh(
            x,
            benign_a,
            benign_b,
            f_mn,
            c_a=4,
            c_m=2,
            c_n=3,
            Y=5.5 + 4 / 3 - 0.0001,
            N_first=False,
        )

        self.assertAlmostEqual(x[-1], 18, msg="get x with fin audit thresh M first")
        self.assertTrue(
            tm.tot_cost(
                x,
                c_m=2,
                c_n=3,
                c_a=4,
                alpha_benign_scores=benign_a,
                beta_benign_scores=benign_b,
                f_mn=f_mn,
            )
            <= 5.5 + 4 / 3 - 0.0001,
            msg="get x with fin audit thresh",
        )

    def test_get_end_threshold_given_audit_threshold(self):
        benign_b = np.array([10, 20, 30, 40, 50, 60])
        b_end_min = tm.get_end_threshold_given_audit_threshold(
            benign_b, 50, c_a=6, c_m=3, c_n=2, Y=4.55
        )
        # We called n, spent 2, and then audited 1/3 of our stuff, spending 2, and now we have 0.55 left, so we can afford to use c_m a little
        # over 1/6 of the time, so we'll get 40 but not 30
        print(b_end_min)
        self.assertTrue((b_end_min > 30) and (b_end_min <= 40), msg="get end threshold")
        self.assertTrue(
            tm.tot_cost(
                np.array([0, 50, b_end_min, 50, b_end_min, 0, 0]),
                c_m=3,
                c_n=2,
                c_a=6,
                alpha_benign_scores=np.zeros_like(benign_b),
                beta_benign_scores=benign_b,
                f_mn=lambda a, b: -1,
            )
            <= 4.55,
            msg="check cost",
        )

        benign_a = np.array([1, 2, 3, 4, 5, 6])
        a_end_min = tm.get_end_threshold_given_audit_threshold(
            benign_a, 6, c_a=6, c_m=3, c_n=2, Y=6, N_first=False
        )
        # Auditing upfront costed 1, and the monitor costed 3. Can. With 2 left, we can always call the second monitor now. End threshold should be 1
        print(a_end_min)
        self.assertTrue(a_end_min == 1, msg="get end threshold a first all audit")
        self.assertTrue(
            tm.tot_cost(
                np.array([1, 6, a_end_min, 0, 0, 0, 0]),
                c_m=3,
                c_n=2,
                c_a=6,
                alpha_benign_scores=benign_a,
                beta_benign_scores=np.zeros_like(benign_a),
                f_mn=lambda a, b: -1,
            )
            <= 6,
            msg="check cost",
        )

        a_end_min = tm.get_end_threshold_given_audit_threshold(
            benign_a, 6, c_a=6, c_m=3, c_n=2, Y=5, N_first=False
        )
        # Auditing upfront costed 1, and the monitor costed 3. Can. With 1 left, we can call monitors on half of all the total samples End threshold should be 1
        print(a_end_min)
        self.assertTrue(
            a_end_min <= 3 and a_end_min > 2,
            msg="get audit threshold a first half audit",
        )
        self.assertTrue(
            tm.tot_cost(
                np.array([1, 6, a_end_min, 0, 0, 0, 0]),
                c_m=3,
                c_n=2,
                c_a=6,
                alpha_benign_scores=benign_a,
                beta_benign_scores=np.zeros_like(benign_a),
                f_mn=lambda a, b: -1,
            )
            <= 5,
            msg="check cost",
        )

        benign_b = np.array([10, 20, 30, 40, 50, 60])
        b_end_min = tm.get_end_threshold_given_audit_threshold(
            benign_b, 40, c_a=6, c_m=3, c_n=2, Y=5
        )
        print(b_end_min)
        self.assertEqual(b_end_min, 40, msg="No money left")
        self.assertTrue(
            tm.tot_cost(
                np.array([0, 0, 0, 40, b_end_min, 0, 0]),
                c_m=3,
                c_n=2,
                c_a=6,
                alpha_benign_scores=np.zeros_like(benign_b),
                beta_benign_scores=benign_b,
                f_mn=lambda a, b: -1,
            )
            <= 5,
            msg="check cost",
        )


def run_tests():
    """
    Runs all tests in the TestNewFrameworkWebapp test suite.
    Can be called from another script or used as a standalone test runner.
    """
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTwoMonitor)
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == "__main__":
    run_tests()
