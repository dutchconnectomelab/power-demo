import numpy as np
from pinn import utils
from scipy.stats import norm, ttest_ind


def test_smallest_effect_size():
    sample_size = 80
    alpha = 0.05
    epsilon = 0.01

    smallest_effect = utils.smallest_significant_effect_ttest(sample_size, alpha)

    def d_to_p(d, n=sample_size):
        t = d * np.sqrt(n / 2)
        return 2 * norm.cdf(-np.abs(t))  # assuming two-tailed

    assert d_to_p(smallest_effect) <= alpha
    assert d_to_p(smallest_effect - epsilon) > alpha
