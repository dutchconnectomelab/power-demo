import numpy as np
import pinn
from statsmodels.stats.power import tt_ind_solve_power


def test_tt_ind_solve_power():
    sample_size = 100
    effect_size = 0.2
    alpha = 0.05

    np.testing.assert_almost_equal(
        pinn.power.tt_ind_solve_power(
            effect_size,
            sample_size,
            alpha,
            reliability=1,
            alternative="larger",
        ),
        tt_ind_solve_power(
            effect_size,
            sample_size,
            alpha,
            alternative="larger",
        ),
    )

    np.testing.assert_almost_equal(
        pinn.power.tt_ind_solve_power(
            effect_size,
            sample_size,
            alpha,
            reliability=0,
            alternative="larger",
        ),
        alpha,
    )

    power = pinn.power.tt_ind_solve_power(
        effect_size,
        sample_size,
        alpha,
        reliability=0.5,
    )
    np.testing.assert_almost_equal(
        effect_size,
        pinn.power.tt_ind_solve_power(
            effect_size=None,
            nobs1=sample_size,
            alpha=alpha,
            power=power,
            reliability=0.5,
        ),
    )
