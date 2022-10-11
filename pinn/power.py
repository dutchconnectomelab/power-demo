import numpy as np
from statsmodels.stats import power as smpower


def tt_ind_solve_power(
    effect_size=None,
    nobs1=None,
    alpha=None,
    power=None,
    reliability=1,
    **kwargs,
):
    """Wrapper of statsmodels.stats.power.tt_ind_solve_power with reliability correction"""
    if not (0 <= reliability <= 1):
        raise ValueError(
            f"Reliability score should lie in [0,1], instead got {reliability}."
        )

    if effect_size is not None:
        return smpower.tt_ind_solve_power(
            effect_size=effect_size * np.sqrt(reliability),
            nobs1=nobs1,
            alpha=alpha,
            power=power,
            **kwargs,
        )
    else:
        return smpower.tt_ind_solve_power(
            nobs1=nobs1,
            alpha=alpha,
            power=power,
            **kwargs,
        ) / np.sqrt(reliability)
