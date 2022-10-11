from statsmodels.stats.power import tt_ind_solve_power


def tt_critical_effect_size(sample_size: int, alpha: float = 0.05) -> float:
    """Find the smallest effect size (Cohen's d) that is still significant in a two-tailed ttest."""
    return tt_ind_solve_power(nobs1=sample_size, alpha=alpha, power=0.5)
