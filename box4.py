import numpy as np

import pinn

if __name__ == "__main__":
    # study design
    rois = 50

    # fixed a priori
    effect_size = 0.5
    alpha_0 = 0.05

    # measured
    rho_ij = 0.25
    rho_uv = 0.81

    # computations
    comparisons = rois * (rois - 1) // 2
    alpha = alpha_0 / comparisons
    e_ij_n = np.ceil(
        pinn.power.tt_ind_solve_power(
            effect_size=effect_size, alpha=alpha, power=0.8, reliability=rho_ij
        )
    )
    e_uv_n = np.ceil(
        pinn.power.tt_ind_solve_power(
            effect_size=effect_size, alpha=alpha, power=0.8, reliability=rho_uv
        )
    )

    print(
        "Sample size required for edge {ij} with reliability "
        f"{rho_ij} to attain 80% power for effect size {effect_size} is {e_ij_n:.0f}"
    )

    print(
        "Sample size required for edge {uv} with reliability "
        f"{rho_uv} to attain 80% power for effect size {effect_size} is {e_uv_n:.0f}"
    )
