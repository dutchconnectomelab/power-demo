import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.power import tt_ind_solve_power

import pinn


def monte_carlo_power(
    mean_controls,
    mean_patients,
    sample_size,
    alpha,
    rm_zeros=None,
    iterations=10_000,
    edge_generator=None,
):
    """Estimate power for edgewise comparison with missing values.

    Args:
      mean_controls: mean edge weight for the control group
      mean_patients: mean edge weight for the patient group
      sample_size: number of observations per group
      alpha: significance threshold
      rm_zeros: whether or not to remove missing values before inference
      iterations: number of simulated studies used to estimate power
      edge_generator: function used to generate edges. Should accept arguments `sample_size` and `mean`

    Return power (ratio of studies in which null hypothesis was correctly rejected)
    """
    # The main loop, simulating studies while keeping
    # track of how many are successful.
    detection_count = 0
    for _ in range(iterations):
        edges_controls = edge_generator(sample_size=sample_size, mean=mean_controls)
        edges_patients = edge_generator(sample_size=sample_size, mean=mean_patients)

        if rm_zeros:
            edges_controls = edges_controls[edges_controls != 0]
            edges_patients = edges_patients[edges_patients != 0]

        # we count a detection as successful if it is significant (at alpha)
        # and in the right direction
        if (ttest_ind(edges_controls, edges_patients)[1] < alpha) and (
            np.sign(np.mean(edges_controls) - np.mean(edges_patients))
            == np.sign(mean_controls - mean_patients)
        ):
            detection_count += 1

    # our power estimation is simply the ratio of successful studies
    return detection_count / iterations


def box5_measurement_error():
    effect_size = 0.5
    alpha = 4e-5
    sample_size = 200
    rho_ij = 0.25
    rho_uv = 0.81

    edge_ij_power = pinn.power.tt_ind_solve_power(
        effect_size, sample_size, alpha, reliability=rho_ij
    )
    edge_uv_power = pinn.power.tt_ind_solve_power(
        effect_size, sample_size, alpha, reliability=rho_uv
    )

    print(
        "Power in edge {ij} with reliability " f"{rho_ij} is {edge_ij_power*100:.2f}%"
    )
    print(
        "Power in edge {uv} with reliability " f"{rho_uv} is {edge_uv_power*100:.2f}%"
    )


def box5_zeros():
    effect_size = 0.5
    mean_controls = 0.53
    sample_size = 200
    alpha = 0.05 / 1225

    print("\n## CASE 1: Thresholding functional connectiviy ##")

    # For simplicity we simply use a standard normal distribution;
    # this allows us to directly use the Cohen's d effect size
    def thresholded_edges(sample_size, mean=0):
        edges = mean + np.random.randn(sample_size)
        edges[edges < 0] = 0  # remove negative edges
        return edges

    print(
        f"Running simulations with effect size d = {effect_size}, sample size = {sample_size} and alpha = {alpha:.6f}."
    )

    print(
        f"In controls, about {np.mean(thresholded_edges(1_000_000, mean_controls) == 0)*100:.0f}% of edges are thresholded."
    )

    power_with_zeros = monte_carlo_power(
        mean_controls,
        mean_controls - effect_size,
        sample_size=sample_size,
        alpha=alpha,
        rm_zeros=False,
        edge_generator=thresholded_edges,
    )

    power_without_zeros = monte_carlo_power(
        mean_controls,
        mean_controls - effect_size,
        sample_size=sample_size,
        alpha=alpha,
        rm_zeros=True,
        edge_generator=thresholded_edges,
    )

    print(f"Power when retaining zeros: {power_with_zeros*100:.2f}%.")
    print(f"Power when removing zeros: {power_without_zeros*100:.2f}%.")

    print("\n## CASE 2: Missing values in structural connectivity ##")
    ratio_missing = 0.3

    # We use a beta distribution to generate SC values representing mean FA
    def edges_with_missing_vals(sample_size, mean=0, prob_missing=ratio_missing):
        a = 3
        b = a / mean - a
        edges = np.random.beta(a, b, sample_size)

        # missing values are distributed randomly
        missing = np.random.rand(sample_size) < prob_missing
        edges[missing] = 0

        return edges

    print(
        f"Running simulations with effect size d = {effect_size}, sample size = {sample_size}, "
        f"alpha = {alpha:.6f} and {ratio_missing*100}% missing values."
    )

    # convert effect size to mean fa difference
    estimated_variance = np.var(
        edges_with_missing_vals(1_000_000, mean=mean_controls, prob_missing=0)
    )
    fa_difference = effect_size * np.sqrt(estimated_variance)
    print(
        f"For the simulated distribution, a Cohen's d of {effect_size} "
        f"corresponds to an absolute mean difference of ~{fa_difference:.2f}."
    )

    power_with_zeros = monte_carlo_power(
        mean_controls,
        mean_controls - fa_difference,
        sample_size=sample_size,
        alpha=alpha,
        rm_zeros=False,
        edge_generator=edges_with_missing_vals,
    )

    power_without_zeros = monte_carlo_power(
        mean_controls,
        mean_controls - fa_difference,
        sample_size=sample_size,
        alpha=alpha,
        rm_zeros=True,
        edge_generator=edges_with_missing_vals,
    )

    print(f"Power when retaining zeros: {power_with_zeros*100:.2f}%.")
    print(f"Power when removing zeros: {power_without_zeros*100:.2f}%.")

    print("\n## Effective sample size and power ##")
    print(f"Evaluating power for effect size d = {effect_size} and alpha = {alpha:6f}.")
    for s in [sample_size, int(sample_size * (1 - ratio_missing))]:
        print(
            f"Power when sample size = {s}: {tt_ind_solve_power(effect_size, s, alpha)*100:.2f}%"
        )


def box5_topology():
    pass


if __name__ == "__main__":
    box5_measurement_error()
    box5_zeros()
    box5_topology()
