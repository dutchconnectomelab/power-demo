from pinn.utils import smallest_significant_effect_ttest

if __name__ == "__main__":
    # impact of bonferroni correction on smallest detectable effect
    rois = 50
    sample_size = 200
    alpha = 0.05

    comparisons = rois * (rois - 1) // 2

    effect_uncorrected = smallest_significant_effect_ttest(sample_size, alpha)
    effect_corrected = smallest_significant_effect_ttest(
        sample_size, alpha / comparisons
    )

    print(
        f"Smallest detectable effect before Bonferroni (alpha = {alpha}) is "
        f"Cohen's d = {effect_uncorrected:.3f}."
    )
    print(
        "Smallest detectable effect after Bonferroni "
        f"(alpha = {alpha}/{comparisons} = {alpha/comparisons:.6f}) is "
        f"Cohen's d = {effect_corrected:.3f}."
    )
