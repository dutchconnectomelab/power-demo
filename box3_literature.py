from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.power import tt_ind_solve_power


def load_litreview() -> pd.DataFrame:
    sc_files = [Path(__file__).parent / "assets" / "sc_2021.csv"]
    fc_files = [Path(__file__).parent / "assets" / "fc_2021.csv"]

    df = pd.DataFrame()

    for file in sc_files:
        df_ = pd.read_csv(file)
        df_["modality"] = "sc"
        df = pd.concat([df, df_])

    for file in fc_files:
        df_ = pd.read_csv(file)
        df_["modality"] = "fc"
        df = pd.concat([df, df_])

    df = df[df["Include"] == 1]
    print(f"Loaded {len(df)} entries.")
    return df


def assess_power(df: pd.DataFrame, alpha: float, effect_size: float, name=""):
    power = []

    for _, row in df.iterrows():
        sample_sizes = np.array([row[f"Sample {i}"] for i in range(1, 6)])
        sample_sizes = sample_sizes[~np.isnan(sample_sizes)]
        if len(sample_sizes) == 1:
            sample_sizes = [sample_sizes[0], sample_sizes[0]]
        sample_sizes = sorted(sample_sizes)[-2:]
        sample_sizes = np.array(sample_sizes, dtype=np.int16)

        power.append(
            tt_ind_solve_power(
                effect_size,
                sample_sizes[0],
                alpha=alpha,
                ratio=sample_sizes[1] / sample_sizes[0],
            )
        )
    power = np.array(power)

    print(
        f"\nPOWER ESTIMATION {name} (d={effect_size}, alpha={alpha}; {len(power)} entries)"
    )
    print("Median: ", np.median(power))
    print("ratio below 30% power: ", np.mean(power < 0.3))
    print("ratio above 80% power: ", np.mean(power >= 0.8))


def power_discussion(df):
    print("\n## POWER DISCUSSIONS ##")
    n_disc = np.sum(df["Mentions power"])
    print(f"Power discussed in {n_disc:.0f} out of {len(df)} papers")
    print(
        f"Power analysis included in {np.sum(df['Power analysis']):.0f} out of {len(df)} papers"
    )

    def clean_lbl(lbl: str) -> str:
        lbl = lbl.strip()
        if lbl[-1] == "s":
            lbl = lbl[:-1]

        mapping = {
            "motivate method": "motivation",
            "motivaiton": "motivation",
            "limitatino": "limitation",
            "discussion": "interpretation",
            "protocol?": "ignore",
            "post-hoc power analysi": "ignore",
        }

        if lbl in mapping.keys():
            lbl = mapping[lbl]

        return lbl

    limitation, motivation, interpretation = 0, 0, 0
    for descr in df["Power discussion"]:
        descr = str(descr)
        if descr == "nan":
            continue
        for lbl in descr.split(","):
            lbl = clean_lbl(lbl)
            if lbl == "limitation":
                limitation += 1
            elif lbl == "motivation":
                motivation += 1
            elif lbl == "interpretation":
                interpretation += 1
            elif lbl == "ignore":
                pass
            else:
                raise ValueError(f"Unknown discussion label {lbl}")

    print("Power was discussed in the following contexts:")
    print(f"- Limitation: {limitation} ({100*limitation/n_disc:.1f}%)")
    print(f"- Motivation: {motivation} ({100*motivation/n_disc:.1f}%)")
    print(f"- Interpretation: {interpretation} ({100*interpretation/n_disc:.1f}%)")


if __name__ == "__main__":
    df = load_litreview()

    print("\n## TOTAL INCLUDED ##")
    for m in ["sc", "fc"]:
        print(f"{m.upper()}: {np.sum(df['modality'] == m)}")

    # print("\n## POWER ESTIMATES ##")
    # for m in ["sc", "fc"]:
    #     for e in [0.2, 0.8]:
    #         for a in [0.05, 0.001, 0.05 / 1225]:
    #             assess_power(df[df["modality"] == m], a, e, m.upper())

    power_discussion(df)
