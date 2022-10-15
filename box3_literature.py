import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.power import tt_ind_solve_power


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    def f(x):
        if x == "1":
            return 1
        if x == "nan":
            return 0
        if x == "0":
            return 0
        if np.isnan(x):
            return 0
        if x == 0:
            return 0
        if x == 1:
            return 1
        raise ValueError(x)

    df["Mentions power"] = [f(x) for x in df["Mentions power"]]

    return df


def load_litreview() -> pd.DataFrame:
    sc_files = [
        Path(__file__).parent / "assets" / "sc_2019.csv",
        Path(__file__).parent / "assets" / "sc_2020.csv",
        Path(__file__).parent / "assets" / "sc_2021.csv",
        Path(__file__).parent / "assets" / "sc_2022.csv",
    ]
    fc_files = [
        Path(__file__).parent / "assets" / "fc_2019.csv",
        Path(__file__).parent / "assets" / "fc_2020.csv",
        Path(__file__).parent / "assets" / "fc_2021.csv",
        Path(__file__).parent / "assets" / "fc_2022.csv",
    ]

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

    df = df.drop_duplicates(subset=["PMID"])
    print(f"Unique PMID {len(df)} entries.")

    df = clean_data(df)
    print(df["Mentions power"].unique())

    return df


def assess_power(df: pd.DataFrame, alpha: float, effect_size: float, name=""):
    power = []

    def read_samplesize(x):
        if x == "nan":
            return np.nan
        if not type(x) == str:
            if np.isnan(x):
                return np.nan
        assert f"{x}".split(".")[0] == f"{int(x)}"
        return int(x)

    for _, row in df.iterrows():
        sample_sizes = np.array(
            [read_samplesize(row[f"Sample {i}"]) for i in range(1, 6)]
        )
        try:
            sample_sizes = sample_sizes[~np.isnan(sample_sizes)]
            if len(sample_sizes) == 1:
                print("SKIPPING")
                continue
                sample_sizes = [sample_sizes[0], sample_sizes[0]]
            sample_sizes = sorted(sample_sizes)[-2:]
            sample_sizes = np.array(sample_sizes, dtype=np.int16)
        except TypeError:
            raise RuntimeError(f"Could not process sample sizes: {sample_sizes}")

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

    resdf = pd.DataFrame(
        {
            "group": f"(d={effect_size}, alpha={alpha}",
            "modality": name,
            "power": power,
        }
    )

    return resdf


def power_violinplot(df):
    ax = sns.violinplot(x="group", y="power", hue="modality", data=df, cut=0)
    ax.set(ylim=(0, 1), yticks=(0, 0.8, 1))
    ax.axhline(0.8, ls="--", color="red")
    plt.savefig(fn := Path("RESULTS") / "power_in_literature_violinplot.svg")
    print(f"Saved figure to {fn }")
    plt.show()


def power_discussion(df):
    print("\n## POWER DISCUSSIONS ##")
    n_disc = np.sum(df["Mentions power"])
    print(f"Power discussed in {n_disc:.0f} out of {len(df)} papers")
    print(
        f"Power analysis included in {np.sum(df['Power analysis']):.0f} out of {len(df)} papers"
    )

    def clean_lbl(lbl: str) -> str:
        lbl = lbl.strip().lower()
        if lbl[-1] == "s":
            lbl = lbl[:-1]

        mapping = {
            "motivate method": "motivation",
            "motivaiton": "motivation",
            "limitatino": "limitation",
            "limiation": "limitation",
            "limitation (lack of power)": "limitation",
            "discussion": "interpretation",
            "protocol?": "ignore",
            "post-hoc power analysi": "ignore",
            "other": "ignore",
        }

        if lbl in mapping.keys():
            lbl = mapping[lbl]

        return lbl

    limitation, motivation, interpretation = 0, 0, 0
    for descr in df[df["Mentions power"] == 1]["Power discussion"]:
        descr = str(descr)
        if descr == "nan":
            continue
        for lbl in descr.replace("/", ",").split(","):
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

    print("\n## POWER ESTIMATES ##")
    resdf = pd.DataFrame()
    for m in ["sc", "fc"]:
        for e, a in zip([0.2, 0.8], [0.05, 0.001]):
            df_ = assess_power(df[df["modality"] == m], a, e, m.upper())
            resdf = pd.concat([resdf, df_])

    Path("RESULTS").mkdir(exist_ok=True)
    power_violinplot(resdf)

    power_discussion(df)
