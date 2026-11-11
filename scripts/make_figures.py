"""Figures for the README, built only from results/*.csv."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# Fixed categorical order (never cycled), plus ink and chrome colours; the same style as the PTB-XL figures.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cc7"]
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
CONDITION_LABELS = {
    "none": "all inputs", "no_wearable": "no wearable", "no_labs": "no labs", "no_exam": "no exam",
    "no_questionnaire": "no\nquestionnaire", "no_clinic": "no clinic visit\n(labs, exam)",
    "questionnaire_only": "questionnaire,\nage, sex only",
}
APPROACH_LABELS = {
    "fused": "fusion, modality dropout", "fused_naive": "fusion, no dropout",
    "cox_subset": "Cox refitted per subset", "cox_impute": "Cox, mean-imputed",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2, "legend.frameon": False, "legend.labelcolor": INK_2, "font.size": 9.5,
    "lines.linewidth": 2, "savefig.dpi": 200,
})


def _err(rows: pd.DataFrame, name: str) -> list[pd.Series]:
    return [rows[name] - rows[f"{name}_lo"], rows[f"{name}_hi"] - rows[name]]


def _title(fig, text: str, legend_ax=None, ncol=None) -> None:
    """Figure title above the panel titles; the legend (if any) in one row beside it, clear of the data."""
    fig.suptitle(text, x=0.01, y=1.06, ha="left", color=INK, fontsize=12)
    if legend_ax is not None:
        handles, labels = legend_ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.99, 1.0), ncol=ncol or len(labels))


def _save(fig, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, bbox_inches="tight")
    plt.close(fig)


def e1_reliability() -> None:
    table = pd.read_csv(RESULTS / "e1_reliability.csv")
    overall = pd.read_csv(RESULTS / "e1_overall.csv").set_index("model")
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    top = 0
    for k, (model, label) in enumerate([("cox_full", "Cox, all inputs"), ("fused", "fusion model")]):
        rows = table[table["model"] == model]
        citl = overall.loc[model, "citl5"]
        ax.plot(rows["mean_pred"], rows["observed"], color=SERIES[k], marker="o", markersize=4,
                label=f"{label} (CITL {citl:+.3f})")
        top = max(top, rows["mean_pred"].max(), rows["observed"].max())
    top *= 1.05
    ax.plot([0, top], [0, top], color=AXIS, linewidth=1, linestyle="--")
    ax.set(xlim=(0, top), ylim=(0, top), xlabel="mean predicted 5-year risk (decile)",
           ylabel="observed 5-year risk (Kaplan-Meier)")
    ax.legend(loc="upper left")
    ax.set_title("E1: calibration, complete cases (dashed = perfect)", loc="left")
    _save(fig, "e1_reliability.png")


def e2_missing() -> None:
    table = pd.read_csv(RESULTS / "e2_simulated.csv")
    conditions = list(CONDITION_LABELS)
    x = np.arange(len(conditions))

    fig, (ax_c, ax_cal) = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
    for k, (approach, label) in enumerate(APPROACH_LABELS.items()):
        rows = table[table["approach"] == approach].set_index("condition").reindex(conditions)
        offset = (k - 1.5) * 0.17
        for ax, name in [(ax_c, "c_index"), (ax_cal, "citl5")]:
            ax.errorbar(x + offset, rows[name], yerr=_err(rows, name), fmt="o", color=SERIES[k], markersize=5,
                        elinewidth=1.5, capsize=0, label=label)
    ax_cal.axhline(0, color=MUTED, linewidth=1, linestyle=":")
    ax_c.set_ylabel("Harrell's C (95% CI)")
    ax_cal.set_ylabel("predicted - observed\n5-year risk (95% CI)")
    ax_c.set_title("Ranking", loc="left")
    ax_cal.set_title("Calibration in the large", loc="left")
    for ax in (ax_c, ax_cal):
        ax.grid(axis="x", visible=False)
    ax_cal.set_xticks(x, [CONDITION_LABELS[c] for c in conditions])
    _title(fig, "E2: inputs removed at test time (complete cases, n=8,022)", legend_ax=ax_c, ncol=2)
    _save(fig, "e2_missing.png")


def e3_days() -> None:
    table = pd.read_csv(RESULTS / "e3_days.csv")
    labels = [("fused", "fusion model"), ("deep_wearable", "wearable + age, sex"), ("cox_full", "Cox, all inputs")]
    fig, (ax_c, ax_cal) = plt.subplots(1, 2, figsize=(10.5, 3.4), gridspec_kw={"wspace": 0.35})
    for k, (model, label) in enumerate(labels):
        rows = table[table["model"] == model].sort_values("days_kept")
        offset = (k - 1) * 0.12
        for ax, name in [(ax_c, "c_index"), (ax_cal, "citl5")]:
            ax.errorbar(rows["days_kept"] + offset, rows[name], yerr=_err(rows, name), fmt="o-", color=SERIES[k],
                        markersize=5, elinewidth=1.2, capsize=0, label=label)
    ax_cal.axhline(0, color=MUTED, linewidth=1, linestyle=":")
    n, deaths = int(table["n"].iloc[0]), int(table["deaths_5y"].iloc[0])
    ax_c.set(ylabel="Harrell's C (95% CI)", xlabel="valid wear days kept at test time")
    ax_cal.set(ylabel="predicted - observed 5-year risk", xlabel="valid wear days kept at test time")
    ax_c.set_title("Ranking", loc="left")
    ax_cal.set_title("Calibration in the large", loc="left")
    for ax in (ax_c, ax_cal):
        ax.set_xticks(sorted(table["days_kept"].unique()))
    _title(fig, f"E3: fewer wear days, people with 7 valid days (n={n:,}, {deaths} deaths within 5 years)")
    handles, labels = ax_c.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=len(labels))
    _save(fig, "e3_days.png")


def e4_e5_calibration() -> None:
    table = pd.read_csv(RESULTS / "e5_groups.csv")
    attributes = [("sex", ["male", "female"]), ("age_band", None), ("race", None)]
    order = []
    for attribute, groups in attributes:
        found = list(table[(table["variant"] == "fused") & (table["attribute"] == attribute)]["group"])
        order += [(attribute, g) for g in (groups or sorted(found))]
    y = np.arange(len(order))[::-1]
    variants = [("fused", "fusion model"), ("fused_recal_global", "+ one recalibration for everyone"),
                ("group", "+ recalibration per group")]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for k, (variant, label) in enumerate(variants):
        xs, lo, hi = [], [], []
        for attribute, group in order:
            name = f"fused_recal_{attribute}" if variant == "group" else variant
            row = table[(table["variant"] == name) & (table["attribute"] == attribute) &
                        (table["group"] == group)].iloc[0]
            xs.append(row["citl5"]), lo.append(row["citl5"] - row["citl5_lo"]), hi.append(row["citl5_hi"] - row["citl5"])
        ax.errorbar(xs, y + (1 - k) * 0.22, xerr=[lo, hi], fmt="o", color=SERIES[k], markersize=5, elinewidth=1.5,
                    capsize=0, label=label)
    n = table[table["variant"] == "fused"].set_index(["attribute", "group"])
    ax.set_yticks(y, [f"{g} (n={n.loc[(a, g), 'n']:,}, {n.loc[(a, g), 'deaths_5y']} deaths in 5 y)" for a, g in order])
    for boundary in np.cumsum([len(table[(table["variant"] == "fused") & (table["attribute"] == a)])
                               for a, _ in attributes])[:-1]:
        ax.axhline(len(order) - boundary - 0.5, color=AXIS, linewidth=1)
    ax.axvline(0, color=MUTED, linewidth=1, linestyle=":")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("predicted - observed 5-year risk (95% CI)")
    ax.set_title("E4-E5: calibration by group, before and after recalibration", loc="left", fontsize=12, pad=10)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.55, 0.0), ncol=3)
    _save(fig, "e4_e5_calibration.png")


def e6_activity() -> None:
    table = pd.read_csv(RESULTS / "e6_activity.csv")
    analyses = list(pd.unique(table["analysis"]))
    variants = [("none", "-", "unadjusted"), ("age and sex", "linear", "age and sex"),
                ("all clinical inputs", "linear", "all clinical inputs, linear"),
                ("all clinical inputs", "boosting", "all clinical inputs, boosted")]
    y = np.arange(len(analyses))[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for k, (adjustment, nuisance, label) in enumerate(variants):
        rows = table[(table["adjustment"] == adjustment) & (table["nuisance"] == nuisance)].set_index("analysis")
        rows = rows.loc[analyses]
        ax.errorbar(rows["rr"], y + (1.5 - k) * 0.17, xerr=[rows["rr"] - rows["rr_low"], rows["rr_high"] - rows["rr"]],
                    fmt="o", color=SERIES[k], markersize=5, elinewidth=1.5, capsize=0, label=label)
    n = table.drop_duplicates("analysis").set_index("analysis")
    ax.set_yticks(y, [f"{a}\n(n={n.loc[a, 'n']:,}, {n.loc[a, 'deaths']} deaths)" for a in analyses])
    ax.set_xscale("log")
    ticks = [0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.axvline(1, color=MUTED, linewidth=1, linestyle=":")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("risk ratio of death within 5 years, most vs least active third (95% CI)")
    ax.set_title("E6: activity and mortality, by adjustment", loc="left", fontsize=12, pad=10)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.55, 0.0), ncol=2)
    _save(fig, "e6_activity.png")


def main() -> None:
    e1_reliability()
    e2_missing()
    e3_days()
    e4_e5_calibration()
    e6_activity()


if __name__ == "__main__":
    main()
