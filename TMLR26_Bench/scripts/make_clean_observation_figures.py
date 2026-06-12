from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "tf_ovcos_full_results_20260526/runs/analysis"
OUT = ROOT / "TMLR26_Bench/figures"

COLORS = {
    "SCLIP": "#4E79A7",
    "NACLIP": "#F28E2B",
    "ResCLIP": "#59A14F",
    "ProxyCLIP": "#B07AA1",
    "CorrCLIP": "#E15759",
    "Trident": "#76B7B2",
    "CASS": "#EDC948",
}


def setup_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.8)
    ax.tick_params(axis="both", labelsize=9)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, format="pdf", bbox_inches="tight")
    plt.close(fig)


def load_e2():
    df = pd.read_csv(ANALYSIS / "e2_vocab_robustness_filled.csv")
    keep = ["SCLIP", "NACLIP", "ResCLIP", "ProxyCLIP", "CorrCLIP"]
    return df[df["method"].isin(keep)].copy()


def fig_vocab_collapse():
    df = load_e2()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7), sharey=True)
    panels = [
        ("Context", [59, 459], "context459_ziou"),
        ("ADE", [150, 847], "ade847_ziou"),
    ]
    for ax, (title, xs, col) in zip(axes, panels):
        setup_ax(ax)
        for _, row in df.iterrows():
            method = row["method"]
            ys = [0, row[col]]
            ax.plot(xs, ys, linewidth=2.2, color=COLORS.get(method, "#555"), label=method)
            ax.text(xs[-1] + (12 if title == "Context" else 18), ys[-1], method, va="center", fontsize=8)
        ax.set_title(f"{title}: compact to large vocabulary", fontsize=12, weight="bold")
        ax.set_xlabel("vocabulary size")
        ax.set_xlim(xs[0] - 20, xs[-1] + (115 if title == "Context" else 190))
        ax.set_ylim(0, 78)
        ax.set_xticks(xs)
        ax.set_yticks(np.arange(0, 81, 20))
        ax.set_yticklabels([f"{v:.0f}%" for v in np.arange(0, 81, 20)])
    axes[0].set_ylabel("zero-IoU class rate")
    fig.suptitle("Vocabulary expansion exposes class-collapse", fontsize=14, weight="bold")
    save(fig, "fig_vocab_collapse_smooth.pdf")


def fig_miou_vs_collapse_waterfall():
    df = load_e2()
    rows = []
    for _, row in df.iterrows():
        rows.append((row["method"], "Ctx-459", row["context459_miou"], row["context459_ziou"]))
        rows.append((row["method"], "ADE-847", row["ade847_miou"], row["ade847_ziou"]))
    data = pd.DataFrame(rows, columns=["method", "dataset", "miou", "ziou"])
    data["label"] = data["method"] + "\n" + data["dataset"]
    data = data.sort_values("miou", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    setup_ax(ax)
    x = np.arange(len(data))
    widths = 0.62
    bars = ax.bar(x, data["ziou"], width=widths, color=[COLORS.get(m, "#777") for m in data["method"]])
    ax.set_ylabel("zero-IoU class rate")
    ax.set_ylim(0, max(data["ziou"]) * 1.22)
    ax.set_xticks(x)
    ax.set_xticklabels(data["label"], rotation=35, ha="right")
    ax.set_title("Large-vocabulary mIoU gains do not guarantee coverage", fontsize=14, weight="bold")
    ax.text(
        0.01,
        0.96,
        "Bars are ordered by large-vocabulary mIoU; height shows class-collapse rate.",
        transform=ax.transAxes,
        fontsize=9,
        color="#555",
        va="top",
    )
    for bar, miou in zip(bars, data["miou"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"{miou:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )
    ax.text(0.99, 0.96, "bar label = mIoU", transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#555")
    save(fig, "fig_miou_vs_class_collapse_smooth.pdf")


def fig_component_scale():
    comp = pd.read_csv(ANALYSIS / "exploratory_scale_components.csv")
    methods = ["SCLIP", "NACLIP", "ResCLIP", "ProxyCLIP", "CorrCLIP"]
    order = ["small", "medium", "large"]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    setup_ax(ax)
    for method in methods:
        sub = comp[comp["method"].str.lower() == method.lower()]
        values = []
        for scale in order:
            s = sub[sub["scale_bin"] == scale]
            values.append(float(np.average(s["mean_component_iou"], weights=s["n_components"])))
        ax.plot(x, values, linewidth=2.4, color=COLORS.get(method, "#555"), label=method)
        ax.text(x[-1] + 0.04, values[-1], method, va="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["small", "medium", "large"])
    ax.set_xlim(-0.1, 2.45)
    ax.set_ylabel("weighted mean component IoU")
    ax.set_title("Component-scale degradation is systematic", fontsize=14, weight="bold")
    save(fig, "fig_component_scale_smooth.pdf")


def fig_efficiency_waterfall():
    rows = [
        ("MaskCLIP-Attn", 13.00, 0.030),
        ("Trident", 43.31, 0.080),
        ("SCLIP", 41.61, 0.115),
        ("SC-CLIP", 42.96, 0.154),
        ("CLIPtrase", 41.13, 0.166),
        ("ProxyCLIP", 43.78, 0.167),
        ("CorrCLIP", 49.22, 0.193),
        ("MaskCLIP-Attn-Slide", 17.04, 0.319),
        ("MaskCLIP", 12.33, 0.680),
        ("NACLIP", 43.23, 0.883),
        ("ResCLIP", 41.11, 1.179),
        ("CASS", 42.25, 2.207),
        ("DINOv2+SAM+CLIP", 25.91, 3.507),
        ("SAM-AMG+CLIP", 24.07, 4.124),
        ("DINOv2+SAM+SigLIP", 27.79, 4.341),
        ("SAM-AMG+SigLIP", 28.85, 5.236),
    ]
    df = pd.DataFrame(rows, columns=["method", "miou", "sec"])
    df = df.sort_values("sec").reset_index(drop=True)
    pareto = []
    best = -1
    for idx, row in df.iterrows():
        if row["miou"] > best:
            pareto.append(idx)
            best = row["miou"]
    colors = ["#D8D8D8"] * len(df)
    for idx in pareto:
        colors[idx] = "#4E79A7"

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    setup_ax(ax)
    y = np.arange(len(df))
    ax.barh(y, df["sec"], color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(df["method"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("seconds per image")
    ax.set_title("Efficiency changes the practical ranking", fontsize=14, weight="bold")
    for yi, (_, row) in enumerate(df.iterrows()):
        ax.text(row["sec"] + 0.05, yi, f"{row['miou']:.1f}", va="center", fontsize=8)
    ax.text(0.99, 0.04, "bar label = E1 avg mIoU; blue = Pareto frontier", transform=ax.transAxes, ha="right", fontsize=9, color="#555")
    ax.set_xlim(0, df["sec"].max() * 1.22)
    save(fig, "fig_efficiency_pareto.pdf")


def fig_diagnostic_decomposition():
    diag = pd.read_csv(ANALYSIS / "table9_diagnostic_probes.csv")
    method = "corrclip"
    sub = diag[diag["method"] == method]
    metrics = [
        ("Naming\nTop-1", "gt_region_naming_top1"),
        ("Text-localization\nIoU", "gt_text_localization_iou"),
        ("Proposal-oracle\nIoU", "proposal_oracle_iou"),
        ("Proposal\nRecall@0.5", "proposal_recall_at_05"),
        ("MCMR@0.5", "mcmr_at_05"),
    ]
    values = [float(sub[col].mean()) for _, col in metrics]
    labels = [m[0] for m in metrics]
    fig, ax = plt.subplots(figsize=(7.8, 4.1))
    setup_ax(ax)
    x = np.arange(len(values))
    ax.bar(x, values, color=["#4E79A7", "#76B7B2", "#59A14F", "#EDC948", "#E15759"], width=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("diagnostic value")
    ax.set_title("Naming, localization, and proposal ceilings separate", fontsize=14, weight="bold")
    for xi, val in zip(x, values):
        ax.text(xi, val + 0.025, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    save(fig, "fig_diagnostic_decomposition.pdf")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelsize": 10,
            "axes.titlesize": 12,
        }
    )
    fig_vocab_collapse()
    fig_miou_vs_collapse_waterfall()
    fig_component_scale()
    fig_efficiency_waterfall()
    fig_diagnostic_decomposition()


if __name__ == "__main__":
    main()
