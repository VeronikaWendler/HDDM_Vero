
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
import statsmodels.api as sm


BASE_DIR = Path(r"C:\Cluster_Github\HDDM_Vero\figures_dir_garcia"
                r"\macleod_cluster_out\garcia_replication_ES_31"
                r"\recovery_ES31")
CSV_PATH     = BASE_DIR / "true_vs_recovered_ES31_2.csv"
OUT_FIG_PATH = BASE_DIR / "regplot_true_vs_recovered_2.png"


def regplot_with_corr(data, x="true", y="recovered",
                      ax=None,
                      annot_kwargs=None,
                      scatter_kwargs=None):
    """Scatter + OLS line + r / p / β-coeff annotations."""
    if ax is None:
        ax = plt.gca()
    annot_kwargs   = annot_kwargs or dict(fontsize=8, xy=(0.95, 0.05),
                                          ha="right", va="bottom")
    scatter_kwargs = scatter_kwargs or dict(s=40, alpha=.6)

    # scatter and OLS line
    sns.regplot(data=data,
                x=x,
                y=y,
                ci=95,
                scatter_kws=scatter_kwargs,
                line_kws={"linewidth": 1.0},
                ax=ax)
    r, p = pearsonr(data[x], data[y])
    X    = sm.add_constant(data[x])
    β0, β1 = sm.OLS(data[y], X).fit().params
    annot = (f"$r={r:.2f}$\n"
             f"{'p < 0.001' if p < .001 else f'p = {p:.3f}'}\n"
             f"$\\beta_0={β0:.2f}$\n$\\beta_1={β1:.2f}$")
    ax.annotate(annot, xycoords="axes fraction", **annot_kwargs)
    return ax


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    n_par   = df["parameter"].nunique()
    n_cols  = 3
    n_rows  = int(np.ceil(n_par / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.2, n_rows * 3.2),
                             squeeze=False)
    for ax, (par, sub) in zip(axes.ravel(), df.groupby("parameter")):
        regplot_with_corr(sub, ax=ax)
        ax.set_title(par, fontsize=9)
        ax.set_xlabel("true value")
        ax.set_ylabel("posterior mean")
    for ax in axes.ravel()[len(df["parameter"].unique()):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_FIG_PATH, dpi=300)
    print(f"✓ regression panels saved → {OUT_FIG_PATH}")


if __name__ == "__main__":
    main()