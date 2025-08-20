import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 8

rng = np.random.default_rng(seed=13)


def plot_data(
    df: pd.DataFrame,
    x: str,
    y_scat: str,
    y_line: str,
    hue: str,
    markers: list[str],
    palette: list[str],
    xlim: tuple[float, float],
    xticks: np.ndarray[float],
    xmticks: np.ndarray[float],
    xlabel: str,
    ylim: tuple[float, float],
    yticks: np.ndarray[float],
    ymticks: np.ndarray[float],
    ylabel: str,
    ax: plt.Axes,
) -> None:
    """Plot the data on a given axis

    Args:
        df (pd.DataFrame): The DataFrame to plot from
        x (str): column to plot on x-axis
        y (str): column to plot on y-axis
        hue (str): column specifiying hue
        markers (list[str]): Markers for plot
        palette (list[str]): color palette for the plot
        xlim (tuple[float, float]): x-axis limits
        xticks (np.ndarray[float]): x-axis ticks
        xmticks (np.ndarray[float]): x-axis minor ticks
        xlabel (str): x-axis label
        ylim (tuple[float, float]): y-axis limits
        yticks (np.ndarray[float]): y-axis ticks
        ymticks (np.ndarray[float]): y-axis minor ticks
        ylabel (str): y-axis label
        ax (plt.Axes): matplotlib axes to plot on
    """
    sns.scatterplot(
        df,
        x=x,
        y=y_scat,
        hue=hue,
        style=hue,
        palette=palette,
        markers=markers,
        ax=ax,
        legend=False,
    )

    sns.lineplot(
        df,
        x=x,
        y=y_line,
        hue=hue,
        style=hue,
        palette=palette,
        ax=ax,
        legend=True,
        sort=False,
    )

    ax.set_xlabel(xlabel)
    ax.set_xticks(xticks)
    ax.set_xticks(xmticks, minor=True)
    # Lim last so ticks won't override
    ax.set_xlim(xlim)
    if xlabel == "":
        ax.set_xticklabels([""] * len(xticks))

    ax.set_ylabel(ylabel)
    ax.set_yticks(yticks)
    ax.set_yticks(ymticks, minor=True)
    ax.set_ylim(ylim)
    if ylabel == "":
        ax.set_yticklabels([""] * len(yticks))

if __name__ == "__main__":
    x = np.arange(-10.0, 10.01, 1.0)
    E1_est = x**2.0 + 5 + rng.normal(0.0, 5.0, size=x.shape)
    E1_act = x**2.0 + 5

    E2_est = -(x**2.0) - 5 + rng.normal(0.0, 5.0, size=x.shape)
    E2_act = -(x**2.0) - 5

    F1_est = 2.0 * x + rng.normal(0.0, 1.0, size=x.shape)
    F1_act = 2.0 * x

    F2_est = -2.0 * x + rng.normal(0.0, 1.0, size=x.shape)
    F2_act = -2.0 * x

    particle = [f"Particle {ii}" for ii in np.repeat([1, 2], len(x))]

    x = np.hstack((x, x))

    E_est = np.hstack((E1_est, E2_est))
    E_act = np.hstack((E1_act, E2_act))

    F_est = np.hstack((F1_est, F2_est))
    F_act = np.hstack((F1_act, F2_act))

    plot_df = pd.DataFrame(
        data=np.column_stack((x, E_est, E_act, F_est, F_act)),
        columns=["x", "E_est", "E_act", "F_est", "F_act"],
    )
    plot_df.loc[:, "particle"] = particle

    # fig size should be set to journal's specifications
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=[6.25, 4.0])

    # Adjust until all plots have a reasonablx tight layout
    fig.subplots_adjust(
        top=0.975, bottom=0.09, left=0.085, right=0.95, hspace=0.1, wspace=0.075
    )

    for ax in axs.flatten():
        ax.tick_params(direction="in", which="both", right=True, top=True)

    markers = ["o", "s"]
    palette = ["#9CA0BF", "#581F18"]
    axs[0, 0].set_ylabel("x$_\\mathdefault{Predicted}$ (m)")
    axs[0, 0].set_yticks(np.arange(-100, 101, 50.0))
    axs[0, 0].set_yticks(np.arange(-125, 126, 5.0), minor=True)
    axs[0, 0].set_ylim([-125, 125])

    axs[0, 0].set_xticks(np.arange(-10, 11, 5.0))
    axs[0, 0].set_xticklabels([""] * 5)
    axs[0, 0].set_xticks(np.arange(-10, 11, 1.0), minor=True)

    axs[0, 0].set_xlim([-10, 10])

    # Adjust such that there are no conflicts with graphical images
    fig.text(0.01, 0.975, "a)")
    fig.text(0.01, 0.505, "b)")
    fig.text(0.51, 0.975, "c)")
    fig.text(0.51, 0.505, "d)")

    plot_data(
        plot_df,
        "x",
        "E_est",
        "E_act",
        "particle",
        markers,
        palette,
        (-11, 11),
        np.arange(-10, 11, 5.0),
        np.arange(-10, 11, 1.0),
        "",
        (-125, 125),
        np.arange(-100, 101, 50.0),
        np.arange(-125, 126, 5.0),
        "E$_\\mathrm{Predicted}$ (kg m$^\\mathdefault{2}$ s$^\\mathdefault{-2}$)",
        axs[0, 0],
    )


    plot_data(
        plot_df,
        "x",
        "F_est",
        "F_act",
        "particle",
        markers,
        palette,
        (-11, 11),
        np.arange(-10, 11, 5.0),
        np.arange(-10, 11, 1.0),
        "x (m)",
        (-25, 25),
        np.arange(-20.0, 20.01, 10.0),
        np.arange(-25.0, 25.01, 1.0),
        "F$_\\mathrm{Predicted}$ (kg m s$^\\mathdefault{-2}$)",
        axs[1, 0],
    )

    plot_data(
        plot_df,
        "F_act",
        "E_est",
        "E_act",
        "particle",
        markers,
        palette,
        (-25, 25),
        np.arange(-20.0, 20.01, 10.0),
        np.arange(-25.0, 25.01, 1.0),
        "",
        (-125, 125),
        np.arange(-100, 101, 50.0),
        np.arange(-125, 126, 5.0),
        "",
        axs[0, 1],
    )


    axs[1, 1].plot(F1_act, F1_est, markers[0], color=palette[0], label="Particle 1")
    axs[1, 1].plot(F2_act, F2_est, markers[1], color=palette[1], label="Particle 2")

    axs[1, 1].plot([-25, 25], [-25, 25], "--", color="#AAAAAA")

    axs[1, 1].set_yticks(np.arange(-20.0, 20.01, 10.0))
    axs[1, 1].set_yticks(np.arange(-25.0, 25.01, 1.0), minor=True)
    axs[1, 1].set_yticklabels([""] * 5)
    axs[1, 1].set_ylim([-25, 25])

    axs[1, 1].set_xticks(np.arange(-20.0, 20.01, 10.0))
    axs[1, 1].set_xticks(np.arange(-25.0, 25.01, 1.0), minor=True)
    axs[1, 1].set_xlim([-25, 25])
    axs[1, 1].set_xlabel("F$_\\mathrm{Actual}$ (kg m s$^\\mathdefault{-2}$)")
    axs[1, 1].legend(frameon=False)
    fig.savefig("fig_x_example_fig.pdf")
