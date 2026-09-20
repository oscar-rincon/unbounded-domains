"""Accuracy and efficiency trade-off figure."""

import os

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, StrMethodFormatter
 

def plot_tradeoff(
    df_mlp,
    df_kan,
    output_dir="figures",
    target_error=1e-2,
):
    """Render and save the accuracy/efficiency comparison figure."""

    # --------------------------------------------------
    # Palette
    # --------------------------------------------------
    palette = {
        "kan": "#6BAED6",
        "mlp": "#999999",
        "kan_highlight": "#1F77B4",
        "mlp_highlight": "#333333",
        "text_muted": "#000000",
        "tick": "#808080",
        "spine": "#808080",
        "target_line": "#A6A6A6",
    }

    # --------------------------------------------------
    # Prepare data
    # --------------------------------------------------
    df_mlp = df_mlp.copy()
    df_kan = df_kan.copy()

    for frame in (df_mlp, df_kan):

        frame["neurons"] = (
            frame["hidden_layers"]
            * frame["hidden_units"]
        )

        frame["err_global_mean"] = frame[
            [
                "err_u_global",
                "err_k_global",
            ]
        ].mean(axis=1)

    # --------------------------------------------------
    # Figure layout
    # --------------------------------------------------
    fig = plt.figure(
        figsize=(6.3, 3.2),
        constrained_layout=False,
        facecolor="#FFFFFF",
    )

    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.2, 0.8],
        hspace=0.75,
    )

    top_grid = grid[0].subgridspec(1, 3, wspace=0.35)
    bottom_grid = grid[1].subgridspec(1, 4, wspace=0.7)

    axes = [
        fig.add_subplot(top_grid[0, 0]),
        fig.add_subplot(top_grid[0, 1]),
        fig.add_subplot(top_grid[0, 2]),
        fig.add_subplot(bottom_grid[0, 0]),
        fig.add_subplot(bottom_grid[0, 1]),
        fig.add_subplot(bottom_grid[0, 2]),
        fig.add_subplot(bottom_grid[0, 3]),
    ]

    plt.subplots_adjust(
        left=0.07,
        right=0.99,
        bottom=0.16,
        top=0.96,
    )

    # --------------------------------------------------
    # Accuracy plots
    # --------------------------------------------------
    def plot_accuracy(
        ax,
        x_col,
        x_label,
        show_ylabel,
        show_legend,
    ):

        for frame, color, marker, name in (
            (
                df_mlp.sort_values(x_col),
                palette["mlp"],
                "s",
                "MLP",
            ),
            (
                df_kan.sort_values(x_col),
                palette["kan"],
                "o",
                "KAN",
            ),
        ):

            # Plot accuracy curve
            ax.plot(
                frame[x_col],
                frame["err_global_mean"],
                marker=marker,
                linestyle="-",
                color=color,
                lw=1.2,
                ms=3.5,
                label=name,
            )

            if frame.empty:
                continue

            # Find first configuration reaching target error.
            # If none reaches the target, select the closest one.
            achieved = frame[
                frame["err_global_mean"] <= target_error
            ]

            best = (
                achieved.iloc[0]
                if not achieved.empty
                else frame.loc[
                    (
                        frame["err_global_mean"]
                        - target_error
                    ).abs().idxmin()
                ]
            )

 

        # Target-error line
        ax.axhline(
            target_error,
            color=palette["target_line"],
            linestyle="--",
            linewidth=0.6,
            alpha=0.7,
        )

        # Logarithmic y-axis
        ax.set_yscale("log")

        # Use regular FLOP ticks and include 140,000 at the right edge.
        if x_label == "FLOPs":
            ax.set_xlim(0, 140000)
            ax.xaxis.set_major_locator(FixedLocator([0, 70000, 140000]))
            ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

        # X-axis title: BLACK
        ax.set_xlabel(
            x_label,
            fontsize=7,
            color="black",
        )

        # Y-axis title: BLACK
        if show_ylabel:
            ax.set_ylabel(
                "MAE",
                fontsize=7,
                color="black",
            )

        # Legend
        if show_legend:
            ax.legend(
                frameon=False,
                fontsize=6,
                loc="upper right",
            )

    # --------------------------------------------------
    # Accuracy panels
    # --------------------------------------------------
    plot_accuracy(
        axes[0],
        "neurons",
        "Number of Neurons",
        True,
        False,
    )

    plot_accuracy(
        axes[1],
        "parameters",
        "Number of Parameters",
        False,
        False,
    )

    plot_accuracy(
        axes[2],
        "flop_counter_flops",
        "FLOPs",
        False,
        True,
    )

    # --------------------------------------------------
    # Efficiency plots
    # --------------------------------------------------
    models = [
        "MLP",
        "KAN",
    ]

    colors = [
        palette["mlp"],
        palette["kan"],
    ]

    measurements = [
        (
            "training_time_sec",
            "Training (s)",
            "{:.1f}s",
        ),
        (
            "evaluation_time_ms",
            "Evaluation (ms)",
            "{:.2f}ms",
        ),
        (
            "first_derivative_time_ms",
            r"$\partial u / \partial x$ (ms)",
            "{:.2f}ms",
        ),
        (
            "second_derivative_time_ms",
            r"$\partial^2 u / \partial x^2$ (ms)",
            "{:.2f}ms",
        ),
    ]

    for ax, (
        column,
        ylabel,
        value_format,
    ) in zip(
        axes[3:],
        measurements,
    ):

        values = [
            df_mlp[column].mean(),
            df_kan[column].mean(),
        ]

        # Bar plot
        bars = ax.bar(
            models,
            values,
            color=colors,
            width=0.65,
            alpha=0.4,
        )

        # Values above bars
        for bar in bars:

            ax.annotate(
                value_format.format(
                    bar.get_height()
                ),
                (
                    bar.get_x()
                    + bar.get_width() / 2,
                    bar.get_height(),
                ),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6,
                color=palette["target_line"],
            )

        # Y-axis limits
        ax.set_ylim(
            0,
            max(values) * 1.2,
        )

        # Y-axis title: BLACK
        ax.set_ylabel(
            ylabel,
            fontsize=7,
            color="black",
        )

 

    # --------------------------------------------------
    # Global formatting
    # --------------------------------------------------
    for ax in axes:

        # Tick labels: GRAY
        ax.tick_params(
            axis="both",
            labelsize=6,
            colors=palette["tick"],
        )

        # Minor y-axis ticks: GRAY
        ax.tick_params(
            axis="y",
            which="minor",
            colors=palette["tick"],
        )

        # Remove top/right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Bottom/left spines
        for spine in (
            ax.spines["bottom"],
            ax.spines["left"],
        ):
            spine.set_color(
                palette["spine"]
            )
            spine.set_linewidth(0.5)

    # --------------------------------------------------
    # Bottom-row x tick labels: BLACK
    # --------------------------------------------------
    for ax in axes[3:]:
        ax.tick_params(
            axis="x",
            labelcolor="black",
        )

    # --------------------------------------------------
    # Save figure
    # --------------------------------------------------
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    svg_path = os.path.join(
        output_dir,
        "tradeoff.svg",
    )

    pdf_path = os.path.join(
        output_dir,
        "tradeoff.pdf",
    )

    fig.savefig(
        svg_path,
        bbox_inches="tight",
        dpi=300,
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        dpi=300,
    )

    plt.show()
 

    return   