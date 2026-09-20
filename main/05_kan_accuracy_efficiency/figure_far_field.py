"""Far-field prediction, error maps, and square-ring MAE figure."""

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def plot_far_field(
    U_pred_mlp,
    U_pred_kan,
    U,
    x,
    y,
    palette,
    output_dir="figures",
    u_vmin=-1.0,
    u_vmax=1.0,
    u_n_ticks=5,
    u_tick_decimals=2,
    error_vmin=0.0,
    error_vmax=None,
    error_n_ticks=5,
    error_tick_decimals=3,
    mae_decimals=3,
):
    """Render and save the far-field comparison figure.

    error_vmax=None auto-scales to the max error visible in the plotted window.
    u_n_ticks / error_n_ticks control how many ticks each colorbar shows.
    """
    error_mlp = np.abs(U_pred_mlp - U)
    error_kan = np.abs(U_pred_kan - U)

    r_square = np.maximum(np.abs(np.meshgrid(x, y)[0]), np.abs(np.meshgrid(x, y)[1]))
    ring_edges = np.arange(0, 101, 10)
    ring_labels = [f"{ring_edges[i]}–{ring_edges[i + 1]}" for i in range(len(ring_edges) - 1)]
    mae_mlp = []
    mae_kan = []
    for index in range(len(ring_edges) - 1):
        lower, upper = ring_edges[index:index + 2]
        mask = ((r_square >= lower) & (r_square <= upper) if index == len(ring_edges) - 2
                else (r_square >= lower) & (r_square < upper))
        mae_mlp.append(np.mean(error_mlp[mask]))
        mae_kan.append(np.mean(error_kan[mask]))

    mae_mlp = np.asarray(mae_mlp)
    mae_kan = np.asarray(mae_kan)
    print("\n========================================")
    print("Square-ring MAE")
    print("========================================")
    for label, mlp_value, kan_value in zip(ring_labels, mae_mlp, mae_kan):
        print(f"{label:>6} : MLP = {mlp_value:.{mae_decimals}e}    KAN = {kan_value:.{mae_decimals}e}")

    plot_xmin, plot_xmax = -10.0, 10.0
    plot_ymin, plot_ymax = -10.0, 10.0
    x_mask = (x >= plot_xmin) & (x <= plot_xmax)
    y_mask = (y >= plot_ymin) & (y <= plot_ymax)
    maps = [
        U_pred_mlp[np.ix_(y_mask, x_mask)],
        error_mlp[np.ix_(y_mask, x_mask)],
        U_pred_kan[np.ix_(y_mask, x_mask)],
        error_kan[np.ix_(y_mask, x_mask)],
    ]

    fig = plt.figure(figsize=(6.2, 3.8))
    grid = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.40], hspace=0.30, wspace=0.40)
    axes = [fig.add_subplot(grid[0, index]) for index in range(4)]
    bar_axis = fig.add_subplot(grid[1, :])
    extent = [plot_xmin, plot_xmax, plot_ymin, plot_ymax]
    prediction_images = [
        axes[0].imshow(maps[0], extent=extent, origin="lower", cmap="RdBu_r", vmin=u_vmin, vmax=u_vmax, aspect="equal"),
        axes[2].imshow(maps[2], extent=extent, origin="lower", cmap="RdBu_r", vmin=u_vmin, vmax=u_vmax, aspect="equal"),
    ]
    # Scale the error colorbar to the errors actually visible in [-10, 10], not the far-field range.
    if error_vmax is None:
        error_vmax = max(np.nanmax(maps[1]), np.nanmax(maps[3]))
    error_images = [
        axes[1].imshow(maps[1], extent=extent, origin="lower", cmap="magma", vmin=error_vmin, vmax=error_vmax, aspect="equal"),
        axes[3].imshow(maps[3], extent=extent, origin="lower", cmap="magma", vmin=error_vmin, vmax=error_vmax, aspect="equal"),
    ]

    for axis in axes:
        axis.add_patch(plt.Rectangle((-5.0, -5.0), 10.0, 10.0, fill=False,
                                     edgecolor="#BDBDBD", linewidth=0.8))
        axis.set_xlim(plot_xmin, plot_xmax)
        axis.set_ylim(plot_ymin, plot_ymax)
        axis.set_xticks([-10, -5, 5, 10])
        axis.set_yticks([-10, -5, 5, 10])
        axis.tick_params(axis="both", labelsize=6, colors="gray", length=2, width=0.5)
        for spine in axis.spines.values():
            spine.set_color("gray")
            spine.set_linewidth(0.5)

    axes[0].text(1.2, 1.12, "MLP", transform=axes[0].transAxes, ha="center", va="bottom", fontsize=7)
    axes[2].text(1.2, 1.12, "KAN", transform=axes[2].transAxes, ha="center", va="bottom", fontsize=7)
    for image, axis, label, n_ticks, vmin, vmax in zip(
        prediction_images + error_images,
        [axes[0], axes[2], axes[1], axes[3]],
        [r"$\hat{u}$", r"$\hat{u}$", r"$|\hat{u}-u|$", r"$|\hat{u}-u|$"],
        [u_n_ticks, u_n_ticks, error_n_ticks, error_n_ticks],
        [u_vmin, u_vmin, error_vmin, error_vmin],
        [u_vmax, u_vmax, error_vmax, error_vmax],
    ):
        colorbar = fig.colorbar(image, ax=axis, orientation="horizontal", fraction=0.045, pad=0.17, aspect=25)
        tick_values = np.linspace(vmin, vmax, n_ticks)
        colorbar.set_ticks(tick_values)
        if label == r"$\hat{u}$":
            formatter = ticker.FormatStrFormatter(f"%.{u_tick_decimals}f")
        else:
            formatter = ticker.FormatStrFormatter(f"%.{error_tick_decimals}f")
        colorbar.ax.xaxis.set_major_formatter(formatter)
        colorbar.set_label(label, fontsize=7)
        colorbar.ax.tick_params(axis="x", labelsize=6, colors="gray", length=2, width=0.5)
        colorbar.outline.set_edgecolor("gray")
        colorbar.outline.set_linewidth(0.5)

    positions = np.arange(len(ring_labels))
    width = 0.40
    bar_axis.bar(positions - width / 2, mae_mlp, width, label="MLP", color=palette["mlp"], alpha=0.4)
    bar_axis.bar(positions + width / 2, mae_kan, width, label="KAN", color=palette["kan"], alpha=0.4)
    bar_axis.set_yscale("log")
    bar_axis.set_ylabel("MAE", fontsize=7)
    bar_axis.set_xlabel("Square-ring interval", fontsize=7)
    bar_axis.set_xticks(positions)
    bar_axis.set_xticklabels(ring_labels, fontsize=6)
    bar_axis.tick_params(axis="both", which="major", labelsize=6, colors="gray", length=2, width=0.5)
    bar_axis.tick_params(axis="y", which="minor", colors="gray")
    bar_axis.yaxis.get_offset_text().set_color("gray")
    for spine in bar_axis.spines.values():
        spine.set_color("gray")
        spine.set_linewidth(0.5)
    bar_axis.spines["top"].set_visible(False)
    bar_axis.spines["right"].set_visible(False)
    bar_axis.legend(fontsize=6, frameon=False, loc="upper left")

    plt.subplots_adjust(left=0.08, right=0.99, bottom=0.12, top=0.90)
    os.makedirs(output_dir, exist_ok=True)
    svg_path = os.path.join(output_dir, "far_field.svg")
    pdf_path = os.path.join(output_dir, "far_field.pdf")
    fig.savefig(svg_path, bbox_inches="tight", dpi=300)
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    plt.show()
    print(f"Figure successfully saved to {svg_path}")
    return fig, mae_mlp, mae_kan
