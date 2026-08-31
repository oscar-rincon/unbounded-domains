#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 20:11:57 2017

@author: mraissi
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#mpl.use('pgf')

# def figsize(scale, nplots = 1):
#     fig_width_pt = 390.0                          # Get this from LaTeX using \the\textwidth
#     inches_per_pt = 1.0/72.27                       # Convert pt to inch
#     golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
#     fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
#     fig_height = nplots*fig_width*golden_mean              # height in inches
#     fig_size = [fig_width,fig_height]
#     return fig_size
def figsize(width_scale=1, height_scale=1, nplots=1):
    fig_width_pt = 390.0  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * width_scale  # width in inches
    fig_height = nplots * fig_width * golden_mean * height_scale  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size

# Configuración de LaTeX para matplotlib
pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "xelatex",        # change this if using xetex or lautex
    "text.usetex": False,                # use LaTeX to write all text
    "font.family": "sans-serif",
    # "font.serif": [],
    #"font.sans-serif": ["DejaVu Sans"], # specify the sans-serif font
    "font.monospace": [],
    "axes.labelsize": 8,               # LaTeX default is 10pt font.
    "font.size": 8,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    # "figure.figsize": (3.15, 2.17),     # default fig size of 0.9 textwidth
    "pgf.preamble": r'\usepackage{amsmath},\usepackage{amsthm},\usepackage{amssymb},\usepackage{mathspec},\renewcommand{\familydefault}{\sfdefault},\usepackage[italic]{mathastext}'
    }

mpl.rcParams.update(pgf_with_latex)

# === Forzar uniformidad en etiquetas de ticks ===
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'
mpl.rcParams['axes.edgecolor'] = 'black'

# Force tick labels to use same font object as the rest of the plot
mpl.rcParams['xtick.major.size'] = 3
mpl.rcParams['ytick.major.size'] = 3
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'

# I make my own newfig and savefig functions
def newfig(width,height, nplots = 1):
    fig = plt.figure(figsize=figsize(width, height, nplots))
    ax = fig.add_subplot(111)
    return fig, ax

def savefig(filename, crop = True):
    if crop == True:
        plt.savefig('{}'.format(filename), bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig('{}'.format(filename))

## Simple plot
#fig, ax  = newfig(1.0)
#
#def ema(y, a):
#    s = []
#    s.append(y[0])
#    for t in range(1, len(y)):
#        s.append(a * y[t] + (1-a) * s[t-1])
#    return np.array(s)
#    
#y = [0]*200
#y.extend([20]*(1000-len(y)))
#s = ema(y, 0.01)
#
#ax.plot(s)
#ax.set_xlabel('X Label')
#ax.set_ylabel('EMA')
#
#savefig('ema')


def annotate_final_value(
    ax,
    y,
    label,
    color,
    y_position,
    fmt="{:.2e}",
):
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(y)

    if not np.any(mask):
        return

    y_final = y[mask][-1]

    ax.text(
        0.97,
        y_position,
        f"{label} = {fmt.format(y_final)}",
        transform=ax.transAxes,

        # Same color as curve
        color=color,

        fontsize=6.5,

        # Position
        ha="right",
        va="top",

    )

def plot_histories_comparison(
    histories,
    titles=None,
    save_path=None,
    figsize=(7.5, 5),
):
    """
    Plot a 3x3 comparison of training histories.

    Columns:
        1. Fixed
        2. PDE schedule
        3. PDE schedule + adaptive

    Rows:
        1. Lambda weights
        2. Losses
        3. Errors

    Parameters
    ----------
    histories : list of dict
        List containing exactly three training histories.

    titles : list of str, optional
        Titles for the three columns.

    save_path : str, optional
        Path to save the figure.

    figsize : tuple
        Figure size.
    """

    # ============================================================
    # CHECK INPUT
    # ============================================================

    if len(histories) != 3:
        raise ValueError(
            "histories must contain exactly three history dictionaries."
        )

    h1, h2, h3 = histories

    if titles is None:
        titles = [
            "Fixed",
            r"$\lambda_{\mathrm{PDE}}$ schedule",
            r"$\lambda_{\mathrm{PDE}}$ schedule + adaptive",
        ]

    if len(titles) != 3:
        raise ValueError(
            "titles must contain exactly three titles."
        )

    # ============================================================
    # PALETTE
    # ============================================================

    palette = {
        "u": "#1f4e79",
        "k": "#2e86c1",
        "pde": "#7fb3d5",
        "text_muted": "#7b7d7d",
        "spine": "#aab7b8",
    }

    colors = {
        "u": palette["u"],
        "k": palette["k"],
        "pde": palette["pde"],
    }

    plt.rcParams.update({
        "axes.edgecolor": palette["spine"],
        "axes.labelcolor": "#2c2c2c",
        "xtick.color": palette["text_muted"],
        "ytick.color": palette["text_muted"],
        "text.color": "#2c2c2c",
    })

    # ============================================================
    # FIGURE
    # ============================================================

    fig, axes = plt.subplots(
        3,
        3,
        figsize=figsize,
        sharex="col"
    )

    # ============================================================
    # LOOP OVER HISTORIES / COLUMNS
    # ============================================================

    for col, (history, title) in enumerate(
        zip(histories, titles)
    ):

        # --------------------------------------------------------
        # ITERATIONS
        # --------------------------------------------------------

        iteration = np.asarray(
            history.get(
                "iteration",
                np.arange(len(history.get("total", [])))
            )
        )

        lambda_iteration = np.asarray(
            history.get(
                "lambda_iteration",
                np.arange(
                    len(history.get("lambda_u", []))
                )
            )
        )

        # ========================================================
        # ROW 0: LAMBDAS
        # ========================================================

        ax = axes[0, col]

        lambda_u = history.get("lambda_u", [])
        lambda_k = history.get("lambda_k", [])
        lambda_pde = history.get("lambda_pde", [])

        if len(lambda_u) > 0:
            ax.plot(
                lambda_iteration[:len(lambda_u)],
                lambda_u,
                color=colors["u"],
                label=r"$u$"
            )

        if len(lambda_k) > 0:
            ax.plot(
                lambda_iteration[:len(lambda_k)],
                lambda_k,
                color=colors["k"],
                label=r"$k$"
            )

        if len(lambda_pde) > 0:
            ax.plot(
                lambda_iteration[:len(lambda_pde)],
                lambda_pde,
                color=colors["pde"],
                label=r"PDE"
            )

        ax.set_title(
            title,
            fontsize=8,
            color="#2c2c2c"
        )

        ax.set_ylim(0.8 * 10e-2, 15)
        ax.set_yscale("log")

        # Only first column gets y-label
        if col == 0:
            ax.set_ylabel(
                r"$\lambda$",
                fontsize=8
            )

        # Legend only in first column
        if col == 0:
            ax.legend(
                fontsize=8,
                frameon=False,
                loc="upper left"
            )

        # ========================================================
        # ROW 1: LOSSES
        # ========================================================

        ax = axes[1, col]

        u = np.asarray(
            history.get("u", []),
            dtype=float
        )

        k = np.asarray(
            history.get("k", []),
            dtype=float
        )

        pde = np.asarray(
            history.get("pde", []),
            dtype=float
        )

        # --------------------------------------------------------
        # u loss
        # --------------------------------------------------------

        if len(u) > 0:

            n = min(len(iteration), len(u))

            ax.plot(
                iteration[:n],
                u[:n],
                color=colors["u"],
                linewidth=1.2
            )

        # --------------------------------------------------------
        # k loss
        # --------------------------------------------------------

        if len(k) > 0:

            n = min(len(iteration), len(k))

            ax.plot(
                iteration[:n],
                k[:n],
                color=colors["k"],
                linewidth=1.2
            )

        # --------------------------------------------------------
        # PDE loss
        # --------------------------------------------------------

        if len(pde) > 0:

            n = min(len(iteration), len(pde))

            ax.plot(
                iteration[:n],
                pde[:n],
                color=colors["pde"],
                linewidth=1.2
            )

        # --------------------------------------------------------
        # Fill between u and k
        # --------------------------------------------------------

        if len(u) > 0 and len(k) > 0:

            n = min(
                len(iteration),
                len(u),
                len(k)
            )

            ax.fill_between(
                iteration[:n],
                u[:n],
                k[:n],
                color=colors["k"],
                alpha=0.12
            )

        # --------------------------------------------------------
        # Final values
        # --------------------------------------------------------

        if len(u) > 0:

            annotate_final_value(
                ax,
                u,
                r"$L_u$",
                colors["u"],
                y_position=(
                    0.32 if col == 0
                    else 0.95
                ),
            )

        if len(k) > 0:

            annotate_final_value(
                ax,
                k,
                r"$L_k$",
                colors["k"],
                y_position=(
                    0.22 if col == 0
                    else 0.85
                ),
            )

        if len(pde) > 0:

            annotate_final_value(
                ax,
                pde,
                r"$L_{\mathrm{PDE}}$",
                colors["pde"],
                y_position=(
                    0.12 if col == 0
                    else 0.75
                ),
            )

        ax.set_yscale("log")
        ax.set_ylim(1e-6, 1000)

        if col == 0:
            ax.set_ylabel(
                "Loss",
                fontsize=8
            )

        # ========================================================
        # ROW 2: ERRORS
        # ========================================================

        ax = axes[2, col]

        # --------------------------------------------------------
        # Convert None -> NaN
        # --------------------------------------------------------

        err_u = np.array(
            [
                np.nan if v is None else v
                for v in history.get("error_u", [])
            ],
            dtype=float
        )

        err_k = np.array(
            [
                np.nan if v is None else v
                for v in history.get("error_k", [])
            ],
            dtype=float
        )

        has_errors = (
            err_u.size > 0
            and not np.all(np.isnan(err_u))
        )

        # --------------------------------------------------------
        # Plot errors
        # --------------------------------------------------------

        if has_errors:

            n_u = min(
                len(iteration),
                len(err_u)
            )

            n_k = min(
                len(iteration),
                len(err_k)
            )

            ax.plot(
                iteration[:n_u],
                err_u[:n_u],
                color=colors["u"]
            )

            if err_k.size > 0:

                ax.plot(
                    iteration[:n_k],
                    err_k[:n_k],
                    color=colors["k"]
                )

            ax.set_yscale("log")

            # ----------------------------------------------------
            # Final errors
            # ----------------------------------------------------

            annotate_final_value(
                ax,
                err_u,
                r"$E_u$",
                colors["u"],
                y_position=0.95,
            )

            if err_k.size > 0:

                annotate_final_value(
                    ax,
                    err_k,
                    r"$E_k$",
                    colors["k"],
                    y_position=0.85,
                )

        else:

            ax.text(
                0.5,
                0.5,
                "no analytical error logged",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
                color=palette["text_muted"]
            )

        ax.set_ylim(1e-2, 1e2)

        if col == 0:
            ax.set_ylabel(
                "Error",
                fontsize=8
            )

        ax.set_xlabel(
            "Iteration",
            fontsize=8
        )

    # ============================================================
    # AXIS STYLE
    # ============================================================

    for ax in axes.flat:

        ax.grid(False)

        ax.tick_params(
            axis="both",
            which="both",
            labelsize=8
        )

    # ============================================================
    # LAYOUT
    # ============================================================

    plt.tight_layout()

    if save_path is not None:

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

    return #fig, axes