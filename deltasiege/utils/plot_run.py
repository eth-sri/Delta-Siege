import matplotlib.pyplot as plt
from matplotlib import colors as cm
import numpy as np

from .. import Run

def plot_violation(ax, x_empirical, y_empirical, x_init, y_theoretical, sorted_thoretical, sorted_empirical, x_is_delta):

    # Print steps if at least two posiible positions to plot
    lower = min(x_empirical, x_init)
    higher = max(x_empirical, x_init)
    mask = (lower <= sorted_thoretical[0]) & (sorted_thoretical[0] <= higher)

    if np.count_nonzero(mask) >= 2:
        
        # Order the epsilon, delta values as pairs to draw arrows
        ordering = 1 if x_empirical >= x_init else -1
        x_low = sorted_thoretical[0][mask][:-1][::ordering]
        x_high = sorted_thoretical[0][mask][1:][::ordering]
        y_low = sorted_thoretical[1][mask][:-1][::ordering]
        y_high = sorted_thoretical[1][mask][1:][::ordering]

        # Make a color progression
        blue = cm.to_rgba("blue")
        orange = cm.to_rgba("orange")
        inc = tuple(1 / x_low.size * (o - b) for o, b in zip(orange, blue))
        
        # Plot the arrows from the initial condition to the empirical result
        c = blue
        for x0, x1, y0, y1 in zip(x_low, x_high, y_low, y_high):
            ax.plot([x0, x1], [y0, y1], color=c, zorder=20)
            c = tuple(c + i for c, i in zip(c, inc))

    # Annotate if all needed values are finite
    if all(x is not None and np.isfinite(x) for x in [x_empirical, y_empirical, y_theoretical]):

        # Annotate the found difference
        ax.annotate(text="", xy=(x_empirical, y_theoretical), xytext=(x_empirical, y_empirical), arrowprops=dict(arrowstyle='<->'))

        # If x is delta, then log scale is used
        if x_is_delta:
            x = x_empirical * np.exp(0.1)
            y = 0.5 * (y_theoretical + y_empirical)
            name = "\\epsilon"
        else:
            x = x_empirical + 0.1
            y = np.sqrt(y_theoretical * y_empirical)
            name = "\\delta"

        # Annotate the difference between the empirical and theoretical estimates
        ax.text(
            x, y, f"$\\Delta_{name} = {y_theoretical - y_empirical : .2e}$", 
            fontsize = 10, rotation=90,
            horizontalalignment = "center", verticalalignment = "center"
        )

    # Plot the empirical curve
    if len(sorted_empirical) > 0:
        if x_is_delta:
            name = "Empirical $\\epsilon(\\delta)$"
        else:
            name = "Empirical $\\delta(\\epsilon)$"
        ax.plot(sorted_empirical[0], sorted_empirical[1], color="black", label=name)

def plot_theoretical_boundary(ax, sorted_thoretical, x_lb, x_ub, y_lb, y_ub, x_is_delta, plot_line = True):

    # Only plot if enough theoretical data points
    if sorted_thoretical.shape[0] >= 2:

        # Fill in background
        ax.fill_between(x=[x_lb] + sorted_thoretical[0].tolist(), y1=y_lb, y2=[sorted_thoretical[1][0]] + sorted_thoretical[1].tolist(), facecolor='green', alpha=0.5)
        ax.fill_between(x=sorted_thoretical[0].tolist() + [x_ub], y1=sorted_thoretical[1].tolist() + [sorted_thoretical[1][-1]], y2=y_ub, facecolor='red', alpha=0.5)


        # # Plot lower bound
        # if x_lb < sorted_thoretical[0][0]:
        #     ax.fill_between(x=[x_lb, sorted_thoretical[0][0]], y1=y_lb, y2=sorted_thoretical[1][0], facecolor='green', alpha=0.5)
        
        # # Plot upper bound
        # if x_ub > sorted_thoretical[0][-1]:
        #     ax.fill_between(x=[max(sorted_thoretical[0]), x_ub], y1=min(sorted_thoretical[1]), y2=y_ub, facecolor='red', alpha=0.5)

        # Plot lines which were searched along
        if x_is_delta:
            name = "Theoretical $\\epsilon(\\delta)$"
        else:
            name = "Theoretical $\\delta(\\epsilon)$"

        if plot_line:
            ax.plot(sorted_thoretical[0], sorted_thoretical[1], color="purple", linestyle="--", label=name)

def set_box(ax, values, border, is_log):

    bounds = []
    for setter, values_, is_log_ in zip([ax.set_xlim, ax.set_ylim], values, is_log):
        
        # Find values
        values_ = [x for x in values_ if x is not None and np.isfinite(x) and x > 0]
        low, high = min(values_), max(values_)
        
        # Handle log scale by transforming
        if is_log_:
            low, high = np.log(low), np.log(high)

        # Add to border
        width = (1 - border) / border * (high - low)
        low, high = low - width, high + width

        # Handle log scale by transforming back
        if is_log_:
            low, high = np.exp(low), np.exp(high)

        setter(low, high)
        bounds.append([low, high])
    
    return bounds

def make_plot(theoretical_curve, empirical_curve, idx, initial_epsilon, initial_delta, **kwargs):

    # Find most violating point
    delta_theoretical = theoretical_curve[1][idx]
    delta_empirical = empirical_curve[1][idx]

    # If delta_theoretical and delta_empirical are not close, we 
    # assume that the clear violation was tried to estabilsh by finding
    # an empirical epsilon which is larger than the theoretical epsilon
    ordering, x_is_delta = (-1, True) if np.isclose(delta_theoretical, delta_empirical) else (1, False)
    
    # Get all points with the correct ordering
    init = [initial_epsilon, initial_delta][::ordering]
    theoretical = [x[idx] for x in theoretical_curve][::ordering]
    empirical = [x[idx] for x in empirical_curve][::ordering]
    empirical_curve_ = [list(x)[::ordering] for x in zip(*empirical_curve) if all(y is not None and np.isfinite(y) and y >= 0 for y in x)]
    theoretical_curve_ = [list(x)[::ordering] for x in zip(*theoretical_curve) if all(y is not None and np.isfinite(y) and y >= 0 for y in x)]

    # Sort according to epsilon values
    sorted_thoretical = np.array(sorted(theoretical_curve_)).T
    sorted_empirical = np.array(sorted(empirical_curve_)).T

    # Find all valid values
    all_x = [x[0] for x in [init, theoretical, empirical] if x[0] is not None  and np.isfinite(x[0])]
    all_y = [x[1] for x in [init, theoretical, empirical] if x[1] is not None  and np.isfinite(x[1])]

    # Test if any values before adding
    if len(sorted_thoretical) > 0:
          all_x += sorted_thoretical[0].tolist()
          all_y += sorted_thoretical[1].tolist()
    
    if len(sorted_empirical) > 0:
          all_x += sorted_empirical[0].tolist()
          all_y += sorted_empirical[1].tolist()

    # Initialize plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # PLot the violation
    plot_violation(ax, *empirical, init[0], theoretical[1], sorted_thoretical, sorted_empirical, x_is_delta)
    plot_theoretical_boundary(ax, sorted_thoretical, min(all_x), max(all_x), min(all_y), max(all_y), x_is_delta)

    # Plot the initial, theoretical and empirical values
    ax.scatter(*init, s=100, marker="o", color="blue", zorder=25, label="Mechanism Initialization")
    
    if all(x is not None and np.isfinite(x) for x in theoretical):
        ax.scatter(*theoretical, s=100, marker="s", color="orange", zorder=25, label="Theoretical Upper Bound")
    
    if all(x is not None and np.isfinite(x) for x in theoretical):
        ax.scatter(*empirical, s=100, marker="^", color="purple", zorder=25, label="Empirical Worst Violation")

    # Filter out all invalid values when log plotting
    if x_is_delta:
        all_x = [x for x in all_x if x > 0]
        ax.set_xscale("log")
        ax.set_xlabel("$\\delta$")
        ax.set_ylabel("$\\epsilon$")
    else:
        all_y = [x for x in all_y if x > 0]
        ax.set_yscale("log")
        ax.set_xlabel("$\\epsilon$")
        ax.set_ylabel("$\\delta$")

    # Set boundary
    ax.set_xlim([min(all_x), max(all_x)])
    ax.set_ylim([min(all_y), max(all_y)])

    # Set legend
    fig.legend()

    return fig

def plot_run(
    run : Run, 
    estimaton_prefix : str = "estimation",
):

    # Iterate over all experiments
    for e_idx, experiment in enumerate(run.experiments):

        # Iterate over all trials
        for t_idx, trial in enumerate(experiment.trials):

            for w_idx, witness in enumerate(trial.witnesses):
                for state_name, state in witness.logger.data.items():

                    if not isinstance(state, dict):
                        continue

                    for estimation_name, estimation in state.items():

                        # Only consider estimations
                        if estimation_name.startswith(estimaton_prefix):
                            fig = make_plot(**estimation)

                            from pathlib import Path

                            out = Path("") / f"experiment_{e_idx}" / f"trial_{t_idx}" / f"witness_{w_idx}" / state_name / estimation_name
                            out.mkdir(exist_ok=True, parents=True)
                            fig.savefig(out / "plot.png", dpi=300)
                            print(out)
