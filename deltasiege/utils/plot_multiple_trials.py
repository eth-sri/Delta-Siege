import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

from .plot_run import plot_theoretical_boundary, set_box
from .. import Run

def plot_all_trials(
    *runs : Tuple[str, str, Run], 
    x_is_delta : bool,
    state_prefix : str = "final",
    estimaton_prefix : str = "estimation_c",
    border : float = 0.1
):

    # Handle ordering
    ordering = -1 if x_is_delta else 1

    # Initialize plot
    fig, ax = plt.subplots(figsize=(10, 10))
    epsilon_empirical, delta_empirical, initial_epsilon, initial_delta = [], [], [], []
    theoretical_curve = []

    # Iterate over all runs
    for (run, label, marker, c) in runs:

        # Iterate over all experiments
        for _, experiment in enumerate(run.experiments):

            # Iterate over all trials
            for _, trial in enumerate(experiment.trials):

                for _, witness in enumerate(trial.witnesses):
                    for state_name, state in witness.logger.data.items():

                        if not state_prefix.startswith(state_name):
                            continue

                        for estimation_name, estimation in state.items():

                            # Only consider estimations
                            if estimation_name.startswith(estimaton_prefix):

                                # Access the data
                                theoretical_curve.append(np.array(estimation["theoretical_curve"]))
                                initial_epsilon.append(estimation["initial_epsilon"])
                                initial_delta.append(estimation["initial_delta"])
                                epsilon_empirical.append(estimation["empirical_curve"][0][estimation["idx"]])
                                delta_empirical.append(estimation["empirical_curve"][1][estimation["idx"]])

                                # Plot the result
                                values = [epsilon_empirical[-1], delta_empirical[-1]][::ordering]

                                if all(x is not None and np.isfinite(x) for x in values):
                                    ax.scatter(*values, s=100, marker=marker, color=c, zorder=25, label=label)
                                    label = None
            
        if not label is None:
            ax.scatter([], [], s=100, marker=marker, color=c, zorder=25, label=label + "(Only Inifinite)")

    # Check that all
    # assert all(np.isclose(x, initial_epsilon[0]) for x in initial_epsilon)
    # assert all(np.isclose(x, initial_delta[0]) for x in initial_delta)

    # Add 
    x_init, y_init = [initial_epsilon, initial_delta][::ordering]
    x_emp, y_emp = [epsilon_empirical, delta_empirical][::ordering]

    # Compute the theoretical curve
    theoretical_curve = np.concatenate(theoretical_curve, axis=1).tolist()
    theoretical_curve_ = [x[::ordering] for x in zip(*theoretical_curve) if all(y is not None and np.isfinite(y) and y > 0 for y in x)]
    sorted_thoretical = np.array(sorted(theoretical_curve_)).T

    
    # Set boundary
    bounds = set_box(ax, [x_init + x_emp, y_init + y_emp], 0.9, [False, True][::ordering])

    # Plot background
    plot_theoretical_boundary(ax, sorted_thoretical, *bounds[0], *bounds[1], x_is_delta, plot_line=False)

    # Plot 
    ax.hlines(y_init[0], *bounds[0], color="black")
    ax.vlines(x_init[0], *bounds[1], color="black")

    # Use log scale
    if x_is_delta:
        ax.set_xscale("log")
    else:
        ax.set_yscale("log")

    # Set legend
    # fig.legend()

    return fig, ax
