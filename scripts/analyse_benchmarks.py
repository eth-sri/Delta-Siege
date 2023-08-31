import argparse
import sys
from pathlib import Path
import numpy as np

from deltasiege import Trial

witness = ["deltasiege", "dpsniper", "dpopt"]

def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str

def make_table(base_dir : Path):

    full_results = {}

    for f in base_dir.glob("*"):

        for w in witness:

            # Load data from files
            try:
                

                trial = Trial(base_folder = f / w / "experiment_0/trial_0/")
                w_ = trial.witnesses[trial.best_witness_idx]
                
                epsilon, delta = w_.mechanism.epsilon, w_.mechanism.delta
                name = w_.mechanism.name
                
                results = w_.logger.data["final"]["estimation_c"]

            except Exception as e:
                print(f"Exception occured in the folder {f} please delete and re-run the experiment. The following exception occured:", file=sys. stderr)
                print(e, file=sys.stderr)
                continue

            # Ensure right structure
            if (epsilon, delta) not in full_results:
                full_results[(epsilon, delta)] = {}
            
            if name not in full_results[(epsilon, delta)]:
                full_results[(epsilon, delta)][name] = {}

            if w not in full_results[(epsilon, delta)][name]:
                full_results[(epsilon, delta)][name][w] = {"rho": [], "eps": [], "delta": [], "eps_h": [], "delta_h": []}
            
            # Add data
            e, d = np.array(results["theoretical_curve"])[:, results["idx"]]
            full_results[(epsilon, delta)][name][w]["eps"].append(e)
            full_results[(epsilon, delta)][name][w]["delta"].append(d)

            e, d = np.array(results["empirical_curve"])[:, results["idx"]]
            full_results[(epsilon, delta)][name][w]["eps_h"].append(e)
            full_results[(epsilon, delta)][name][w]["delta_h"].append(d)

            full_results[(epsilon, delta)][name][w]["rho"].append(results["initial_guarantee"] / results["empirical_guarantee"][results["idx"]])

    top = "\\begin{table*}[hbt!]" \
          "\n\\centering" \
          "\n\\footnotesize" \
          "\\caption{#1} \n" \
          "\\label{table:results#2}" \
          "\n\\begin{tabular}{l|lll|lll|lll}" \
          "\n\\textbf{Method [Implementation]} & \multicolumn{3}{l}{Delta-Siege} & \multicolumn{3}{|l}{DP-Sniper} & \multicolumn{3}{|l}{DP-Opt} \\\\ \\toprule" \
          "\n& \#  violations & $\mu_{\\text{Delta-Siege}}$ & \\begin{tabular}{@{}c@{}}$(\\epsilon_1, \\delta_1)$ \\\\ $(\\hat{\\epsilon}_1, \\hat{\\delta}_1)$\end{tabular} & \# violations & $\\mu_{\\text{DP-Sniper}}$ & \\begin{tabular}{@{}c@{}}$(\\epsilon_1, \\delta_1)$ \\\\ $(\\hat{\\epsilon}_1, \\hat{\\delta}_1)$\end{tabular} & \# violations & $\mu_{\\text{DP-Opt}}$ & \\begin{tabular}{@{}c@{}}$(\\epsilon_1, \\delta_1)$ \\\\ $(\\hat{\\epsilon}_1, \\hat{\\delta}_1)$\end{tabular} \\\\ \\toprule"

    bottom = "\end{tabular} \n" \
            "\\end{table*}"

    for (epsilon, delta), v1 in sorted(full_results.items(), key=lambda x: x[0]):

        
        curr_top = top.replace(
            "#1", f"Results for $\\epsilon_0 = {epsilon}$ and $\\delta_0 = {latex_float(delta)}$"
        ).replace(
            "#2", f"_e{epsilon}_d{delta}"
        )
        print(curr_top)

        for name, v2 in sorted(v1.items(), key=lambda x: x[0]):

            line = []
            for w in witness:
                line.append(f"{len([x for x in v2[w]['rho'] if x > 1])} / {len(v2[w]['rho'])}")

                x = f"{np.median(v2[w]['rho']).item() : .3f}"
                x = x.replace("inf", "\\infty")

                line.append(x)

                eps = np.median(v2[w]["eps"]).item()
                delta = np.median(v2[w]["delta"]).item()
                
                try:
                    eps_h = np.median(v2[w]["eps_h"]).item()
                except:
                    eps_h = float("inf")

                try:
                    delta_h = np.median(v2[w]["delta_h"]).item()
                except:
                    delta_h = 1

                line.append("\\begin{tabular}{@{}c@{}}$" + f"({eps: .3f}, {latex_float(delta)})$ \\\\ $({eps_h: .3f}, {latex_float(delta_h)})".replace("inf", "\\infty") + "$\end{tabular}")

            print(f"{name} & {' & '.join(line)} \\\\")

        print(bottom, end="\n\n\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=Path("experiments"), help="Output dir from running benchmarks")
    args = parser.parse_args()

    make_table(args.out_dir)
