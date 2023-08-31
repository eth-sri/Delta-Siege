import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from run_benchmarks import mechs

from deltasiege import Trial

witness_key = {"deltasiege": "DeltaSiege", "dpsniper": "DP-Sniper", "dpopt": "DP-Opt"}
witness = list(witness_key.keys())

def get_data(base_dir, n_target, mech_target, eps_target, delta_target):

    full_results = {w: [] for w in witness}
    for f in base_dir.glob(f"*"):
        for w in witness:
            
            # stats
            epsilon, delta, n, idx = f.name.split("_")[-4:]
            mech = "_".join(f.name.split("_")[:-4])

            if int(n_target) != int(n) or mech_target != mech or float(eps_target) != float(epsilon) or float(delta_target) != float(delta):
                continue

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
                
            full_results[w].append(results["initial_guarantee"] / results["empirical_guarantee"][results["idx"]])

    return full_results

def get_plot(data, n_samples, factor):

    # prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = ["blue", "purple", "orange"]
    

    fig, ax = plt.subplots()

    for c, (w, values) in zip(colors, data.items()):
        # print(values)
        k = len(values[0])
        for i in range(k):
            # print([v[i] for v in values])
            ax.plot(n_samples, [v[i] for v in values], color=c, alpha=0.1, linestyle="-.")
        ax.plot(n_samples, [np.median(v) for v in values], label=witness_key[w], color=c,)
    
    ax.hlines(1, n_samples[0], n_samples[-1], linestyles="--")
    ax.set_xscale("log")
    ax.set_xticks(n_samples)
    ax.set_xticklabels([f"{n} ({factor * n : .1f}h)" for n in n_samples], rotation=45)
    ax.set_xlabel("Sample size (CPU Hours)")
    ax.set_yscale("log")
    # ax.set_ylim([0.1, 10**4])
    ax.set_ylabel("$\\mu$")
    ax.legend()
    plt.tight_layout()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism", type=str, default="mst_internal", choices=list(mechs.keys()), help=f"The mechanisms to run {list(mechs.keys())}")
    parser.add_argument("--epsilon", type=float, nargs="+", default=[3.0], help="The epsilon values to use")
    parser.add_argument("--delta", type=float, nargs="+", default=[1e-6], help="The delta values to use")
    parser.add_argument("--out_dir", type=Path, default=Path("."), help="Output dir")
    parser.add_argument("--n", type=int, nargs="+", default=[25_000, 12500, 5_000, 2_500, 1_250, 500], help="Number samples")
    parser.add_argument("--factor", type=float, default=0.000472, help="Runtime in hours to create one sample")
    args = parser.parse_args()

    base_dir = args.out_dir

    for eps in args.epsilon:
        for delta in args.delta:
            data = {
              w: [
                  get_data(base_dir, f"{n}", args.mechanism, str(eps), str(delta))[w] 
                  for n in args.n
              ]
              for w in witness
            }
            get_plot(data, args.n, args.factor)
            plt.savefig(f"{args.mechanism}_{eps}_{delta}.png")
