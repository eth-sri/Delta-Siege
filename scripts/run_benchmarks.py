import argparse
from itertools import product
from deltasiege.mechanism.pgm_mechansims import PGMMechanism
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import torch
import random

from deltasiege.utils import Config
from deltasiege import DeltaSiege
from deltasiege.mechanism import *
from deltasiege.ddsampler import SimpleSampler
from deltasiege.classifiers import *
from deltasiege.attack import *
from deltasiege.logging.logger import Logger
from deltasiege.classifiers.legacy.multi_layer_perceptron import MultiLayerPerceptron
from deltasiege.classifiers.legacy.torch_optimizer_factory import AdamOptimizerFactory

from deltasiege.utils.plot_multiple_trials import plot_all_trials

mechs = {
    "laplace_inversion": LaplaceInversionMechanism,
    "laplace_opendp": LaplaceOpenDPMechanism,
    "laplace_ibm": LaplaceIBMMechanism,
    "laplace_pydp": LaplacePyDPDPMechanism,
    "gauss_opendp": GaussOpenDPMechanism,
    "gauss_ibm": GaussIBMMechanism,
    "gauss_ibm_analytic": GaussIBMAnalyticalMechanism,
    "gauss_ibm_discrete": GaussIBMDiscreteMechanism,
    "gauss_pydp": GaussPyDPMechanism,
    "gauss_opacus": GaussOpacusMechanism,
    "gauss_ziggurat": GaussZigguratMechanism,
    "gauss_polar": GaussPolarMechanism,
    "gauss_boxmuller": GaussBoxMullerMechanism,
    "aim_internal": AIMInternalMechanism,
    "mst_internal": MSTInternalMechanism,
    "aim": AIMMechanism,
    "mst": MSTMechanism,
}


def run(n : int, epsilon : float, delta : float, mechanism : str, idx : int, out_dir : Path, plot: bool, n_experiments : int):

    # Set up the out directory
    out_dir = out_dir / f"{mechanism}_{epsilon}_{delta}_{n}_{idx}"

    # Set up search space
    # The DP synthetic data mechanisms need a different setup than the noise adding mechanisms
    if mechanism in ["aim", "aim_internal", "mst", "mst_internal"]:
        kwargs = {"input_pairs": [(
            (Path("data/data0.csv"), "data/data-domain.json"), 
            (Path("data/data1.csv"), "data/data-domain.json")
        )]}
        n_steps = 90

    else:
        kwargs = {"input_pairs": [(0.0, 1.0)], "sensitivity": (1.0, 1.0)}
        n_steps = 900

    
    # Choose the Range for the delta parameter
    if delta is None:
        delta = 0.0
    else:
        kwargs["delta"] = (delta, delta)
    
    if "ibm" in mechanism:
        kwargs["fixed_kwargs"] = {"seed": idx}

    # Use the parameters to set up the search space
    search_space = SimpleSampler(
        mechs[mechanism],
        epsilon = (epsilon, epsilon),
        **kwargs
    )

    # Set up a config
    config = Config(
        n_train=n, n_init=n, 
        n_check=n, n_final=n,
        n_experiments=n_experiments,
        batch_size=n // 10,
        n_jobs=1,
        seed=idx
    )

    # Set up classifier
    if any(issubclass(mechs[mechanism], x) for x in [AIMMechanism, MSTMechanism]):
        kwargs = {"in_dimensions": 64 * 2 ** 10}        
    if any(issubclass(mechs[mechanism], x) for x in [AIMInternalMechanism, MSTInternalMechanism]):
        kwargs = {"in_dimensions": 64 * 10}
    else:
        kwargs = {"in_dimensions": 64}

    classifier_factory = MultiLayerPerceptron.get_factory(
        optimizer_factory=AdamOptimizerFactory(learning_rate=0.1, step_size=500),
        hidden_sizes=(),
        n_test_batches=1,
        normalize_input=False,
        epochs=10, 
        regularization_weight=0.0001,
        feature_transform=BitPatternFeatureTransformer(),
        **kwargs
    )

    # Set up the search
    deltasiege = DeltaSiege(
        config = config, 
        search_space = search_space, 
        classifier_factories = classifier_factory,
        logger = Logger(Logger.State.SILENT),  # Surpresses  all outputs. Can be VERBOSE, SILENT, DOWN. 
        base_folder = out_dir,
        classifier_folder = out_dir / "shared", # Store classifier in seperate folder to share between runs
        mechanism_folder= out_dir / "shared"    # Store mechanism in seperate folder to share between runs
    )

    # Set up full estimator for Delta Siege
    delta_steps = np.concatenate([
        np.zeros(1),
        delta * np.ones(1),
        np.exp(np.linspace(np.log(1e-9), 0, num = n_steps))
    ], axis=0)
    estimator = LinesearchEstimator(delta_steps=delta_steps)

    # Make factory
    w_factory = WitnessOptimization.get_factory(estimator=estimator)
    run1 = deltasiege.run("deltasiege", witnesses_factories=w_factory)

    # Fixed delta
    estimator = LinesearchEstimator(delta_steps=np.array([delta]))
    w_factory = WitnessDPSniper.get_factory(estimator=estimator, fixed_delta=delta)
    run2 = deltasiege.run("dpsniper", witnesses_factories=w_factory)

    # DP-Opt seeks to establish a witness by maximizing epsilon while keeping delta fixed
    estimator = LinesearchEstimator(delta_steps=np.array([delta]))
    w_factory = WitnessDPOpt.get_factory(estimator=estimator, fixed_delta=delta)
    run3 = deltasiege.run("dpopt", witnesses_factories=w_factory)

    if plot:
        x_is_delta=False
        fig, ax = plot_all_trials(
            (run1, "DeltaSiege", "o", "blue"),
            (run2, "DP-Sniper", "^", "orange"),
            (run3, "DP-Opt", "x", "purple"),
            x_is_delta=x_is_delta,
        )

        # Plot the results
        ax.axis("off")
        if x_is_delta:
            ax.set_xticks([delta]) 
            ax.set_xticklabels(["$\\delta_0$"], fontsize=20)
            ax.set_yticks([epsilon]) 
            ax.set_yticklabels(["$\\epsilon_0$"], fontsize=20)
        else:
            ax.set_xticks([epsilon]) 
            ax.set_xticklabels(["$\\epsilon_0$"], fontsize=20)
            ax.set_yticks([delta]) 
            ax.set_yticklabels(["$\\delta_0$"], fontsize=20)
        fig.legend()
        fig.savefig(out_dir / "plot.png", dpi=300)


if __name__ == "__main__":

    # Parse inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism", type=str, choices=list(mechs.keys()), help=f"The mechanisms to run {list(mechs.keys())}")
    parser.add_argument("--epsilon", type=float, nargs="+", default=None, help="The epsilon values to use")
    parser.add_argument("--delta", type=float, nargs="+", default=None, help="The delta values to use")
    parser.add_argument("--n_runs", type=int, default=10, help="Number of reruns for each tasks")
    parser.add_argument("--procs", type=int, default=40, help="Number of tasks to run in paralell")
    parser.add_argument("--out_dir", type=Path, default=Path("experiments"), help="Output dir")
    parser.add_argument("--n", type=int, default=None, help="Number samples")
    parser.add_argument("--plot", help="Plot the results", action="store_true")
    parser.add_argument("--n_experiments", type=int, default=1, help="Number experiments per job")
    args = parser.parse_args()

    # Set the default values for n
    if args.n is None:
        if any(x in args.mechanism for x in ["aim", "mst"]):
            args.n = 25_000
        else:
            args.n = 1_000_000

    # Set the default values for delta
    if args.delta is None:
        if issubclass(mechs[args.mechanism], LaplaceMechanism):
            args.delta = [None]
        elif issubclass(mechs[args.mechanism], PGMMechanism):
            args.delta = [1e-6, 0.1]
        else:
            args.delta = [1e-6, 1e-3, 0.1]
    
    # Set the default values for epsilon
    if args.epsilon is None:
      if issubclass(mechs[args.mechanism], GaussIBMMechanism) or issubclass(mechs[args.mechanism], GaussMechanismClassic):
          args.epsilon = [0.1, 1.0]
      elif issubclass(mechs[args.mechanism], PGMMechanism):
            args.epsilon = [3.0]
      else:
          args.epsilon = [0.1, 1.0, 3.0, 10.0]

    # Set up arguments
    arguments = list(product(*([args.n], args.epsilon, args.delta, [args.mechanism], range(args.n_runs), [args.out_dir], [args.plot], [args.n_experiments])))

    # Run all experiments
    if args.procs > 1:
        pool = Pool(args.procs)
        pool.starmap(run, arguments)
        pool.close()

    else:
        for x in arguments:
            run(*x)
