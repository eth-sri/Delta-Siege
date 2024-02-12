# Group and Attack: Auditing Differential Privacy <a href="https://www.sri.inf.ethz.ch/"><img width="100" alt="portfolio_view" align="right" src="http://safeai.ethz.ch/img/sri-logo.svg"></a>

In this repository, we provide the code accompanying our ACM CCS'23 paper: [**Group and Attack: Auditing Differential Privacy**](https://www.sri.inf.ethz.ch/publications/lokna2023groupandattack).

## Installation and Environment

We require Python 3.8 or newer installed on the system. You can then install our prerequisites using [pip](https://pypi.org/project/pip/) and the ```requirements.txt``` supplied. 
Below, we show how to install Python and our prerequisites via [Conda](https://docs.conda.io/en/latest/):

```bash
git clone https://github.com/eth-sri/Delta-Siege.git
cd Delta-Siege
conda create --name deltasiege --yes python=3.8
conda activate deltasiege
pip install -r requirements.txt
```

To run some of the experiments in the paper, the code for the DP mechanisms we audit needs to be downloaded from their respective external repositories, as described below:

```bash
cd deltasiege/mechanism
git clone https://github.com/barryZZJ/dp-opt.git dp-opt ; cd dp-opt ; git checkout cc3bf7e7de7c62722133cb40587d404a0fd124f1 ; cd -
git clone https://github.com/ryan112358/private-pgm.git private-pgm ; cd private-pgm ; git checkout 5b9126295c110b741e5426ddbff419ea1e60e788 ; cd -
git clone https://github.com/dpcomp-org/hdmm.git hdmm ; cd hdmm ; git checkout 7a5079a7d4f1a06b0be78019adadf83c538d0514 ; cd -
cd ../../

export PYTHONPATH=$PYTHONPATH:$(pwd)/deltasiege/mechanism/private-pgm/:$(pwd)/deltasiege/mechanism/hdmm/src/:$(pwd)/deltasiege/mechanism/dp-opt/:$(pwd)/deltasiege/mechanism/private-pgm/src/:$(pwd)/deltasiege/mechanism/private-pgm/mechanisms/:$(pwd)/deltasiege/mechanism/private-pgm/:$(pwd)/deltasiege/mechanism/hdmm/src/
```

**Note:** After the installation, you need to call the code below in every new terminal you intend to use:
```bash
cd Delta-Siege
conda activate deltasiege
export PYTHONPATH=$PYTHONPATH:$(pwd)/deltasiege/mechanism/private-pgm/:$(pwd)/deltasiege/mechanism/hdmm/src/:$(pwd)/deltasiege/mechanism/dp-opt/:$(pwd)/deltasiege/mechanism/private-pgm/src/:$(pwd)/deltasiege/mechanism/private-pgm/mechanisms/:$(pwd)/deltasiege/mechanism/private-pgm/:$(pwd)/deltasiege/mechanism/hdmm/src/
```

We strongly recommend using a Linux-based system to execute our code, as we thoroughly tested the code in this environment. However, to the best of our knowledge, other operating systems should also work, as all of our code is written in Python. 
All experiments in the paper were conducted using less than 1GB of RAM on a single core from AMD EPYC 7742 CPU, with a clock speed of 2250MHz and a total of 64 cores. The machine used was running CentOS Linux 7.

Computational requirements vary depending on the audited DP-mechanism and chosen machine learning model to learn $\tilde{p}$ from Algorithm 2 in the paper.

## Confirming Your Installation 

To confirm your installation run:

```bash
python scripts/run_benchmarks.py --mechanism gauss_opacus --epsilon 0.1 --delta 0.1 --n 100
```

Your output should look like, as follows:

```bash
[I 2023-10-24 15:34:42,883] A new study created in memory with name: no-name-179c5b05-ff46-4097-bc92-c32ea09aaa35
[I 2023-10-24 15:34:42,923] A new study created in memory with name: no-name-e313c050-8304-45e9-bdb0-9e94b844f4f8
[I 2023-10-24 15:34:42,902] A new study created in memory with name: no-name-0f1cd693-8d53-45cf-a335-f44c069ec86f
[I 2023-10-24 15:34:42,903] A new study created in memory with name: no-name-5af7a029-e03e-4803-9223-6752a9bb5c3b
[I 2023-10-24 15:34:42,904] A new study created in memory with name: no-name-6100deca-4385-463d-b556-2490058f2321
[I 2023-10-24 15:34:42,905] A new study created in memory with name: no-name-c5620387-ec7d-4804-a252-fe546ed5d901
[I 2023-10-24 15:34:42,924] A new study created in memory with name: no-name-50924570-71ea-48bf-8aa3-ed72a4b27751
[I 2023-10-24 15:34:42,925] A new study created in memory with name: no-name-8579bd39-1d1a-4c55-81c2-a3378d177591
[I 2023-10-24 15:34:42,927] A new study created in memory with name: no-name-535ca2f4-d070-4053-94a2-48acb1e2c875
[I 2023-10-24 15:34:42,928] A new study created in memory with name: no-name-6e883a87-baae-4f73-8545-10e621755184
[I 2023-10-24 15:34:53,893] Trial 0 finished with value: inf and parameters: {'input_pair': 0, 'epsilon': 0.1, 'delta': 0.1, 'sensitivity': 1.0}. Best is trial 0 with value: inf.
[I 2023-10-24 15:34:53,959] Trial 0 finished with value: inf and parameters: {'input_pair': 0, 'epsilon': 0.1, 'delta': 0.1, 'sensitivity': 1.0}. Best is trial 0 with value: inf.
[I 2023-10-24 15:34:54,023] Trial 0 finished with value: inf and parameters: {'input_pair': 0, 'epsilon': 0.1, 'delta': 0.1, 'sensitivity': 1.0}. Best is trial 0 with value: inf.
[I 2023-10-24 15:34:54,040] Trial 0 finished with value: inf and parameters: {'input_pair': 0, 'epsilon': 0.1, 'delta': 0.1, 'sensitivity': 1.0}. Best is trial 0 with value: inf.
[I 2023-10-24 15:34:54,068] Trial 0 finished with value: inf and parameters: {'input_pair': 0, 'epsilon': 0.1, 'delta': 0.1, 'sensitivity': 1.0}. Best is trial 0 with value: inf.
[I 2023-10-24 15:34:54,226] Trial 0 finished with value: inf and parameters: {'input_pair': 0, 'epsilon': 0.1, 'delta': 0.1, 'sensitivity': 1.0}. Best is trial 0 with value: inf.
[I 2023-10-24 15:34:54,228] Trial 0 finished with value: inf and parameters: {'input_pair': 0, 'epsilon': 0.1, 'delta': 0.1, 'sensitivity': 1.0}. Best is trial 0 with value: inf.
[I 2023-10-24 15:34:54,275] Trial 0 finished with value: inf and parameters: {'input_pair': 0, 'epsilon': 0.1, 'delta': 0.1, 'sensitivity': 1.0}. Best is trial 0 with value: inf.
[I 2023-10-24 15:34:54,445] Trial 0 finished with value: inf and parameters: {'input_pair': 0, 'epsilon': 0.1, 'delta': 0.1, 'sensitivity': 1.0}. Best is trial 0 with value: inf.
[I 2023-10-24 15:34:54,689] Trial 0 finished with value: inf and parameters: {'input_pair': 0, 'epsilon': 0.1, 'delta': 0.1, 'sensitivity': 1.0}. Best is trial 0 with value: inf.
[I 2023-10-24 15:34:55,280] A new study created in memory with name: no-name-338955b5-d659-443a-9972-f236a9d799e1
[I 2023-10-24 15:34:55,428] A new study created in memory with name: no-name-2d50cb6a-4127-4e8c-8351-80f1d5cc590f
[I 2023-10-24 15:34:55,520] A new study created in memory with name: no-name-3df079d0-6751-491f-81fa-92fee7d73572
[I 2023-10-24 15:34:55,523] A new study created in memory with name: no-name-e1b6968d-ebf8-4165-8d81-9dd72406e1b6
[I 2023-10-24 15:34:55,625] A new study created in memory with name: no-name-ccf3cd3f-3f6e-4692-b137-b26545348451
[I 2023-10-24 15:34:55,692] A new study created in memory with name: no-name-0051ab58-f0c5-444c-9d19-a9814caf0b60
[I 2023-10-24 15:34:55,772] A new study created in memory with name: no-name-37cf9739-ffae-4abd-a3bc-77d74bccbe3a
[I 2023-10-24 15:34:55,734] A new study created in memory with name: no-name-ed283781-6952-4613-a383-fb56e4c04dcd
[I 2023-10-24 15:34:55,912] A new study created in memory with name: no-name-6f8f0656-0336-4d14-80f3-ee5e80e2e027
[I 2023-10-24 15:34:56,153] A new study created in memory with name: no-name-e6107257-713c-46bb-9ee6-43c0e8012332
[I 2023-10-24 15:34:56,194] Trial 0 finished with value: inf and parameters: {'input_pair': 0, 'epsilon': 0.1, 'delta': 0.1, 'sensitivity': 1.0}. Best is trial 0 with value: inf.
[I 2023-10-24 15:34:56,390] Trial 0 finished with value: inf and parameters: {'input_pair': 0, 'epsilon': 0.1, 'delta': 0.1, 'sensitivity': 1.0}. Best is trial 0 with value: inf.
[I 2023-10-24 15:34:56,439] Trial 0 finished with value: inf and parameters: {'input_pair': 0, 'epsilon': 0.1, 'delta': 0.1, 'sensitivity': 1.0}. Best is trial 0 with value: inf.
[I 2023-10-24 15:34:56,444] Trial 0 finished with value: inf and parameters: {'input_pair': 0, 'epsilon': 0.1, 'delta': 0.1, 'sensitivity': 1.0}. Best is trial 0 with value: inf.
[I 2023-10-24 15:34:56,460] Trial 0 finished with value: inf and parameters: {'input_pair': 0, 'epsilon': 0.1, 'delta': 0.1, 'sensitivity': 1.0}. Best is trial 0 with value: inf.
[I 2023-10-24 15:34:56,583] Trial 0 finished with value: inf and parameters: {'input_pair': 0, 'epsilon': 0.1, 'delta': 0.1, 'sensitivity': 1.0}. Best is trial 0 with value: inf.
[I 2023-10-24 15:34:56,585] Trial 0 finished with value: inf and parameters: {'input_pair': 0, 'epsilon': 0.1, 'delta': 0.1, 'sensitivity': 1.0}. Best is trial 0 with value: inf.
[I 2023-10-24 15:34:56,623] Trial 0 finished with value: inf and parameters: {'input_pair': 0, 'epsilon': 0.1, 'delta': 0.1, 'sensitivity': 1.0}. Best is trial 0 with value: inf.
[I 2023-10-24 15:34:56,707] Trial 0 finished with value: inf and parameters: {'input_pair': 0, 'epsilon': 0.1, 'delta': 0.1, 'sensitivity': 1.0}. Best is trial 0 with value: inf.
[I 2023-10-24 15:34:56,766] A new study created in memory with name: no-name-2ad5b39b-d485-4032-bc05-1a7dbef2eb17
[I 2023-10-24 15:34:56,977] A new study created in memory with name: no-name-04fd8a79-9f32-4eb3-807f-c5a46b446787
[I 2023-10-24 15:34:57,000] A new study created in memory with name: no-name-1879c7b9-df58-4893-adab-9e35ae483837
[I 2023-10-24 15:34:57,004] Trial 0 finished with value: inf and parameters: {'input_pair': 0, 'epsilon': 0.1, 'delta': 0.1, 'sensitivity': 1.0}. Best is trial 0 with value: inf.
[I 2023-10-24 15:34:57,072] A new study created in memory with name: no-name-388c9424-ebac-4c0c-a4eb-8c412b796fbf
[I 2023-10-24 15:34:57,019] A new study created in memory with name: no-name-41b24bf0-913c-4b42-aa38-e6cd3979253d
[I 2023-10-24 15:34:57,157] A new study created in memory with name: no-name-d6c0a629-6fa0-4ede-81a2-cbe402ffde75
[I 2023-10-24 15:34:57,260] A new study created in memory with name: no-name-19543ec7-9986-449b-9b67-6a9c3a1e70bb
[I 2023-10-24 15:34:57,276] A new study created in memory with name: no-name-75d0fff2-bc55-4a49-8f79-810f4f9d6681
[I 2023-10-24 15:34:57,428] A new study created in memory with name: no-name-9fb059e7-79f0-4241-a541-dbc429bac85c
[I 2023-10-24 15:34:57,707] A new study created in memory with name: no-name-6fad02ae-ec09-4e6f-966a-3ab2f24649ea
[I 2023-10-24 15:35:15,014] Trial 0 finished with value: inf and parameters: {'input_pair': 0, 'epsilon': 0.1, 'delta': 0.1, 'sensitivity': 1.0}. Best is trial 0 with value: inf.
[I 2023-10-24 15:35:15,049] Trial 0 finished with value: inf and parameters: {'input_pair': 0, 'epsilon': 0.1, 'delta': 0.1, 'sensitivity': 1.0}. Best is trial 0 with value: inf.
[I 2023-10-24 15:35:15,123] Trial 0 finished with value: inf and parameters: {'input_pair': 0, 'epsilon': 0.1, 'delta': 0.1, 'sensitivity': 1.0}. Best is trial 0 with value: inf.
[I 2023-10-24 15:35:15,154] Trial 0 finished with value: inf and parameters: {'input_pair': 0, 'epsilon': 0.1, 'delta': 0.1, 'sensitivity': 1.0}. Best is trial 0 with value: inf.
[I 2023-10-24 15:35:15,159] Trial 0 finished with value: inf and parameters: {'input_pair': 0, 'epsilon': 0.1, 'delta': 0.1, 'sensitivity': 1.0}. Best is trial 0 with value: inf.
[I 2023-10-24 15:35:15,262] Trial 0 finished with value: inf and parameters: {'input_pair': 0, 'epsilon': 0.1, 'delta': 0.1, 'sensitivity': 1.0}. Best is trial 0 with value: inf.
[I 2023-10-24 15:35:15,247] Trial 0 finished with value: inf and parameters: {'input_pair': 0, 'epsilon': 0.1, 'delta': 0.1, 'sensitivity': 1.0}. Best is trial 0 with value: inf.
[I 2023-10-24 15:35:15,413] Trial 0 finished with value: inf and parameters: {'input_pair': 0, 'epsilon': 0.1, 'delta': 0.1, 'sensitivity': 1.0}. Best is trial 0 with value: inf.
[I 2023-10-24 15:35:15,628] Trial 0 finished with value: inf and parameters: {'input_pair': 0, 'epsilon': 0.1, 'delta': 0.1, 'sensitivity': 1.0}. Best is trial 0 with value: inf.
[I 2023-10-24 15:35:15,740] Trial 0 finished with value: inf and parameters: {'input_pair': 0, 'epsilon': 0.1, 'delta': 0.1, 'sensitivity': 1.0}. Best is trial 0 with value: inf.
```

## Running the Benchmarks for Different Mechanisms

To produce the results in the paper, run the following command:

```bash
python scripts/run_benchmarks.py --mechanism <m>
```

Here, ```<m>``` describes the mechanism to audit and can be one of: 
1. **laplace_{opendp,ibm,pydp}** - The Laplace mechanism implemented in the OpenDP, IBM(i.e. Diffprivlib), or PyDP libraries, respectively
2. **laplace_inversion** - Our own floating-point-unsound implementation of the Laplace mechanism 
3. **gauss_{opendp,pydp,opacus}** - The Gaussian mechanism implemented in the OpenDP, PyDP, or Opacus libraries, respectively
4. **gauss_{ibm,ibm_analytic,ibm_discrete}** - Different implementations of the Gaussian mechanism in the IBM(i.e. Diffprivlib) library. Respectively, the different options correspond to Float Gauss, Analytic Float Gauss, and Discrete Gauss from the paper.
5. **gauss_{ziggurat,polar,boxmuller}** - Our Ziggurat, Polar, and Box-Muller implementations of the Gaussian mechanism.
6. **aim_internal** - Auditing the intermediate results of the AIM synthetic data generation algorithm
7. **mst_internal** - Auditing the intermediate results of the MST synthetic data generation algorithm

**Note:** To produce the full results from the paper, you need to execute the command with **every** mechanism above. For convenience, we supply a Linux bash script that does that for you:
```bash
scripts/audit_all.sh
```

The above command will run the auditing procedure of mechanism \<m\> with the exact same parameters as was done in the paper.
To see the full list of parameters to our script, please run:

```bash
python scripts/run_benchmarks.py --help
```

This will also display all other possible command-line arguments. For instance, to produce results for the Gaussian Mechanism from Opacus for ϵ = 0.1 and δ = 0.1:

```bash
python scripts/run_benchmarks.py --mechanism gauss_opacus --epsilon 0.1 --delta 0.1
```

**Note:** This may take up to an hour to run on a laptop. Using the ``--n 100`` option runs Delta-Siege using only 100 samples (results are still sound, but it is unlikely that any violation is found for so few samples):

```bash
python scripts/run_benchmarks.py --mechanism gauss_opacus --epsilon 0.1 --delta 0.1 --n 100
```

After running the previous steps, the Latex tables presented in Appendix D in the paper (from which Tables 4 and 6 in the main paper were extracted) can be recreated by running the following command:

```bash
python scripts/analyse_benchmarks.py > tables.tex
```

**Note:** The algorithm, by default, caches the finished computations in subfolders of the ```experiments``` folder. If you want to rerun particular experiments from scratch after they have been already executed, delete the particular subfolders corresponding to the experiments you want to rerun.

**Note:** Printing the Latex code for the tables directly to the terminal might yield invalid linebreaks. Therefore, it is recommended to stream into a file.

**Note** Using the ```--plot``` argument when running scripts/run_benchmarks.py will store a plot of all results from the experiment in the corresponding folder.


## Recreating the Figures from the Paper

To produce Figure 4 from the paper please follow the steps in the notebook scripts/Figure4.ipynb.
It can be run using the following command and opening the provided link in the browser:

```bash
PYTHONPATH=$PYTHONPATH:$(pwd) python -m jupyter notebook --no-browser --port 8888
```

We provide a Linux bash script to help obtain the information contained in Figure 5 in our paper:

```bash
scripts/extract_fig_5.sh --out_dir experiments/fig5
```

To get the plot please run the following command:
```python
python scripts/analyse_evolve.py --out_dir experiments/fig5
```

The plot is saved as ```mst_internal_3.0_1e-06.png```.

## Folder Structure

The repository has the following structure:

```
- deltasiege
    - .gitignore
    - LICENSE.txt
    - README.md
    - data
    - deltasiege
        - attack
            - optimizers
            - witnesses
        - classifiers
        - dataset
        - logging
        - main
        - mechanism
        - utils
    - scripts
```

The various folders have the following content:

- data: Contains our datasets constructed for auditing DP synthetic data generation mechanisms (AIM and MST).
- deltasiege: Contains the source code for our auditing tool
    - attack: Contains the implementation of our attack
        - optimizers: Describes the implementation of $\mathcal{V}(\rho, \underline{p}, \overline{p})$, line 20 in Algorithm 2 in the paper
        - witnesses: Implements the violation search for different search methodologies
            - witness_optimization.py: The attack methodology used by Delta Siege (Algorithm 2 in the paper)
            - witness_*.py: Other possible attack methodologies used by prior tools such as DP-Sniper or DP-Opt adapted to $\epsilon$, $\delta$ DP. These generally perform worse
    - classifier: Describes the methods for learning $\tilde{p}$, line 3 in Algorithm 2 in the paper
    - dataset: Describes an abstract data handler used in the implementation
    - logging: Describes a logger for results and resource usage
    - main: Describes the complete framework of our tool
    - mechanism: Provides a general DP mechanism interface to be implemented by algorithms that want to be audited by our tool. The folder also contains the code of the DP mechanisms we audit in our paper, including the external mechanisms described above. 
    - utils: Describes helper functionalities used throughout the implementation and for post-processing the results
- scripts: Contains scripts used to produce the results and visualizations presented in the paper
    - analyse_benchmarks.py: Script to postprocess the results from run_benchmarks.py and produce the Tables in Appendix D of the paper
    - Figure4.ipynb: Notebook used to produce Figure 4 in the paper. This notebook provides a thorough introduction to the code
    - minimal_example.ipynb: Notebook containing the minimal example for auditing your own algorithm, as described below
    - run_benchmarks.py: Script that executes the audit of different DP mechanisms and store the results

## Minimal Working Example for Auditing New DP Mechanisms

This provides a minimal working example to define and audit a custom DP mechanism. We use the Gaussian noise-adding mechanism as an example.
The entire code is available in scripts/minimal_example.ipynb. It can be run using the following command and opening the provided link in the browser:

```bash
PYTHONPATH=$PYTHONPATH:$(pwd) python -m jupyter notebook --no-browser --port 8888
```

The first step is to define the mechanism:

```python
import numpy as np
from deltasiege.mechanism import Mechanism

class GaussianMechanismExample(Mechanism):
    """
    An example (floating point insecure) implementation of the Gaussian mechanism
    as given in https://www.nowpublishers.com/article/Details/TCS-042
    """
    
    def _init_helper(self, epsilon: float, delta : float, sensitivity : float) -> None:
        """
        Initialization helper. Do not overwrite __init__
        """
        self.sensitivity = sensitivity
        self.std = self.guarantee_(epsilon, delta)
        super()._init_helper(epsilon, delta)
    
    def __call__(self, x : float, n: int) -> np.ndarray:
        """
        The specific mechanism - Gaussian noise is added to x
        n samples are drawn.
        """
        std = self.guarantee_(self.epsilon, self.delta)
        return x + np.random.normal(0, std, (n,))
    
    def constraint(self, epsilon : float, delta : float) -> bool:
        """
        Returns if epsilon and delta are valid DP parameters for the mechanism
        For this mechanism it must hold that 0 <= epsilon, delta <= 1
        """
        return 0 <= epsilon <= 1 and 0 <= delta <= 1
        
    def guarantee_(self, epsilon : float, delta : float) -> float:
        """
        A mapping of (epsilon, delta) to a parameter rho, which uniquely specifies the privacy level
        Is non-increasing in both epsilon and delta.
        For the classical Gaussian mechanism, one possible parameter is the standard deviation.
        """
        if epsilon <= 0 or delta <= 0:
            return float("inf")

        return np.sqrt(2 * np.log(1.25 / delta) * np.square(self.sensitivity / epsilon))
    
    def perturb_delta(self, new_delta : float) -> float:
        pass

    def perturb_epsilon(self, new_epsilon : float) -> float:
        pass
```

The required definitions are:
- ```_init_helper```: This is the initialization part of the mechanism. Do not override ```__init__```
- ```__call__```: This defines the mechanism itself. In this example, we would add Gaussian noise to ```x```, returning ```n``` samples.
- ```constraint```: This defines any constraints on $\epsilon$ and $\delta$. In this case, we have $\epsilon, \delta \in (0, 1)$
- ```guarantee_```: This defines the $\rho$ function from the paper. In this case, we have $\rho(\epsilon, \delta) = \sqrt{2 \log(1.25 / \delta) \Delta^2 / \epsilon^2}$

It is also highly recommended to define ```perturb_delta``` and ```perturb_epsilon```, but this is not strictly necessary. 
These two functions take a new $\delta$ or $\epsilon$ value and return another $\epsilon$ or $\delta$, respectively, such that the value of $\rho$ is unchanged.
This allows us to efficiently compute the pairs of $(\epsilon_1, \delta_1)$ in Figure 2 from the paper, making translation along the level curve of $\rho$ very efficient.
If these two functions are not explicitly defined, we employ binary search to find suitable values.

When the method is defined, one can easily audit the method by using the following example code:

```python
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from deltasiege.ddsampler import SimpleSampler
from deltasiege.classifiers import SklearnClassifier, BitPatternFeatureTransformer
from deltasiege.attack import WitnessOptimization, LinesearchEstimator

from deltasiege import DeltaSiege
from deltasiege.utils import Config
from deltasiege.logging import Logger

epsilon_low = 0.1
epsilon_high = 0.1
delta_low = 0.1
delta_high = 0.1
sensitivity_low = 1.0
sensitivity_high = 1.0

# Set up the sampler
search_space = SimpleSampler(
    GaussianMechanismExample,
    epsilon = (epsilon_low, epsilon_high),
    delta = (delta_low, delta_high),
    sensitivity = (sensitivity_low, sensitivity_high),
    input_pairs = [(0.0, 1.0)]
)

# Set up the learning method for learning \tilde{p}
classifier_factory = SklearnClassifier.get_factory(
    classifier=Pipeline([("bit_transformer", BitPatternFeatureTransformer()), ("log_reg", LogisticRegression())])
)

# Use a Delta Siege style witness
witnesses_factory = WitnessOptimization.get_factory(estimator=LinesearchEstimator())

# Set up and run the auditing
deltasiege = DeltaSiege(
    config = Config(), 
    search_space = search_space, 
    classifier_factories = classifier_factory,
    witnesses_factories=witnesses_factory,
    logger = Logger(Logger.State.SILENT),  # Surpresses all outputs. Can be VERBOSE, SILENT, DOWN. 
    base_folder = Path("path_to_storing_run"),
)

deltasiege_run = deltasiege.run("DeltaSiege")
```

We define the following variables:
- ```search_space```: This defines the space of (hyper-) parameters used by the mechanism, including the input pairs. We optimize this using ```optuna```
- ```classifier_factory```: This defines the learning algorithm for $\tilde{p}$. Any ```sklearn``` pipeline can be used
- ```witnesses_factory```: This defines the auditing methodology (the suggested factory is ```LinesearchEstimator``` corresponding to Algorithm 2 in our paper)
- ```deltasiege```: This is the main auditing tool

## Root Cause Analysis of DP Violations
In Section 6 of our paper, we describe how we use Delta-Seige to perform a root cause analysis of a floating point violation in the Laplace mechanism. To do so, we used a variation of the minimal example from the previous section:

```python
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from deltasiege.mechanism import LaplaceInversionMechanism
from deltasiege.ddsampler import SimpleSampler
from deltasiege.classifiers import SklearnClassifier, BitPatternFeatureTransformer

# Set up the sampler
search_space = SimpleSampler(
    LaplaceInversionMechanism,
    epsilon = (epsilon_low, epsilon_high),
    delta = (delta_low, delta_high),
    sensitivity = (sensitivity_low, sensitivity_high)
    input_pairs = [(0.0, 1.0)]
)

# Set up the learning method for learning \tilde{p}
classifier_factory = SklearnClassifier.get_factory(
    classifier=Pipeline([("bit_transformer", BitPatternFeatureTransformer()), ("log_reg", LogisticRegression())])
)

# Use a Delta Siege style witness
witnesses_factory = WitnessOptimization.get_factory(estimator=LinesearchEstimator())

# Set up and run the adutiting
deltasiege = DeltaSiege(
    config = Config(), 
    search_space = search_space, 
    classifier_factories = classifier_factory,
    witnesses_factories=witnesses_factory,
    logger = Logger(Logger.State.SILENT),  # Surpresses all outputs. Can be VERBOSE, SILENT, DOWN. 
    base_folder = Path("path_to_storing_run"),
)

deltasiege_run = deltasiege.run("DeltaSiege")

for experiment in deltasiege_run.experiments:
    for trial in experiment.trials:

        # Extract logistic regression
        trained_clf = trial.witnesses[trial.best_witness_idx].classifier.classifier
        log_reg = trained_clf.steps[1][1]

        # Get weights and intercept
        W, b = log_reg.coef_ , log_reg.intercept_
        
        # Analyze the weights ....witness
```
