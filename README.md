# Description
This codebase generates all the simulations and data presented in our paper [1]. Briefly, two cells are modeled energetically within the phase-field framework and undergo head-on collisions. Our main goal is to study how the speed and contact angle of the cell affect its chances to remain persistent upon head-on collisions.

[1] Zadeh, Pedrom, and Brian A., Camley. "Picking winners in cell-cell collisions: wetting, speed, and contact".bioRxiv (2022).

# Getting started

## Installation guide
1. Clone the repo:
    ```bash
    git clone https://github.com/pedromzadeh/collider.git
    ```

2. Create a clean `conda` environment with the right dependencies:
    ```bash
    conda env create -f environment.yaml
    conda activate collider
    ```

3. Install packages at the root of repo:
    ```bash
    cd collider
    pip install -e .
    ```

## Configuration file
Each run requires a configuration file from which a cell is initialized. To generate the exact feature space used for simulations in [1], simply do 
```bash 
python config/build.py
``` 
If you prefer to define your own, then you *must* follow the tree structure below:

```
config/
|
└─── sva
|    └─── grid_id0
|      │   cell0.yaml
|      │   cell1.yaml|    
|       ...
│   
└─── ffcr
     └─── grid_id0
       │   cell0.yaml
       │   cell1.yaml
        ...
```
Note that only the two modalities of static velocity-aligning (sva) and front-front contact repolarization (ffcr) are implemented. Moreover, note that `cell0.yaml` builds the *left* cell while `cell1.yaml` builds the *right* one. Read the docstring of `cell.py` to learn about the hyperparameters present in these files.

## Running a simulation
As an end user, you really need to only interact with the `driver` directory, which houses code for running simulations, processing results, and making visualizations.

To simulate a single two-body collision, simply run 
```bash 
cd driver
python single_run.py arg1 arg2
``` 
   - `arg1`: `int`, defines the sub directory to use for config files, 
   - `arg2`: `str`, defines the cell polarity mechanism -- options are `sva` and `ffcr`.

For example, `python single_run.py 0 sva` will run a collision with cells initialized from the files at `config/sva/grid_id0/`.

The results are stored in `output/sva/grid_id0/run_0/results.csv`, and the following important values are recorded:

1. surface tension $\gamma$, 
2. adhesion to the substrate $A$, 
3. strength of protrusion $\beta$,
4. center-of-mass speed $v_{\rm CM}$,
5. contact angle $\theta$, and
6. the binary representation of whether cells trained to the left (1) or to the right (0).

The last two observables are time series collected only after the cell has equilibrated and before it has collided -- this constitutes the pre-collision history.

## Processing collision outcomes
At the end of the day, we want relative center-of-mass speeds and contact angles averaged over the pre-collision times, defined as $\delta v=v_R - v_L$ and $\delta \theta = \theta_R - \theta_L$, respectively. `driver/process_data.py` does this for us and stores the processed results in `processed/*.csv`. In particular, the columns present are:

1. surface tension $\gamma$, 
2. adhesion to the substrate $A$, 
3. strength of protrusion $\beta$,
4. $\delta v$ in $\mu \rm m/ min$,
5. $\delta \theta$ in degrees, and
6. $P_{\rm win}$, the binary representation of whether cells trained to the left (1) or to the right (0).

The processed data is attached to this repository.

## Plots
Check out `driver/plots.ipynb` for a detailed look into how each plot presented in [1] was made. Feel free to check out `analysis.py` to see our data analysis schema in detail.  