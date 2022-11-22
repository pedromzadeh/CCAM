import sys
import yaml
import os
import numpy as np

pol_type = sys.argv[1]

subset = 30
SIZE = subset * 3

# beta, gamma, A
default_cell_tup = (500, 0.9, 0.48)
default_cell_tuples = list((default_cell_tup,) * SIZE)

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{pol_type}")
centers = [12.5, 25]

# grid search over cell size and angular noise
Ds = [0.05, 0.1, 0.5]
Rs = [3.5]

Ds = [[e] * subset for e in Ds]
Rs = [[e] * SIZE for e in Rs]

Ds = np.array(Ds).reshape(-1)
Rs = np.array(Rs).reshape(-1)

for id, D, R in list(zip(range(SIZE), Ds, Rs)):
    path = os.path.join(root, f"grid_id{id}")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    beta, gamma, A = default_cell_tup
    config = dict(
        id=0,
        center=centers,
        gamma=gamma,
        A=A,
        beta=beta,
        R_eq=float(R),
        R_init=3.5,
        eta=0.5,
        N_wetting=500,
        g=2,
        alpha=0.5,
        D=float(D),
        J=3,
        lam=0.8,
        polarity_mode=str(pol_type).upper(),
    )

    with open(os.path.join(path, f"cell{0}.yaml"), "w") as yfile:
        yaml.dump(config, yfile)
