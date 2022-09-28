import sys
import numpy as np
import yaml
import os

pol_type = sys.argv[1]

betas = np.linspace(4, 10, 6)
gammas = np.linspace(0.9, 1.8, 7)
As = np.linspace(0.32, 0.64, 11)
SIZE = len(betas) * len(gammas) * len(As)
grid = np.meshgrid(betas, gammas, As)
varied_cell_tuples = list(zip(grid[0].flatten(), grid[1].flatten(), grid[2].flatten()))

default_cell_tup = (6, 1.26, 0.48)
default_cell_tuples = list((default_cell_tup,) * SIZE)

print(f"speed: {betas} \ngamma: {gammas}\nsubA: {As}\n")
print(f"Total # of runs: {SIZE}")

tuples = [default_cell_tuples, varied_cell_tuples]
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{pol_type}")
centers = [[6, 9], [43, 9]]

for id in range(len(varied_cell_tuples)):
    path = os.path.join(root, f"grid_id{id}")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    for cell_id in range(2):
        beta, gamma, A = list(map(float, tuples[cell_id][id]))
        config = dict(
            id=cell_id,
            center=centers[cell_id],
            gamma=gamma,
            A=A,
            beta=beta,
            R_eq=3.5,
            R_init=3.5,
            eta=0.5,
            N_wetting=500,
            g=50,
            D=0.01,
            J=3,
            lam=0.8,
            polarity_mode=str(pol_type).upper(),
        )

        with open(os.path.join(path, f"cell{cell_id}.yaml"), "w") as yfile:
            yaml.dump(config, yfile)
