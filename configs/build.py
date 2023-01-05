import numpy as np
import yaml
import os

pol_type = "IM"
N_cells = 1

# parameters to sweep through
#   - beta  : strength with which pol field responds to shape changes
#   - tau   : timescale of pol field decay
#   - D     : magnitude of noise

betas = np.linspace(1, 1, 1)
taus = np.linspace(1, 1, 1)
Ds = np.linspace(0.1, 0.1, 1)
SIZE = len(betas) * len(taus) * len(Ds)
grid = np.meshgrid(betas, taus, Ds)
varied_cell_tuples = list(zip(grid[0].flatten(), grid[1].flatten(), grid[2].flatten()))

print(f"betas: {betas} \ntaus: {taus}\nDs: {Ds}\n")
print(f"Total # of runs: {SIZE}")

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{pol_type}")

for id in range(SIZE):
    path = os.path.join(root, f"grid_id{id}")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    for cell_id in range(N_cells):
        beta, tau, D = list(map(float, varied_cell_tuples[id]))
        config = dict(
            center=[12.5, 25],
            gamma=1.0,
            A=0,
            R_eq=4,
            R_init=4,
            eta=0.5,
            g=2,
            lam=0.8,
            N_wetting=500,
            alpha=1,
            tau_mp=1,
            id=cell_id,
            polarity_mode=str(pol_type).upper(),
            beta=beta,
            tau=tau,
            D=D,
        )

        with open(os.path.join(path, f"cell{cell_id}.yaml"), "w") as yfile:
            yaml.dump(config, yfile)
