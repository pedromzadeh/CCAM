import sys
import yaml
import os

pol_type = sys.argv[1]

SIZE = 5

# beta, gamma, A
default_cell_tup = (500, 0.9, 0.48)
default_cell_tuples = list((default_cell_tup,) * SIZE)

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{pol_type}")
centers = [12.5, 25]

for id in range(SIZE):
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
        R_eq=3.5,
        R_init=3.5,
        eta=0.5,
        N_wetting=500,
        g=2,
        alpha=0.5,
        D=0.01,
        J=3,
        lam=0.8,
        polarity_mode=str(pol_type).upper(),
    )

    with open(os.path.join(path, f"cell{0}.yaml"), "w") as yfile:
        yaml.dump(config, yfile)
