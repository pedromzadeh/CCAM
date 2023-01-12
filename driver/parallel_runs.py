from simulator.simulator import Simulator
import multiprocessing
import sys

# command line args
grid_id = int(sys.argv[1])
pol_type = "IM"

# define a simulator object
simulator = Simulator()
n_workers = 1

# run a total of n_workers simulations
for batch_id in range(1):
    processes = [
        multiprocessing.Process(
            target=simulator.execute,
            args=[run_id, grid_id, pol_type, n_workers * grid_id + run_id],
        )
        for run_id in range(batch_id * n_workers, (batch_id + 1) * n_workers)
    ]

    # begin each process
    for p in processes:
        p.start()

    # wait to proceed until all finish
    for p in processes:
        p.join()
