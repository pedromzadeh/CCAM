from simulator.simulator import Simulator
import multiprocessing
import sys

# command line args
grid_id = int(sys.argv[1])
pol_type = sys.argv[2]

# define a simulator object
simulator = Simulator()
n_workers = 48

# run a total of 2*n_workers simulations
for batch_id in range(2):
    processes = [
        multiprocessing.Process(
            target=simulator.execute, args=[run_id, grid_id, pol_type]
        )
        for run_id in range(batch_id * n_workers, (batch_id + 1) * n_workers)
    ]

    # begin each process
    for p in processes:
        p.start()

    # wait to proceed until all finish
    for p in processes:
        p.join()
