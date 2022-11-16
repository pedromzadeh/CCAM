from simulator.simulator import Simulator
import multiprocessing
import sys

# command line args
grid_id = int(sys.argv[1])
pol_type = sys.argv[2]

# define a simulator object
simulator = Simulator()
n_workers = 3

processes = [
    multiprocessing.Process(target=simulator.execute, args=[0, id + grid_id, pol_type])
    for id in range(n_workers)
]

# begin each process
for p in processes:
    p.start()

# wait to proceed until all finish
for p in processes:
    p.join()
