from simulator.simulator import Simulator
import sys

# grid id
grid_id = int(sys.argv[1])

# define a simulator object
simulator = Simulator()
simulator.execute(0, grid_id=grid_id)
