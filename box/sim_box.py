import yaml


class SimulationBox:
    def __init__(self, config_file):
        """
        Initialize the simulation box.

        Parameters
        ----------
        config_file : str
            Path to configuration file.
            Expected to include:
                - N : total simulation time
                - dt : timestep
                - N_mesh : number of lattice sites
                - L_box : physical size of the box.
            The resolution of the simulation is dx = L_box / (N_mesh - 1).
        """
        # read simulation box parameters
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
        N = config["N"]
        dt = config["dt"]
        N_mesh = config["N_mesh"]
        L_box = config["L_box"]

        self.sim_time = N
        self.dt = dt
        self.N_mesh = N_mesh
        self.L_box = L_box
        self.dx = L_box / (N_mesh - 1)
