from box.sim_box import SimulationBox
from potential.force import Force
from cell.cell import Cell
from substrate.substrates import Substrate
from helper_functions import helper_functions as hf
from visuals.figure import Figure

import glob
import pandas as pd
import numpy as np
import time
import os

# assumption:
#   --> cell.id == 0 is the left cell
#   --> cell.id == 1 is the right cell


class Simulator:
    """
    Implements a simulator object, which runs two-body collisions and collects
    relevant statistics.

    Attributes
    ----------
    self.root_dir : str
        Path to project's root directory.

    Methods
    -------
    __init__(self, config_dir, results_dir, figures_dir)
        Initialize the simulator.

    execute(self, exec_id, save_results_at, save_figures_at)
        Run one complete simulation.

    _build_system(self, simbox)
        Builds the cell and substrate system.

    _set_polarity_angle(self, cells, n)
        Sets the polarity of the cells.

    _inside_simbox(self, cells)
        Determines if cells are inside the box.

    _collision_detected(self, cells)
        Determines whether collision is detected.

    _collision_orientation(self, cells)
        Determines the orientation of collision.

    _proper_train_formed(self, cells, ref_pt)
        Decides whether a proper train of cells is formed.

    _train_direction(self, cells)
        Determines the direction of the train.

    _define_paths(self)
        Setup various paths for the simulator to use.
    """

    def __init__(self):
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def execute(self, run_id, grid_id, polarity_type):
        """
        Executes one complete simulation run colliding two cells head-on.

        Parameters
        ----------
        run_id : int
            Defines the id of this particular run.

        grid_id : int
            Defines the id of the grid point in the 3D feature space.

        polarity_type : str
            Specifies the polarity type being used. Options are "SVA"
            for static velocity-alignment, and "FFCR" for front-front
            contact repolarization.
        """
        # define various paths
        paths = self._define_paths(run_id, grid_id, polarity_type)

        # time based seeding so every function call gets a new generator
        np.random.seed(int(time.time()) + run_id)

        # initialize the simulation box
        simbox = SimulationBox(paths["simbox"])
        cell, chi = self._build_system(simbox, paths["cell"])

        # initialize the force calculator
        force_calculator = Force(paths["energy"])

        cms = pd.DataFrame()

        # carry out the simulation
        for n in range(simbox.sim_time):

            # collect statistics
            if n % simbox.n_stats == 0:
                cms = pd.concat([cms, pd.DataFrame(cell.cm[1])])

            # view the simulation box
            if n % simbox.n_view == 0:
                Figure.view_simbox(
                    cell,
                    chi,
                    os.path.join(paths["figures"], f"img_{n}.png"),
                )

            # set polarity and active force modality based on time step n
            force_modality = "constant"

            # update each cell to the next time step
            hf.evolve_cell(cell, force_calculator, force_modality)

    def _build_system(self, simbox, cell_config):
        """
        Builds the substrate and cell system.

        Parameters
        ----------
        simbox : SimulationBox object
            Defines the box.

        cell_config : str
            Path to directory containing cell's config.

        Returns
        -------
        tuple
            cell : Cell object
            chi : Substrate object

        Raises
        ------
        ValueError
            Ensures only 1 cell is defined.
        """
        # unpack
        N_mesh, L_box = simbox.N_mesh, simbox.L_box

        # define base substrate
        sub = Substrate(N_mesh, L_box)
        chi = sub.two_state_sub()

        # read cell config files && ensure only 2 exist
        # IMPORTANT -- glob returns arbitrary order, sort
        config_file = sorted(glob.glob(os.path.join(cell_config, "cell*")))
        if len(config_file) != 1:
            raise ValueError("Ensure there is exactly 1 configuration file.")

        # initialize cells with R_init at center
        # set the cumulative substrate they will interact with
        cell = Cell(config_file[0], simbox)
        cell.W = 0.5 * cell.g * chi

        return cell, chi

    def _define_paths(self, run_id, grid_id, polarity_type):
        SIMBOX_CONFIG = os.path.join(self.root_dir, "configs/simbox.yaml")
        ENERGY_CONFIG = os.path.join(self.root_dir, "configs/energy.yaml")

        if polarity_type not in ["sva", "ffcr"]:
            raise KeyError(f"Polarity mode {polarity_type} is invalid.")

        CELL_CONFIG = os.path.join(
            self.root_dir, f"configs/{polarity_type}/grid_id{grid_id}"
        )
        assert os.path.exists(CELL_CONFIG)
        assert os.path.isdir(CELL_CONFIG)

        run_root = os.path.join(
            self.root_dir,
            "output",
            f"{polarity_type}",
            f"grid_id{grid_id}",
            f"run_{run_id}",
        )
        if not os.path.exists(run_root):
            os.makedirs(run_root)

        RESULT_PATH = os.path.join(run_root, "result.csv")

        FIGURES_PATH = os.path.join(run_root, "visuals")
        if not os.path.exists(FIGURES_PATH):
            os.makedirs(FIGURES_PATH)

        return dict(
            simbox=SIMBOX_CONFIG,
            energy=ENERGY_CONFIG,
            cell=CELL_CONFIG,
            result=RESULT_PATH,
            figures=FIGURES_PATH,
        )
