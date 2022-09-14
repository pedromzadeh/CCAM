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

    def execute(self, run_id, grid_id):

        # define various paths
        paths = self._define_paths(run_id, grid_id)

        print(f"Run ID = {run_id}")

        # time based seeding so every function call gets a new generator
        np.random.seed(int(time.time()) + run_id)

        # initialize the simulation box
        simbox = SimulationBox(paths["simbox"])
        cells, chi = self._build_system(simbox, paths["cells"])

        # initialize the force calculator
        force_calculator = Force(paths["energy"])

        # collision status not yet set
        n_coll = -1
        coll_orientation = -1
        train_direction = -1
        stats_df = pd.DataFrame([])

        # carry out the simulation && collect statistics
        for n in range(simbox.sim_time):

            if self._inside_simbox(cells):

                # collect statistics
                if n % 500 == 0 and n > cells[0].N_wetting + 1000:
                    stats_df = hf.collect_stats(cells, table=stats_df)

                # view the simulation box
                if n % 100 == 0:  # and n > cells[0].N_wetting + 1000:
                    Figure.view_simbox(
                        cells,
                        chi,
                        os.path.join(paths["figures"], f"img_{n}.png"),
                    )

                # set polarity and active force modality based on time step n
                force_modality = self._set_polarity_angle(cells, n)

                # check for collision while it hasn't happened
                if n_coll == -1:
                    status, c0_xcm_at_coll = self._collision_detected(cells)
                    n_coll = (n + 1) if status else -1

                # set orientation at collision only
                if n_coll != -1 and coll_orientation == -1:
                    coll_orientation = self._collision_orientation(cells)

                # if a proper train if formed, assess direction, and then we're done
                if n_coll != -1 and self._proper_train_formed(cells, c0_xcm_at_coll):
                    train_direction = self._train_direction(cells)
                    break

                # update each cell to the next time step
                [
                    hf.evolve_cell(
                        cell, cells, force_calculator, force_modality, chi, n_coll, n
                    )
                    for cell in cells
                ]

            else:
                break

        # simulation is done; store data
        stats_df["coll time"] = n_coll
        stats_df["coll orientation"] = train_direction
        stats_df = stats_df.sort_values(by="cell id").reset_index(drop=True)
        stats_df.to_csv(paths["result"])

    def _build_system(self, simbox, cells_config):
        """
        Builds the substrate and cell system.

        Parameters
        ----------
        simbox : SimulationBox object
            Defines the box.

        cells_config : str
            Path to directory containing each cell's config.

        Returns
        -------
        tuple
            cells : list of Cell objects
            chi : Substrate object

        Raises
        ------
        ValueError
            Ensures only 2 cells are defined.
        """
        # unpack
        N_mesh, L_box = simbox.N_mesh, simbox.L_box

        # define base substrate
        sub = Substrate(N_mesh, L_box)
        arr = np.linspace(0, L_box, N_mesh)
        x, y = np.meshgrid(arr, arr)
        chi = sub.get_substrate(x, y, type="rectangular")

        # read cell config files && ensure only 2 exist
        # IMPORTANT -- glob returns arbitrary order, sort
        config_files = sorted(glob.glob(os.path.join(cells_config, "cell*")))
        if len(config_files) != 2:
            raise ValueError("Ensure there is exactly 2 configuration files.")

        # initialize cells with R_init at center
        # set the cumulative substrate they will interact with
        cells = []
        for config in config_files:
            cell = Cell(config, simbox)
            chi_yB = sub.get_substrate(x, y + cell.lam, type="rectangular")
            cell.W = (
                -36 * cell.A / sub.xi * chi**2 * (1 - chi) ** 2
                + 0.5 * cell.g * chi_yB
            )
            cells.append(cell)

        return cells, chi

    def _set_polarity_angle(self, cells, n):
        """
        Manually sets the polarity angle of each cell based on the simulation time step.
        Stages are:
        1. push cells down to emulate gravity and have them wet on substrate,
        2. pull cells parallel to the substrate to generate lamellipodia,
        3. leave cells be so their polarity is now updated via system dynamics.

        Parameters
        ----------
        cells : list of Cell objects
            Cells in system.

        n : int
            Current simulation time.

        Returns
        -------
        str
            The modality of the active force we want the cell to generate based
            on time in simulation.
        """
        N_wetting = cells[0].N_wetting
        # push down to wet
        if n < N_wetting:
            cells[0].theta = -np.pi / 2
            cells[1].theta = -np.pi / 2
            return "constant"

        # pull parallel to substrate to form lamellipodia
        elif n < N_wetting + 1000:
            cells[0].theta = 0
            cells[1].theta = -np.pi
            return "actin-poly"

        # let polarity evolve based on dynamics now
        else:
            return "actin-poly"

    def _inside_simbox(self, cells):
        """
        Assesses whether both cells are inside the simulation box.

        Parameters
        ----------
        cells : list of Cell objects
            Cells in system.

        Returns
        -------
        bool
            True if both cells are still within the simulation box, False otherwise.
        """
        x_l, x_r = [cell.contour[0][:, 1] for cell in cells]
        x_lmin, x_rmax = np.min(x_l), np.max(x_r)
        N_mesh, L_box = cells[0].simbox.N_mesh, cells[0].simbox.L_box
        scale = L_box / N_mesh
        threshold = 1

        if (x_lmin * scale) < threshold or (x_rmax * scale) > (L_box - threshold):
            return False
        else:
            return True

    def _collision_detected(self, cells):
        """
        Assesses whether two cells have collided.

        Parameters
        ----------
        cells : list of Cell objects
            Cells in the system.

        Returns
        -------
        tuple
            bool : True if cells have collided, False otherwise.
            float or NoneType: x-component of cell 0 CM at time of collision,
            None if collision hasn't happened.
        """
        lam = cells[0].lam
        x_l, x_r = [cell.contour[0][:, 1] for cell in cells]
        x_lmax, x_rmin = np.max(x_l), np.min(x_r)

        if np.fabs(x_lmax - x_rmin) < lam:
            return True, cells[0].cm[1][0]

        return False, None

    def _collision_orientation(self, cells):
        """
        Assesses whether the two cells collide head-head or head-tail by
        looking at the x component of their polarity vectors.

        Parameters
        ----------
        cells : list of Cell objectsq
            Cells in the system.

        Returns
        -------
        str
            "HH" if p_1.x > 0 and p_2.x < 0,
            "HT" otherwise.
        """
        thetas = [cell.theta for cell in cells]
        p_1, p_2 = list(map(lambda x: [np.cos(x), np.sin(x)], thetas))

        if p_1[0] > 0 and p_2[0] < 0:
            return "HH"
        else:
            return "HT"

    def _proper_train_formed(self, cells, ref_pt):
        """
        Assesses whether a proper train is formed, i.e. a two-body system where
        cells are in contact && have migrated at least a cell radius away from
        collision site.

        Parameters
        ----------
        cells : list of Cell objects
            Cells in the system.

        ref_pt : float
            x-coordinate of a reference point at which collision happened.

        Returns
        -------
        bool
            True if cells in contact and train equilibrated,
            False otherwise.
        """
        cells_in_contact = False
        train_equilibrated = False
        if np.max(cells[0].phi * cells[1].phi) > 0.15:
            cells_in_contact = True
        if np.fabs(cells[0].cm[1][0] - ref_pt) > cells[0].R_eq:
            train_equilibrated = True

        return cells_in_contact and train_equilibrated

    def _train_direction(self, cells):
        """
        Assesses the direction of the cohesive train.

        Parameters
        ----------
        cells : list of Cell objects
            Cells in the system.

        Returns
        -------
        str
            "right" if both polarities have + x components,
            "left" if both polarities have - x components,
            "unknown" otherwise.
        """
        thetas = [cell.theta for cell in cells]
        p_1, p_2 = list(map(lambda x: [np.cos(x), np.sin(x)], thetas))

        if p_1[0] > 0 and p_2[0] > 0:
            return "right"
        elif p_1[0] < 0 and p_2[0] < 0:
            return "left"
        else:
            return "unknown"

    def _define_paths(self, run_id, grid_id):
        SIMBOX_CONFIG = os.path.join(self.root_dir, "configs/simbox.yaml")
        ENERGY_CONFIG = os.path.join(self.root_dir, "configs/energy.yaml")

        CELLS_CONFIG = os.path.join(self.root_dir, f"configs/grid_id{grid_id}")
        assert os.path.exists(CELLS_CONFIG)
        assert os.path.isdir(CELLS_CONFIG)

        run_root = os.path.join(
            self.root_dir, "output", f"grid_id{grid_id}", f"run_{run_id}"
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
            cells=CELLS_CONFIG,
            result=RESULT_PATH,
            figures=FIGURES_PATH,
        )
