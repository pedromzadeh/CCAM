from box.sim_box import SimulationBox
from potential.force import Force
from potential.energy import Energy
from cell.cell import Cell
from src.substrates import Substrate
from src import utils
from visuals.figure import Figure

import yaml
import glob
import pandas as pd
import numpy as np
import time

# assumption: 
#   --> cell.id == 0 is the left cell
#   --> cell.id == 1 is the right cell

def pair_collision(r, save_path):

    # time based seeding so every function call gets a new generator
    np.random.seed(int(time.time())+r)

    # read global hyperparameters
    with open('../configs/global_parameters.yaml', 'r') as file:
        gp = yaml.safe_load(file)
    N = gp['N']                    
    dt = gp['dt']
    N_mesh = gp['N_mesh']
    L_box = gp['L_box']
    N_wetting = gp['N_wetting']  
    kappa = gp['kappa']          
    omega = gp['omega']          
    mu = gp['mu']                
    eta = gp['eta']              

    # initialize the simulation box
    simbox = SimulationBox(L_box, N_mesh, dt)
    cells, chi = _build_system(simbox)

    # initialize the force and energy calculators
    force_calculator = Force(mu=mu, omega=omega, kappa=kappa)
    energy_calculator = Energy(mu=mu, omega=omega, kappa=kappa)

    # collision status not yet set
    n_coll = -1
    coll_orientation = -1
    train_direction = -1
    stats_df = pd.DataFrame([])

    for n in range(N):

        if _inside_simbox(cells):

            # collect statistics
            if n % 100 == 0 or n == 0:
                stats_df = utils.collect_stats(cells, table=stats_df)

            # view the simulation box    
            if n % 500 == 0 and n != 0:
                Figure.view_simbox(cells, chi, f"../visuals/temp/img_{n}.png")

            # set polarity and active force modality based on time step n
            force_modality = _set_polarity_angle(cells, n, N_wetting)

            # check for collision while it hasn't happened
            if n_coll == -1:
                status, c0_xcm_at_coll = _collision_detected(cells) 
                n_coll = (n+1) if status else -1
            
            # set orientation at collision only
            if n_coll != -1 and coll_orientation == -1:
                coll_orientation = _collision_orientation(cells)

            # if a proper train if formed, assess direction, and then we're done
            if n_coll != -1 and _proper_train_formed(cells, c0_xcm_at_coll):
                train_direction = _train_direction(cells)
                break
            
            # update each cell to the next time step
            [utils.evolve_cell(
                cell, 
                cells, 
                force_calculator, 
                force_modality,
                chi,
                eta,
                n_coll) for cell in cells]

        else:
            break

    # simulation is done; store data
    stats_df['coll time'] = n_coll
    stats_df['coll orientation'] = train_direction
    stats_df = stats_df.sort_values(by='cell id').reset_index(drop=True)
    stats_df.to_csv(save_path)

def _build_system(simbox):
   """
   Builds the substrate and cell system.
   
   Parameters
   ----------
   simbox : SimulationBox object
       Defines the box.
   
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
   xi = 0.5   # interfacial thickness
   
   # define base substrate
   sub = Substrate(N_mesh, L_box, 'rectangular', xi)
   arr = np.linspace(0, L_box, N_mesh)
   x, y = np.meshgrid(arr, arr)
   chi = sub.get_sub(x, y)

   # read cell config files && ensure only 2 exist
   config_files = glob.glob("../configs/config*")
   if len(config_files) != 2:
      raise ValueError("Ensure ONLY 2 configuration files exist so that only 2 cells are defined. This codebase is only for a two-body system.")

   # initialize cells with R_init at center
   # set the cumulative substrate they will interact with
   cells = []
   for config in config_files:
      cell = Cell(config, simbox)
      chi_yB = sub.get_sub(x, y+cell.lam)
      cell.W = -36 * cell.A/xi * chi**2*(1-chi)**2 + 0.5 * cell.g * chi_yB
      cells.append(cell)

   return cells, chi

def _set_polarity_angle(cells, n, N_wetting):
    """
    Manually sets the polarity angle of each cell based on the simulation time step. Stages are:
    1. push cells down to emulate gravity and have them wet on substrate,
    2. pull cells parallel to the substrate to generate lamellipodia,
    3. leave cells be so their polarity is now updated via system dynamics.
    
    Parameters
    ----------
    cells : list of Cell objects
        Cells in system.
    
    n : int
        Current simulation time.
    
    N_wetting : int
        Total time spent pushing cells down.
    
    Returns
    -------
    str
        The modality of the active force we want the cell to generate based on time in simulation.
    """

    # push down to wet
    if n < N_wetting:
        cells[0].theta = -np.pi/2
        cells[1].theta = -np.pi/2
        return 'constant'
        
    # pull parallel to substrate to form lamellipodia
    elif n < N_wetting + 1000:
        cells[0].theta = 0
        cells[1].theta = -np.pi
        return 'actin-poly'
        
    # let polarity evolve based on dynamics now
    else:
        return 'actin-poly'

def _inside_simbox(cells):
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
    x_l, x_r = [cell.contour[0][:,1] for cell in cells]
    x_lmin, x_rmax = np.min(x_l), np.max(x_r)
    N_mesh, L_box = cells[0].simbox.N_mesh, cells[0].simbox.L_box
    scale = L_box / N_mesh
    threshold = 1

    if (x_lmin * scale) < threshold or (x_rmax * scale) > (L_box - threshold):
        return False
    else:
        return True

def _collision_detected(cells):
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
        float or NoneType: x-component of cell 0 CM at time of collision, None if collision hasn't happened.
    """
    lam = cells[0].lam
    x_l, x_r = [cell.contour[0][:,1] for cell in cells]
    x_lmax, x_rmin = np.max(x_l), np.min(x_r)

    if np.fabs(x_lmax - x_rmin) < lam:
        return True, cells[0].cm[1][0]
    
    return False, None

def _collision_orientation(cells):
    """
    Assesses whether the two cells collide head-head or head-tail by looking at the x component of their polarity vectors.
    
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

def _proper_train_formed(cells, ref_pt):
    """
    Assesses whether a proper train is formed, i.e. a two-body system where cells are in contact && have migrated at least a cell radius away from collision site.
    
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

def _train_direction(cells):
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