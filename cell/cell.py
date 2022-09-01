from src import utils
import numpy as np
import yaml

class Cell():
    """
    The Cell class defines a phase-field cell and implements methods to update the cell's field from one timestep to the next.

    Attributes
    ----------
    self.id : int
        Specifies the ID (by index) of the cell.

    self.R_eq : float
        Specifies the target radius, thus target area of the cell.

    self.R_init : float
        Specifies the initial radius for building the cell.

    self.center : list of floats
        Specifies the centroid of the cell when building it.

    self.gamma : float
        Specifies cell surface tension.

    self.A : float
        Specifies cell-substrate strength of adhesion.

    self.g : float
        Specifies cell-substrate strength of repulsion.

    self.beta : float
        Specifies strength with which cell generates protusions on substrate.

    self.D : float
        Specifies cell diffusion coefficient.

    self.J : float
        Specifies strength of polarity alignment to cell velocity.

    self.lam : float
        Specifies the phase field interfacial thickness.

    self.polarity_mode : str
        Specifies the modality used to update the cell's polarity.

    self.phi : ndarray of shape (N_mesh, N_mesh)
        The phase-field of the cell.

    self.W : ndarray of shape (N_mesh, N_mesh)
        Cumulative substrate the cell sees and interacts with.

    self.contour : list of ndarray, of shape (m, 2)
        The x, y points defining the half-contour of the field, where m is the number of such points.
    
    self.cm : ndarray of shape (2, 2)
        Rows are previous and current CM, respectively, with ordering (x, y).
    
    self.theta : float
        Defines cell polarity direction.

    self.v_cm : ndarray of shape (2,)
        Cell center of mass velocity (v_cm_x, v_cm_y).

    self.vx, self.vy : ndarray of shape (N_mesh, N_mesh)
        Specify the velocity fields.

    self.r_CR : ndarray of shape (2,)
        Defines the direction due to contact inhibition of locomotion (x, y).

    self.simbox : SimulationBox object
        Directly gives each cell access to simulation box parameters.
        
    Methods
    -------
    __init__(self, config_file)
        Initializes the cell.

    create(self, R, center)
        Builds the phase field.

    _tahn(self, r, R, epsilon)
        Returns the 2D hyperbolic tangent.

    _load_parameters(self, path)
        Loads cell's hyperparameters from file.
    """

    def __init__(self, config_file, _sim_box_obj):
        """
        Initializes the cell object with some hypterparameters and physical and spatial features.
        
        Parameters
        ----------
        config_file : str
            Path to where hyperparameters of the cell are stored.

        _sim_box_obj : SimulationBox object
            Gives access to simulation box parameters directly to each cell.
        """
        # read cell hyperparameters from file
        self._load_parameters(config_file)
        self.simbox = _sim_box_obj

        # spatial features of the cell
        self.phi = self._create()
        self.W = None
        self.contour = utils.find_contour(self.phi)
        self.cm = np.array([self.center, self.center])

        # physical features of the cell
        self.vx = np.zeros((_sim_box_obj.N_mesh, _sim_box_obj.N_mesh))
        self.vy = np.zeros((_sim_box_obj.N_mesh, _sim_box_obj.N_mesh))
        self.theta = np.random.rand() * np.pi
        self.v_cm = np.array([0, 0])
        self.r_CR = np.array([0, 0])

    def _create(self):
        """
        Computes the cell's phase-field from intial values and sets self.phi.
        """
        N_mesh, dx = self.simbox.N_mesh, self.simbox.dx
        phi = np.zeros((N_mesh, N_mesh))
        center, R = self.center, self.R_init
        epsilon = 1
        one_dim = np.arange(0, N_mesh, 1)
        x, y = np.meshgrid(one_dim, one_dim)
        r = np.sqrt((center[1] - y*dx)**2 + (center[0] - x*dx)**2)
        phi[y,x] = self._tanh(r, R, epsilon)
        return phi

    def _tanh(self, r, R, epsilon):
        return 1/2 + 1/2 * np.tanh(-(r-R)/epsilon)

    def _load_parameters(self, path):
        with open(path, 'r') as file:
            cell_params = yaml.safe_load(file)

        self.id = cell_params['id']
        self.R_eq = cell_params['R_eq']
        self.R_init = cell_params['R_init']
        self.center = cell_params['center']
        self.gamma = cell_params['gamma']
        self.A = cell_params['A']
        self.g = cell_params['g']
        self.beta = cell_params['beta']
        self.D = cell_params['D']
        self.J = cell_params['J']
        self.lam = cell_params['lam']
        self.polarity_mode = cell_params['polarity_mode']