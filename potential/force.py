import numpy as np
import yaml
from helper_functions import helper_functions as hf


class Force:
    def __init__(self, config_file):
        """
        Initialize the force object.

        Parameters
        ----------
        config_file : str
            Path to energy parameters config file.

        """
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
        self.kappa = config["kappa"]
        self.omega = config["omega"]
        self.mu = config["mu"]

    def cahn_hilliard_func_deriv(self, cell):
        """
        Computes the functional derivative of Cahn-Hilliard energy w.r.t field phi.

        Parameters
        ----------
        cell : Cell object
            Cell of interest.

        Returns
        -------
        ndarray of shape (N_mesh, N_mesh)
            The force per length due to Cahn-Hilliard energy.
        """
        _, _, lap = hf.compute_gradients(cell.phi, cell.simbox.dx)
        gamma, lam = cell.gamma, cell.lam
        term1 = 8 * (gamma / lam) * cell.phi * (cell.phi - 1) * (2 * cell.phi - 1)
        term2 = -2 * gamma * lam * lap
        return term1 + term2

    def area_func_deriv(self, cell):
        """
        Computes the functional derivative of the area conservation term w.r.t
        field phi.

        Parameters
        ----------
        cell : Cell object
            Cell of interest.

        Returns
        -------
        ndarray of shape (N_mesh, N_mesh)
            The force per length due to area conservation energy.
        """
        A = np.sum(cell.phi**2) * cell.simbox.dx**2
        A_target = np.pi * cell.R_eq**2
        return 2 * self.mu * (1 - A / A_target) * (-2 / A_target) * cell.phi

    def substrate_int_func_deriv(self, cell):
        """
        Computes the functional derivative of the cell-substrate interaction
        energy w.r.t field phi.

        Parameters
        ----------
        cell : Cell object
            The cell in question.

        Returns
        -------
        ndarray of shape (N_mesh, N_mesh)
            The force per length due to substrate interaction energy.
        """
        return 4 * cell.phi * (cell.phi - 2) * (cell.phi - 1) * cell.W

    def total_func_deriv(self, cell):
        """
        Computes the total functional derivative of the cell w.r.t field phi.

        Parameters
        ----------
        cell : Cell object
            Cell in question.

        Returns
        -------
        ndarray of shape (phi.shape[0], phi.shape[0])
            dF/dphi.
        """
        dFch_dphi = self.cahn_hilliard_func_deriv(cell)
        dFarea_dphi = self.area_func_deriv(cell)
        # dFchi_dphi = self.substrate_int_func_deriv(cell)

        return dFch_dphi + dFarea_dphi

    def constant_motility_force(self, cell):
        """
        Computes a constant motility force as $f = alpha * phi * p$

        cell : Cell object
            Cell in question.

        Returns
        -------
        tuple
            fx_motil : ndarray of shape (phi.shape[0], phi.shape[0])
            fy_motil : ndarray of shape (phi.shape[0], phi.shape[0])
        """
        # obtain relevant variables at time n
        phi_i = cell.phi
        theta_i = cell.theta
        p_i = [np.cos(theta_i), np.sin(theta_i)]

        fx_motil = cell.alpha * phi_i * p_i[0]
        fy_motil = cell.alpha * phi_i * p_i[1]
        return fx_motil, fy_motil

    def integrin_motility_force(self, cell, grad_phi, mp):

        # field normal to the boundary
        norm_grad_phi = np.sqrt(np.sum(grad_phi * grad_phi, axis=0))
        n_field = -1 * grad_phi / (norm_grad_phi + 1e-10)

        # make a field out of cell polarity
        phi = cell.phi
        p = [np.cos(cell.theta), np.sin(cell.theta)]
        p_field = np.array([np.ones(phi.shape) * p[0], np.ones(phi.shape) * p[1]])

        # denote front and rear of cell
        p_dot_n = np.sum(p_field * n_field, axis=0)
        cell_front = np.where(p_dot_n > 0, p_dot_n, 0)
        cell_rear = np.where(p_dot_n < 0, p_dot_n, 0)

        alpha_front = cell.alpha
        alpha_rear = alpha_front * 0.5

        magnitude = phi * (1 - mp) * (alpha_front * cell_front - alpha_rear * cell_rear)
        fx_motil = magnitude * p[0]
        fy_motil = magnitude * p[1]

        return fx_motil, fy_motil
