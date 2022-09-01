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

    def cell_adhesion_func_deriv(self, cell, cells):
        """
        Computes the functional derivative of the cell-cell adhesion energy
        w.r.t field phi.

        Parameters
        ----------
        cell : Cell object
            The cell of interest.

        cells : list of Cells
            All cells in the system.

        Returns
        -------
        ndarray of shape (N_mesh, N_mesh)
            The force per length due to cell-cell adhesion.
        """
        j = 0 if cell.id == 1 else 1
        nc = cells[j]
        neighbors_present = nc.phi**2 * (1 - nc.phi) ** 2

        return (
            -4
            * self.omega
            / cell.lam
            * cell.phi
            * (2 * cell.phi - 1)
            * (cell.phi - 1)
            * neighbors_present
        )

    def cell_repulsion_func_deriv(self, cell, cells):
        """
        Computes the functional derivative of cell-cell overlap w.r.t field phi.

        Parameters
        ----------
        cell : Cell object
            The cell of interest.

        cells : list of Cells
            All cells in the system.

        Returns
        -------
        ndarray of shape (N_mesh, N_mesh)
            The force per length due to cell-cell repulsion.
        """
        j = 0 if cell.id == 1 else 1
        nc = cells[j]
        neighbors_present = nc.phi**2

        return 4 * self.kappa / cell.lam * cell.phi * neighbors_present

    def total_func_deriv(self, cell, cells, n):
        """
        Computes the total functional derivative of the cell w.r.t field phi.

        Parameters
        ----------
        cell : Cell object
            Cell in question.

        cells : list of Cell objects
            All cells in system.

        Returns
        -------
        ndarray of shape (phi.shape[0], phi.shape[0])
            dF/dphi.
        """
        dFch_dphi = self.cahn_hilliard_func_deriv(cell)
        dFarea_dphi = self.area_func_deriv(cell)
        dFchi_dphi = self.substrate_int_func_deriv(cell)
        dFadh_dphi = self.cell_adhesion_func_deriv(cell, cells)
        dFrep_dphi = self.cell_repulsion_func_deriv(cell, cells)

        return dFch_dphi + dFarea_dphi + dFchi_dphi + dFadh_dphi + dFrep_dphi

    def constant_motility_force(self, cell, alpha):
        """
        Computes a constant motility force as $f = alpha * phi * p$

        cell : Cell object
            Cell in question.

        alpha : float, optional
            Specifies the strength of constant force.

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

        fx_motil = alpha * phi_i * p_i[0]
        fy_motil = alpha * phi_i * p_i[1]
        return fx_motil, fy_motil

    def actin_motility_force(self, cell, chi, grad_phi, x_hat):
        """
        Computes the motility force as $f = \beta * (p * x) * |grad phi|
        |grad chi| * H(p * n) * x$. This approximates forces generated at the
        cell front by actin polymerization, and it leads to lamellipodia.

        Parameters
        ----------
        cell : Cell object
            Cell in question.

        chi : ndarray of shape (phi.shape[0], phi.shape[0])
            Specifies the substrate via its field.

        grad_phi : ndarray of shape (2, phi.shape[0], phi.shape[0])
            Gradient of cell's field, phi.

        x_hat : list of floats, optional
            Specifies the direction parallel to the substrate.

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

        grad_chix, grad_chiy, _ = hf.compute_gradients(chi, cell.simbox.dx)
        grad_chi = np.array([grad_chix, grad_chiy])
        p_field = np.array(
            [np.ones(phi_i.shape) * p_i[0], np.ones(phi_i.shape) * p_i[1]]
        )
        norm_grad_phi = np.sqrt(np.sum(grad_phi * grad_phi, axis=0))
        norm_grad_chi = np.sqrt(np.sum(grad_chi * grad_chi, axis=0))
        n_field = -1 * grad_phi / (norm_grad_phi + 1e-10)
        p_dot_t = np.dot(p_i, x_hat)
        p_dot_n = np.sum(p_field * n_field, axis=0)
        H_p_dot_n = np.where(p_dot_n > 0, 1, 0)

        # smoothen norm_grad_phi to remove interface thickness artifacts
        eps = 0.25
        T = 0.1
        soft_T_norm_grad_phi = 0.5 * (np.tanh((norm_grad_phi - T) / eps) + 1)

        # force density
        magnitude = (
            cell.beta * soft_T_norm_grad_phi * norm_grad_chi * H_p_dot_n * p_dot_t
        )

        fx_motil = magnitude * x_hat[0]
        fy_motil = magnitude * x_hat[1]
        return fx_motil, fy_motil
