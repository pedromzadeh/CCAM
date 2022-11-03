import numpy as np


def PRW(cell):
    """
    Models cell polarity as a persistent random walk.

    Parameters
    ----------
    cell : Cell object
        The cell whose polarity we want to update.

    Returns
    -------
    float
        The change in cell polarity if modeled as PRW.
    """
    D = cell.D
    dt = cell.simbox.dt

    return np.sqrt(D * dt) * np.random.randn()


def dynamic_velocity_aligning(cell):
    """
    Computes the change in polarity angle at the next time step according to
    the velocity-aligning model, where the timescale of alignment scales by
    the speed of the cell, i.e. tau ~ 1/Jv.

    Parameters
    ----------
    cell : Cell object
        The cell whose polarity we want to update.

    Returns
    -------
    float
    """
    theta = cell.theta
    v = cell.v_cm
    p = [np.cos(theta), np.sin(theta)]
    D = cell.D
    J = cell.J
    dt = cell.simbox.dt

    v_norm = np.sqrt(v[0] ** 2 + v[1] ** 2) + 10e-6
    vxp_z = v[0] * p[1] - v[1] * p[0]
    gamma = np.arcsin(vxp_z / v_norm)
    tau = 1 / (J * v_norm)

    return -(1 / tau) * gamma * dt + np.sqrt(D * dt) * np.random.randn()


def static_velocity_aligning(cell, tau=3.0):
    """
    Computes the change in polarity angle at the next time step according to
    the velocity-aligning model, where the timescale of alignment is constant.

    Parameters
    ----------
    cell : Cell object
        The cell whose polarity we want to update.

    tau : float
        The timescale for velocity alignment, by default 3.

    Returns
    -------
    float
    """
    theta = cell.theta
    v = cell.v_cm
    p = [np.cos(theta), np.sin(theta)]
    D = cell.D
    dt = cell.simbox.dt

    v_norm = np.sqrt(v[0] ** 2 + v[1] ** 2) + 10e-6
    vxp_z = v[0] * p[1] - v[1] * p[0]
    gamma = np.arcsin(vxp_z / v_norm)

    return -(1 / tau) * gamma * dt + np.sqrt(D * dt) * np.random.randn()


def FFCR(cells, i, t, n_collision, tau_CR=1.5, tau_CG=3.0):
    """
    Computes the change in polarity of the cell for the next time step as
    governed by the front-front contact repolarization mechanism.

    Parameters
    ----------
    cells : list of Cell objects
        All cells in the system.

    i : int
        Specifies, by index, the cell whose polarity is getting updated.

    t : list of floats
        Vector parallel to the substrate.

    n_collision : int
        Specifies whether collision has happened.

    tau_CR : float
        Specifies the timescale for contact repolarization, by default 1.5.
        Note this should be 0.5 * tau_CG to enable repolarization.

    tau_CG : float
        Specifies the timescale for contact guidance, by default 3.

    Returns
    -------
    float
        Change incurred in polarity during one time step.
    """
    # unpack values
    j = 1 if (i == 0) else 0
    cm = cells[i].cm[1]
    D = cells[i].D
    dx = cells[i].simbox.dx
    dt = cells[i].simbox.dt
    pns = [[np.cos(cell.theta), np.sin(cell.theta)] for cell in cells]
    phis = [cell.phi for cell in cells]
    pn_i, pn_j = pns[i], pns[j]
    phi_i, phi_j = phis[i], phis[j]

    # contact guidance:
    #    --> align parallel to the substrate (i.e. direction of vector t)
    t_x_pi = t[0] * pn_i[1] - pn_i[0] * t[1]
    pi_dot_t = np.dot(t, pn_i)
    term_CG = -1 / tau_CG * np.arcsin(t_x_pi * _sgn(pi_dot_t))

    # front-front contact repolarization:
    #    --> by default, 0
    #    --> nonzero only when
    #            (1) cells have collided, and
    #            (2) cells are making head-head contact
    term_CR = 0
    cells[i].r_CR = [0, 0]

    # cells in contact if collision has happened and fields overlap
    if n_collision != -1 and np.max(phi_i * phi_j) > 0.1:

        # compute r_cm - r'
        dim = phi_i.shape[0]
        dr_x = np.zeros((dim, dim))
        dr_y = np.zeros((dim, dim))
        y, x = np.where(phi_i > 0.5)
        dr_x[y, x] = cm[0] - x * dx
        dr_y[y, x] = cm[1] - y * dx

        # threshold phi_j for leakage issues at high wetting
        phi_j = np.where(phi_j > 0.2, phi_j, 0)

        # compute unit direction desired by FFCR
        r_CR_x = np.sum(dr_x * phi_i * (phi_i * phi_j)) * dx**2
        r_CR_y = np.sum(dr_y * phi_i * (phi_i * phi_j)) * dx**2
        r_CR = np.array([r_CR_x, r_CR_y])
        norm = np.linalg.norm(r_CR)
        r_CR = r_CR / norm

        # ensure contact is actually head-head; if so, proceed to activate FFCR
        if _Heaviside(-pn_i[0] * pn_j[0]) * _Heaviside(r_CR[0] * pn_j[0]):
            cells[i].r_CR = r_CR
            rCR_x_pi = r_CR[0] * pn_i[1] - pn_i[0] * r_CR[1]
            term_CR = -1 / tau_CR * np.arcsin(rCR_x_pi)

    # update cell polarity by this amount
    noise = np.sqrt(D * dt) * np.random.randn()

    return term_CG * dt + term_CR * dt + noise


def polarity_potential(cell, mp, dx):
    """
    Computes the potential energy of the cell associated with its polarity.
    There is a cost if the polarity points toward where there is
    no substrate. This lets us bring the cell back onto the substrate.

    Parameters
    ----------
    cell : Cell object
        The cell.

    mp : ndarray of shape (n_mesh, n_mesh)
        The micropattern's phase field.

    dx : float
        The lattice spacing of the simulation box.

    Returns
    -------
    tuple
        angles : ndarray of shape (n_contours, )
        pol_pot : ndarray of shape (n_contours, )
    """
    # get cell's contour points
    cntr = cell.contour[0]
    x, y = cntr[:-1, 1], cntr[:-1, 0]

    # if point outside mp, costly; else not costly
    pol_pot_cntr = [mp[int(y), int(x)] - 1 for x, y in zip(x, y)]

    # evaluate angles associated with each contour point, then sort
    cm = cell.cm[0] / dx
    angles = [np.arctan2(y - cm[1], x - cm[0]) * 180 / np.pi for x, y in zip(x, y)]
    tups = zip(angles, pol_pot_cntr)
    tups = np.array(sorted(tups, key=lambda x: x[0]))
    angles, pol_pot = tups[:, 0], tups[:, 1]

    return angles, pol_pot


def _Heaviside(x):
    if x >= 0:
        return 1
    else:
        return 0


def _sgn(x):
    if x == 0:
        return 0
    elif x > 0:
        return 1
    else:
        return -1
