import warnings
from polarity import polarity
from skimage import measure
import numpy as np
import pandas as pd


def find_contour(field, level=0.5):
    """
    Computes the contour at the specified level.

    Parameters
    ----------
    field : ndarray of shape (n, n)
        Phase field.

    level : float, optional
        Level of the contour points, by default 0.5

    Returns
    -------
    list of ndarray of shape (m, 2)
        m is the number of points, and the ordering is (y, x). More than one
        element in the list suggests disjointed sets of contour points.
    """
    return measure.find_contours(field, level=level)


def compute_CM(cell):
    """
    Computes the center of mass of cell.

    Parameters
    ----------
    cell : Cell object
        Cell in question.

    Returns
    -------
    tuple of ndarray of shape (2,)
        current center-of-mass, new center-of-mass, respectively.
    """
    N_mesh, dx = cell.simbox.N_mesh, cell.simbox.dx
    x = np.arange(0, N_mesh, 1)
    xg, yg = np.meshgrid(x, x)
    cm_x, cm_y = np.sum(xg * cell.phi), np.sum(yg * cell.phi)
    norm = np.sum(cell.phi)

    return cell.cm[1], np.array([cm_x / norm * dx, cm_y / norm * dx])


def compute_v_CM(cell):
    """
    Computes the center of mass velocity.

    Parameters
    ----------
    cell : Cell object
        Cell in question.

    Returns
    -------
    float
    """
    delta_cm = np.diff(cell.cm, axis=0)[0]
    return delta_cm / cell.simbox.dt


def compute_area(cell, order=2):
    """
    Computes the approximate area of the cell as $int phi^(order) dr$.

    Parameters
    ----------
    cell : Cell object
        Cell in question.

    order : int, optional
        Specifies the power of phi. Larger values give smaller areas as field
        is tightened, by default 2

    Returns
    -------
    float
    """
    return np.sum(np.power(cell.phi, order) * cell.simbox.dx**2)


def compute_gradients(field, dx):
    """
    Computes grad(field) and laplacian(field).

    Parameters
    ----------
    field : ndarray of shape (n, n)
        Phase field.

    dx : float
        Resolution of the simulation box.

    Returns
    -------
    tuple of ndarray of shape (field.shape[0], field.shape[0])
            grad_x, grad_y, laplacian
    """
    # compute the gradients, each component is separate
    field_right = np.roll(field, 1, axis=1)
    field_left = np.roll(field, -1, axis=1)
    field_up = np.roll(field, -1, axis=0)
    field_down = np.roll(field, 1, axis=0)

    # compute the gradients, each component is separate
    grad_x = (field_left - field_right) / (2 * dx)
    grad_y = (field_up - field_down) / (2 * dx)

    # compute laplacian
    laplacian = (field_left + field_right + field_up + field_down - 4 * field) / dx**2

    return (grad_x, grad_y, laplacian)


def compute_contact_angle(cell):
    """
    Computes the approximate contact angle between the cell front and the
    substrate, in degrees.

    Parameters
    ----------
    cell : Cell object
        The cell in question.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        Ensures only 2 cells exist in the system.

    Notes
    -----
    There are two strict assumptions that have to be met:
    1. The entire cell is in one frame and contour set is not disjoint
    2. There are only 2 cells, IDed 0 and 1. Cell 0 travels from left to right,
    and cell 1 travels from right to left.
    """
    # checks the two assumptions
    assert cell.id < 2
    assert len(cell.contour) == 1

    # get the two endpoints of the cell that are in contact with the substrate
    #   --> assign a range of y values to be on the substrate
    #   --> get indices of all y within this range
    #   --> find the two extrema from the list of associated x values
    #   --> find the indices of these two x values
    #           - Contour points are given from top and go counter clockwise. If
    #             there are multiple y values with same xmin, len(ind_xmin) > 1.
    #             Due to CCW nature, the last index is associated with ymin.
    x, y = cell.contour[0][:, 1], cell.contour[0][:, 0]
    buffer = 1
    ymin = np.min(y)
    y_slice = np.where((y >= ymin) & (y <= ymin + buffer))[0]
    xmin, xmax = np.min(x[y_slice]), np.max(x[y_slice])
    ind_xmin, ind_xmax = np.where(x == xmin)[0], np.where(x == xmax)[0][0]
    ind_xmin = ind_xmin[len(ind_xmin) - 1]
    eps = 10e-8

    # if phi is at the border, return nan
    if xmin < 1 or xmax > cell.simbox.N_mesh - 1:
        return None

    # left -> right, contact angle is on the right side
    if cell.id == 0:
        x1, x2 = x[ind_xmax + 4], x[ind_xmax + 10]
        y1, y2 = y[ind_xmax + 4], y[ind_xmax + 10]

        m = (y2 - y1) / (x2 - x1 + eps)
        ca = np.arctan(-m) * 180 / np.pi

        return ca

    # right -> left, contact angle is on the left side
    elif cell.id == 1:
        x1, x2 = x[ind_xmin - 10], x[ind_xmin - 4]
        y1, y2 = y[ind_xmin - 10], y[ind_xmin - 4]

        m = (y2 - y1) / (x2 - x1 + eps)
        ca = np.arctan(m) * 180 / np.pi

        return ca

    else:
        raise ValueError(
            "This method only works for two-body cells. Check to make sure only \
                2 cells exist in the system. Also, read the assumptions carefully."
        )


def collect_stats(cells, table=None):
    """
    Measures various statistics about the cells and appends it as a new row to
    the ongoing table of measurements.

    Parameters
    ----------
    cells : list of Cell objects
        All cells in the system.

    table : pd.DataFrame, optional
        The running table holding data, by default None

    Returns
    -------
    pd.DataFrame
        Table of data with the new measurements added as rows. The columns are
        ['cell id',
        'lambda',
        'gamma',
        'A',
        'beta',
        'x_CM',
        'y_CM',
        'polarity angle',
        'CM speed',
        'contact angle',
        'x_rCR',
        'y_rCR',
        'N_mesh',
        'L_box',
        'dt']
    """
    for cell in cells:
        v = np.linalg.norm(cell.v_cm)
        ca = compute_contact_angle(cell)
        simbox = cell.simbox

        cell_df = pd.DataFrame(
            data=[
                [
                    cell.id,
                    cell.lam,
                    cell.gamma,
                    cell.A,
                    cell.beta,
                    cell.cm[1][0],
                    cell.cm[1][1],
                    cell.theta,
                    v,
                    ca,
                    cell.r_CR[0],
                    cell.r_CR[1],
                    simbox.N_mesh,
                    simbox.L_box,
                    simbox.dt,
                ]
            ],
            columns=[
                "cell id",
                "lambda",
                "gamma",
                "A",
                "beta",
                "x_CM",
                "y_CM",
                "polarity angle",
                "CM speed",
                "contact angle",
                "x_rCR",
                "y_rCR",
                "N_mesh",
                "L_box",
                "dt",
            ],
        )
        table = pd.concat([table, cell_df])

    return table


def evolve_cell(cell, force, force_modality):
    """
    Evolves the cell by updating its class variables from time t to time t_dt.
    Attributes updated are
    1. field
    2. polarity
    3. center of mass
    4. center of mass speed
    5. velocity fields, v_x and v_y
    6. contour of the field.

    Parameters
    ----------
    cell : Cell object
        Cell of interest to update.

    force : Force object
        Gives access to computing forces.

    force_modality : str
        Specifies what kind of active force the cell generates, options are
        'constant' and 'actin-poly'.
    """

    # needed more than once
    grad_x, grad_y, _ = compute_gradients(cell.phi, cell.simbox.dx)
    grad_phi = np.array([grad_x, grad_y])
    eta = cell.eta

    # phi_(n+1)
    phi_i_next, dF_dphi = _update_field(cell, grad_phi, force)

    # theta_(n+1)
    if cell.polarity_mode == "SVA":
        theta_i_next = cell.theta + polarity.static_velocity_aligning(cell)

    elif cell.polarity_mode == "DVA":
        theta_i_next = cell.theta + polarity.dynamic_velocity_aligning(cell)

    else:
        raise ValueError(f"{cell.polarity_mode} invalid.")

    # compute motility forces at time n
    if force_modality == "constant":
        fx_motil, fy_motil = force.constant_motility_force(cell, alpha=0.5)

    else:
        warnings.warn(
            f"force_modality == {force_modality} is not understood. \
                Defaulting to fx_motil = 0, fy_motil = 0... BE AWARE!"
        )
        fx_motil, fy_motil = 0.0, 0.0

    # compute thermodynamic forces at time n
    fx_thermo = dF_dphi * grad_x
    fy_thermo = dF_dphi * grad_y

    # UPDATE class variables now
    cell.phi = phi_i_next
    cell.theta = theta_i_next
    cell.cm = compute_CM(cell)
    cell.v_cm = compute_v_CM(cell)
    cell.vx = (fx_thermo + fx_motil) / eta
    cell.vy = (fy_thermo + fy_motil) / eta


def _update_field(cell, grad_phi, force):

    # obtain relevant variables at time n
    dt = cell.simbox.dt
    phi_i = cell.phi
    vx_i = cell.vx
    vy_i = cell.vy

    # compute equation of motion at time n
    dF_dphi = force.total_func_deriv(cell)
    grad_x, grad_y = grad_phi
    v_dot_gradphi = vx_i * grad_x + vy_i * grad_y

    return phi_i - dt * (dF_dphi + v_dot_gradphi), dF_dphi
