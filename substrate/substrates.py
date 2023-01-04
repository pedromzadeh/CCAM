import numpy as np


class Substrate:
    """
    Implements various confinements within the phase-field framework.

    Attributes
    ----------
    self.N_mesh : int
        Number of lattice points.

    self.L_box : float
        Size of box.

    self.xi : float
        Specifies the interfacial thickness of the field.

    Methods
    -------
    __init__(self, N_mesh, L_box, xi)
        Initialize an instance.

    __str__(self)
        Format how to print the class instance.

    get_substrate(self, x, y, type)
        Generate the requested confinement.

    _rectangular(self, x, y)
        Returns a rectangular confinement.

    _line(self, y, yB=5)
        Returns a floor confinement at yB.

    _circular(self, x, y)
        Returns a circular confinement.

    _Y(self, x, y)
        Returns a Y-channel confinement.

    _plus(self, x, y)
        Returns a plus-channel confinement.
    """

    def __init__(self, N_mesh, L_box, xi=0.5):
        """
        Initialize an instance.

        Parameters
        ----------
        self.N_mesh : int
            Number of lattice points.

        self.L_box : float
            Size of box.

        self.xi : float, optional
            Specifies the interfacial thickness of the field, by default 0.5.

        """
        self.N_mesh = N_mesh
        self.L_box = L_box
        self.xi = xi

    def __str__(self):
        return "\t" + " + You are currently using the {} substrate.".format(self.type)

    def rectangular(self, xL, xR, yB, yT):
        """rectangular confinement"""
        # controls inteface width of wall
        eps = self.xi
        N_mesh = self.N_mesh
        L_box = self.L_box

        # lattice points
        x, y = np.meshgrid(np.linspace(0, L_box, N_mesh), np.linspace(0, L_box, N_mesh))

        # a rectangular sub
        chi_y = 0.5 * ((1 - np.tanh((y - yB) / eps)) + (1 + np.tanh((y - yT) / eps)))
        chi_x = 0.5 * ((1 - np.tanh((x - xL) / eps)) + (1 + np.tanh((x - xR) / eps)))
        chi = chi_x + chi_y

        return chi

    def line(self, yB=5):
        """floor confinement"""
        eps = self.xi
        N_mesh = self.N_mesh
        L_box = self.L_box

        # lattice points
        x, y = np.meshgrid(np.linspace(0, L_box, N_mesh), np.linspace(0, L_box, N_mesh))

        return 0.5 * (1 - np.tanh((y - yB) / eps))

    def circular(self, Rl=18, Rs=10):
        """circular confinement"""
        N_mesh = self.N_mesh
        L_box = self.L_box

        # lattice points
        x, y = np.meshgrid(np.linspace(0, L_box, N_mesh), np.linspace(0, L_box, N_mesh))

        x_center, y_center = L_box / 2, L_box / 2
        a, b, c = 1, 1, 1
        x = x - x_center
        y = y - y_center
        chi_sqrd = (x / a) ** 2 + (y / b) ** 2
        chi_sqrd *= c**2
        chi_1 = np.sqrt(chi_sqrd) - Rl
        chi_2 = -(np.sqrt(chi_sqrd) - Rs)
        chi_1 = 1 / (1 + np.exp(-chi_1))
        chi_2 = 1 / (1 + np.exp(-chi_2))
        chi = chi_1 + chi_2
        return chi

    def Y(self):
        """Y substrate"""
        eps = self.xi
        N_mesh = self.N_mesh
        L_box = self.L_box

        # lattice points
        x, y = np.meshgrid(np.linspace(0, L_box, N_mesh), np.linspace(0, L_box, N_mesh))

        width = 3
        eps = 0.5
        x = x - 25
        y = y - 25
        chiL = (
            1
            / 2
            * (
                (np.tanh((y - x + width + 2.5) / eps))
                + (-np.tanh((y - x - width) / eps))
            )
        )
        chiR = (
            1
            / 2
            * (
                (np.tanh((y + x + width + 2.5) / eps))
                + (-np.tanh((y + x - width) / eps))
            )
        )
        chiC = 1 - 1 / 2 * (
            (np.tanh((x + width) / eps)) + (-np.tanh((x - width) / eps))
        )
        chiL_trunc = np.zeros(chiL.shape)
        chiL_trunc[0:50, 0:50] = chiL[0:50, 0:50]
        chiR_trunc = np.zeros(chiR.shape)
        chiR_trunc[0:50, 50:100] = chiR[0:50, 50:100]
        chiC_trunc = np.ones(chiC.shape)
        chiC_trunc[50:100, :] = chiC[50:100, :]
        chi = chiC_trunc - chiR_trunc - chiL_trunc
        chi = np.where(chi < 0, 0, chi)
        return chi

    def plus(self):
        """+ substrate"""
        eps = self.xi
        N_mesh = self.N_mesh
        L_box = self.L_box

        # lattice points
        x, y = np.meshgrid(np.linspace(0, L_box, N_mesh), np.linspace(0, L_box, N_mesh))

        width = 3
        eps = 0.5
        x = x - 25
        y = y - 25
        chiH = 1 / 2 - 1 / 2 * (
            (np.tanh((x + width) / eps)) + (-np.tanh((x - width) / eps))
        )
        chiV = 1 / 2 - 1 / 2 * (
            (np.tanh((y + width) / eps)) + (-np.tanh((y - width) / eps))
        )
        chi = chiV + chiH
        chi = np.where(chi < 0.001, 0, chi)
        return chi

    def two_state_sub(self, square_width=10, bridge_width=5):
        # useful variables
        sq_half_dim = square_width / 2
        d = (square_width - bridge_width) / 2
        N_mesh = self.N_mesh
        L_box = self.L_box
        xi = self.xi

        # lattice points
        x, y = np.meshgrid(np.linspace(0, L_box, N_mesh), np.linspace(0, L_box, N_mesh))

        # index of the two squares, used for symmetric positioning
        ind = [1, 3]

        # build the bridge
        xL = ind[0] * L_box / 4 - sq_half_dim + 1
        xR = ind[1] * L_box / 4 + sq_half_dim - 1
        yB, yT = L_box / 2 - sq_half_dim + d, L_box / 2 + sq_half_dim - d
        chi_y = 0.5 * ((1 - np.tanh((y - yB) / xi)) + (1 + np.tanh((y - yT) / xi)))
        chi_x = 0.5 * ((1 - np.tanh((x - xL) / xi)) + (1 + np.tanh((x - xR) / xi)))
        bridge = chi_x + chi_y
        bridge = np.where(bridge > 1, 1, bridge)

        # build the two squares
        squares = None
        for k in range(2):
            xL = ind[k] * L_box / 4 - sq_half_dim
            xR = ind[k] * L_box / 4 + sq_half_dim
            yB, yT = L_box / 2 - sq_half_dim, L_box / 2 + sq_half_dim
            chi_y = 0.5 * ((1 - np.tanh((y - yB) / xi)) + (1 + np.tanh((y - yT) / xi)))
            chi_x = 0.5 * ((1 - np.tanh((x - xL) / xi)) + (1 + np.tanh((x - xR) / xi)))
            chi = chi_x + chi_y
            chi = np.where(chi > 1, 1, chi)
            if squares is None:
                squares = chi
            else:
                squares += chi

        # bridge + squares ==> two state substrate
        # refactoring so range is [0, 1]
        squares -= 1
        two_state = squares + bridge - 1
        two_state = np.where(two_state < 0, 0, two_state)

        return two_state
