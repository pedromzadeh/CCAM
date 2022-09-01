class SimulationBox():

    def __init__(self, L_box, N_mesh, dt):
        self.L_box = L_box
        self.N_mesh = N_mesh
        self.dt = dt
        self.dx = L_box / (N_mesh - 1)
