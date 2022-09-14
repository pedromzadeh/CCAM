import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class Figure:
    def __init__(self):
        pass

    @classmethod
    def view_simbox(cls, cells, chi, path):

        L = cells[0].simbox.L_box
        for cell in cells:
            plt.imshow(
                cell.phi, cmap="Greys", origin="lower", extent=[0, L, 0, L], alpha=0.5
            )

            plt.contour(cell.phi, levels=[0.5], extent=[0, L, 0, L], colors="black")

            xcm, ycm = cell.cm[1]
            px, py = [np.cos(cell.theta), np.sin(cell.theta)]
            vx, vy = cell.v_cm
            rx, ry = cell.r_CR

            plt.quiver(
                xcm,
                ycm,
                px,
                py,
                angles="xy",
                scale_units="xy",
                color="blue",
                label="Polarity",
                alpha=0.7,
            )

            plt.quiver(
                xcm,
                ycm,
                vx,
                vy,
                angles="xy",
                scale_units="xy",
                color="red",
                label="CM Velocity",
                alpha=0.7,
            )

            if rx != 0 and ry != 0:
                plt.quiver(
                    xcm,
                    ycm,
                    rx,
                    ry,
                    angles="xy",
                    scale_units="xy",
                    color="black",
                    label=r"$r_{{CR}}$",
                    alpha=0.7,
                )

        plt.contour(chi, levels=[0.5], extent=[0, L, 0, L])
        plt.ylim([0, 20])
        plt.savefig(path)
        plt.close()
