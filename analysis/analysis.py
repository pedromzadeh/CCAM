import pandas as pd
import numpy as np
from glob import glob
import psutil
import os
from multiprocessing import Pool
import statsmodels.stats.proportion as smp

import matplotlib.pyplot as plt

plt.style.use("../configs/stylesheet.mplstyle")


def outcomes(files, n_workers=psutil.cpu_count()):
    """
    Reads collision results in parallel and stores them in a dataframe.

    Parameters
    ----------
    files : list of str
        List of all paths to .csv files.

    n_workers : int
        Specifies the number of cores to use, by default the total number available.

    Returns
    -------
    pd.DataFrame
        Outcome of all collisions.

        Columns are:
        [
            gamma -- tension of varying cell,

            A -- adhesion of varying cell,

            beta -- protrusion strength of varying cell,

            dv -- pre-collision average CM speed difference in mu/min,

            dtheta -- pre-collision average contact angle difference in degree,

            Pwin -- 0/1 denoting loss or win
        ]

        There are 462 * 96 == 44,352 records
    """
    with Pool(processes=n_workers) as p:
        res = p.map_async(_collision_outcome, files)
        p.close()
        p.join()

    res = res.get(timeout=1)
    table = [
        pd.DataFrame([el], columns=["gamma", "A", "beta", "dv", "dtheta", "Pwin"])
        for el in res
    ]
    return pd.concat(table).reset_index(drop=True)


def two_feature_plots(binary_outcomes):
    """
    Makes a 3x3 plot showcasing the win probability, diff in CM speed,
    and diff in contact angle as a function of two features with the
    third held fixed.

    Parameters
    ----------
    binary_outcomes : pd.DataFrame
        Outcome of all collisions.
    """
    # default feature values
    GAMMA = 1.26
    A = 0.48
    BETA = 6

    # the complete span of feature-space
    betas = np.linspace(4, 10, 6)
    gammas = np.linspace(0.9, 1.8, 7)
    As = np.linspace(0.32, 0.64, 11)

    # two feature states we want to visit
    two_feat_sets = [
        dict([("beta", betas), ("A", As)]),
        dict([("beta", betas), ("gamma", gammas)]),
        dict([("gamma", gammas), ("A", As)]),
    ]

    # feature to hold constant
    const_feat_set = [("gamma", 1.2), ("A", 0.48), ("beta", 6.4)]

    # find bounds
    min_dv, max_dv, min_dtheta, max_dtheta = _bounds(
        binary_outcomes, two_feat_sets, const_feat_set
    )

    # set up figure
    plt.figure(figsize=(15, 12))
    cmap = "PuBu"
    n_levels = 6

    # make 1x3 plots
    for i, (features, const_feat) in enumerate(zip(two_feat_sets, const_feat_set)):
        res = _two_feature_stats(binary_outcomes, features, const_feat)
        x, y = res["x"], res["y"]
        zs = [res["Pwin"], res["dv"], res["dtheta"]]
        js = np.array([0, 1, 2]) + 3 * i
        ls = [
            np.linspace(0, 1, n_levels),
            np.linspace(min_dv, max_dv, n_levels),
            np.linspace(min_dtheta, max_dtheta, n_levels),
        ]
        titles = "ABCDEFGHI"
        z_feats = ["Pwin", "dv", "dtheta"] * 3
        formats = ["%.1f", "%.2f", "%.0f"] * 3
        x_0 = [BETA, BETA, BETA, BETA, BETA, BETA, GAMMA, GAMMA, GAMMA]
        y_0 = [A, A, A, GAMMA, GAMMA, GAMMA, A, A, A]

        for j, z, l in zip(js, zs, ls):
            plt.subplot(3, 3, j + 1)
            plt.contourf(x / x_0[j], y / y_0[j], z, levels=l, cmap=cmap, alpha=0.7)
            x_feat, y_feat = features.keys()
            plt.xlabel(_label(x_feat))
            plt.ylabel(_label(y_feat))
            plt.title(f"({titles[j]})", loc="left", x=-0.3)
            cbar = plt.colorbar(format=formats[j])
            cbar.set_label(_label(z_feats[j]))

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5, hspace=0.3)


def predictor_plots(agg_outcomes, binary_ac_outcomes):
    """
    Plots:
    1. Pwin as a function of diff(contact angle) and diff(CM speed),
    2. diff(contact angle) vs diff(CM speed).
    3. Logistic regressions on points where dtheta and dv anti correlate.


    Parameters
    ----------
    agg_outcomes : pd.DataFrame
        Outcome of collisions, aggregated over the 96 runs per point in feature-space.

    binary_ac_outcomes : pd.DataFrame
        Outcome of all collisions, confined to where dv and dtheta
        anti correlate. Used for fitting LR.

    Returns
    -------
    plt.Figure
    """
    min_dtheta = agg_outcomes.dtheta.min()
    max_dtheta = agg_outcomes.dtheta.max()
    min_dv = agg_outcomes.dv.min()
    max_dv = agg_outcomes.dv.max()

    # associate colors to each quadrant of each point in (dv, dtheta) space
    colors = [plt.cm.get_cmap("Set1")(j) for j in agg_outcomes.quadrant]

    # set up the figure
    fig = plt.figure(figsize=(14, 9))
    ax1 = plt.subplot2grid(shape=(2, 6), loc=(0, 0), colspan=2)
    ax2 = plt.subplot2grid((2, 6), (0, 2), colspan=2)
    ax3 = plt.subplot2grid((2, 6), (0, 4), colspan=2)
    ax4 = plt.subplot2grid((2, 6), (1, 1), colspan=2)
    ax5 = plt.subplot2grid((2, 6), (1, 3), colspan=2)

    # Pwin(dtheta)
    ax1.set_title("(A)", loc="left", x=-0.25, y=1.05)
    ax1.set_xlabel(_label("dtheta"))
    ax1.set_ylabel(_label("Pwin"))
    ax1.scatter(agg_outcomes["dtheta"], agg_outcomes["Pwin"], color=colors, alpha=0.7)

    # Pwin(dv)
    ax2.set_title("(B)", loc="left", x=-0.25, y=1.05)
    ax2.set_xlabel(_label("dv"))
    ax2.set_ylabel(_label("Pwin"))
    ax2.scatter(agg_outcomes["dv"], agg_outcomes["Pwin"], color=colors, alpha=0.7)

    # dtheta(dv)
    ax3.set_title("(C)", loc="left", x=-0.25, y=1.05)
    ax3.set_xlabel(_label("dv"))
    ax3.set_ylabel(_label("dtheta"))
    ax3.scatter(agg_outcomes["dv"], agg_outcomes["dtheta"], color=colors, alpha=0.7)

    # focus on anti-correlated space now
    agg_ac_outcomes = agg_outcomes.query("quadrant == 0 or quadrant == 2")
    colors = [plt.cm.get_cmap("Set1")(j) for j in agg_ac_outcomes.quadrant]

    # Pwin(dtheta) LR on anti-correlated space
    x = binary_ac_outcomes["dtheta"]
    y = binary_ac_outcomes["Pwin"]
    lr_model = train_lr(x, y)
    x = np.linspace(min_dtheta, max_dtheta, 500)
    Pwin_pred = lr_model.predict_proba(x.reshape(-1, 1))[:, 1]
    ax4.set_title("(D)", loc="left", x=-0.25, y=1.05)
    ax4.set_xlabel(_label("dtheta"))
    ax4.set_ylabel(_label("Pwin"))
    ax4.scatter(
        agg_ac_outcomes["dtheta"], agg_ac_outcomes["Pwin"], color=colors, alpha=0.7
    )
    ax4.plot(x, Pwin_pred, lw=4, color="black")
    ax4.set_ylim(0, 1)

    # Pwin(dv) LR
    x = binary_ac_outcomes["dv"]
    y = binary_ac_outcomes["Pwin"]
    lr_model = train_lr(x, y)
    x = np.linspace(min_dv, max_dv, 500)
    Pwin_pred = lr_model.predict_proba(x.reshape(-1, 1))[:, 1]
    ax5.set_title("(E)", loc="left", x=-0.25, y=1.05)
    ax5.set_xlabel(_label("dv"))
    ax5.set_ylabel(_label("Pwin"))
    ax5.scatter(agg_ac_outcomes["dv"], agg_ac_outcomes["Pwin"], color=colors, alpha=0.7)
    ax5.plot(x, Pwin_pred, lw=4, color="black")
    ax5.set_ylim(0, 1)

    # format spaces
    plt.subplots_adjust(wspace=0.9, hspace=0.45)

    return fig


def Pwin_plot(
    predictor,
    agg_outcomes,
    binary_ac_outcomes=None,
    fit_lr=False,
    conf_int=False,
    legend=False,
):
    """
    Plots winning probability as a function of the given predictor. You can also
    choose to draw the logistic curve and the confidence intervals.

    Parameters
    ----------
    predictor : str
        Specifies a feature as the predictor.

    agg_outcomes : pd.DataFrame
        Outcome of collisions, aggregated over the 96 runs per point in feature-space.

    binary_ac_outcomes : pd.DataFrame, optional
        Outcome of all collisions, confined to where dv and dtheta
        anti correlate. Needed if 'fit_lr==True'.

    fit_lr : bool, optional
        Specifies whether a logistic curve will be fitted and drawn,
        by default False

    conf_int : bool, optional
        Specifies whether 95% binomial confidence intervals are drawn,
        by default False

    legend : bool, optional
        Specifies whether the legend is drawn, by default False
    """
    # associate colors to each quadrant of each point in (dv, dtheta) space
    colors = [plt.cm.get_cmap("Set1")(j) for j in agg_outcomes.quadrant]
    plt.figure(figsize=(5, 5))
    plt.scatter(agg_outcomes[predictor], agg_outcomes["Pwin"], color=colors, alpha=0.7)
    plt.ylabel(_label("Pwin"))
    plt.xlabel(_label(predictor))
    plt.ylim(0, 1)

    # train LR on anti-correlated data
    if fit_lr:
        assert binary_ac_outcomes is not None

        min_val = agg_outcomes[predictor].min()
        max_val = agg_outcomes[predictor].max()
        x = binary_ac_outcomes[predictor]
        y = binary_ac_outcomes["Pwin"]
        lr_model = train_lr(x, y)
        x = np.linspace(min_val, max_val, 500).reshape(-1, 1)
        Pwin_pred = lr_model.predict_proba(x)[:, 1]
        plt.plot(x, Pwin_pred, lw=4, color="black", label="LR fit")
        # compute binomial confidence intervals
        if conf_int:
            n_obs = 96
            ci_low, ci_upp = smp.proportion_confint(
                Pwin_pred * n_obs, n_obs, alpha=0.05, method="beta"
            )

            plt.plot(x, ci_low, lw=4, color="black", linestyle="dashed", alpha=0.5)
            plt.plot(
                x,
                ci_upp,
                lw=4,
                color="black",
                linestyle="dashed",
                alpha=0.5,
                label="95% CI",
            )

    if legend:
        plt.legend()


def simbox_view(file):
    """
    Interactive drawing of a .png.

    Parameters
    ----------
    file : str
        Path to a .png file.
    """
    from IPython.display import Image

    return Image(file, retina=True)


def train_lr(x, y):
    """
    Train an L2 regularized logistic curve.

    Parameters
    ----------
    x : arraylike
        The training data, dv or dtheta values.

    y : arraylike
        Associated winning probability. Binary such that 0 -- lose, 1 -- win.

    Returns
    -------
    sklearn.linear_model.LogisticRegression
        A trained model.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0, shuffle=True
    )
    x_train = np.array(x_train).reshape(-1, 1)

    model = LogisticRegression(penalty="l2", solver="liblinear")

    model.fit(x_train, y_train)
    return model


def _quadrant(x):
    if x.dv > 0 and x.dtheta > 0:
        return 1
    elif x.dv < 0 and x.dtheta > 0:
        return 2
    elif x.dv < 0 and x.dtheta < 0:
        return 3
    else:
        return 4


def _label(feature_name):
    if feature_name == "gamma":
        return "Rel. tension"

    if feature_name == "beta":
        return "Rel. protrusion strength"

    if feature_name == "A":
        return "Rel. adhesion strength"

    if feature_name == "dv":
        return "$\delta v\ [\mathrm{\mu m/min}]$"

    if feature_name == "dtheta":
        return "$\delta \\theta\ [\degree]$"

    if feature_name == "Pwin":
        return "$P_{win}$"


def _bounds(binary_outcomes, two_feat_sets, const_feat_set):
    min_dv, max_dv = 100, -100
    min_dtheta, max_dtheta = 100, -100
    for features, const_feat in zip(two_feat_sets, const_feat_set):
        res = _two_feature_stats(binary_outcomes, features, const_feat)

        local_min, local_max = res["dv"].min(), res["dv"].max()
        min_dv = local_min if local_min < min_dv else min_dv
        max_dv = local_max if local_max > max_dv else max_dv

        local_min, local_max = res["dtheta"].min(), res["dtheta"].max()
        min_dtheta = local_min if local_min < min_dtheta else min_dtheta
        max_dtheta = local_max if local_max > max_dtheta else max_dtheta

    return min_dv, max_dv, min_dtheta, max_dtheta


def _collision_outcome(file):
    """
    Summarizes the outcome of one collision from its temporal history.

    Parameters
    ----------
    file : str
        Path to .csv file storing temporal stats for one simulation.

    Returns
    -------
    list
        [
            tension of varying cell,
            adhesion of varying cell,
            protrusion strength of varying cell,
            pre-collision average CM speed difference in mu/min,
            pre-collision average contact angle difference in degree,
            0/1 denoting loss or win
        ]
    """
    df = pd.read_csv(os.path.join(file, "result.csv"))

    # fixing negative contact angles
    df["contact angle"] = df["contact angle"].apply(lambda x: 180 + x if x < 0 else x)

    v_scale = 6 / 8  # mu/min conversion
    df["CM speed"] = df["CM speed"].apply(lambda x: v_scale * x)

    # compute pre-collision speed and contact angle, then difference
    dv, dtheta = np.diff(
        df.groupby("cell id")[["CM speed", "contact angle"]].apply("mean").values,
        axis=0,
    )[0]

    # decide whether Pwin 1 or 0
    n_left = len(df[df["coll orientation"] == "left"])
    n_records = len(df)
    Pwin = 1 if n_left > n_records / 2 else 0

    return [df.gamma.iloc[-1], df.A.iloc[-1], df.beta.iloc[-1], dv, dtheta, Pwin]


def _all_files(path_to_data):
    file_path = os.path.join(path_to_data, "grid_id*/run_*")
    files = glob(file_path)
    return files


def _two_feature_stats(binary_outcome, features, const_feat):
    """
    Computes the mean win probability, difference in CM speed,
    and difference in contact angle, averaged over 96 runs for
    the given set of 2D features with the third held fixed.

    Parameters
    ----------
    binary_outcome : pd.DataFrame
        Outcome of all collisions.
    features : dict
        Keys -- str, denoting features to plot on the x and y axes, respectively
        Vals -- ndarray, denoting the span of each feature.
    const_feat : tuple
        (name of feature to hold constant, value to assign)

    Returns
    -------
    dict
        You get the following keys:
           - x, y : values of features
           - Pwin, dv, dtheta : averages over the 96 runs
    """
    # cf: feature to keep constant
    # cv: value of this feature
    cf, cv = const_feat
    df = binary_outcome[binary_outcome[cf] == cv]

    # x and y features, in that order
    x_feat, y_feat = features.keys()
    x_feat_vals, y_feat_vals = features.values()

    # group data by the relevant features and aggregate by mean
    df = df.groupby([y_feat, x_feat]).apply("mean")

    # build the observables in this two-feature space
    dv_mat, dtheta_mat, Pwin_mat = [], [], []

    for _, df in df.groupby(y_feat):
        dv_mat.append(df.dv.values)
        dtheta_mat.append(df.dtheta.values)
        Pwin_mat.append(df.Pwin.values)

    return {
        "x": x_feat_vals,
        "y": y_feat_vals,
        "Pwin": np.array(Pwin_mat),
        "dv": np.array(dv_mat),
        "dtheta": np.array(dtheta_mat),
    }


def _ffcr_det_sol(theta_cr, ax=None, lb=None, plot_legend=False):
    """
    Solves the ffcr deterministic model.

    Parameters
    ----------
    theta_cr : float
        Specifies the target repolarization angle, in degrees.

    ax : plt.Axes
        Axis to plot to, by default None.

    lb : str
        Title of the axis, by default None.

    plot_legend : bool, optional
        Whether to draw legend, by default False.

    Returns
    -------
    ndarray
        Time to repolarize, i.e. reach pi/2.
    """

    # hyperparameters
    dt = 0.002
    N = 5000
    tau_cr = 1.5
    tau_cg = 2 * tau_cr
    tau_ffcr = 10e10
    theta_cr = theta_cr * np.pi / 180
    theta_cg = 0
    SOLVE_FFCR = True
    REAL_UNITS_TIME = 8  # min

    tau_1 = tau_cr * tau_cg / (tau_cr - tau_cg)
    tau_2 = tau_cr * tau_cg / (tau_cr + tau_cg)

    x = [np.cos(theta_cg), np.sin(theta_cg)]
    r = [np.cos(theta_cr), np.sin(theta_cr)]

    # numerical evolution of theta
    theta = [0 * np.pi / 180]
    for i in range(N):
        t = theta[i]
        pos = [np.cos(t), np.sin(t)]
        cross_cg = np.cross(x, pos)
        cross_cr = np.cross(r, pos)
        sgn = 1 if t < np.pi / 2 else -1
        tau_cr_eff = tau_ffcr if (t > np.pi / 2 and SOLVE_FFCR) else tau_cr
        dtheta = (
            -1 / tau_cg * np.arcsin(sgn * cross_cg) * dt
            - 1 / tau_cr_eff * np.arcsin(cross_cr) * dt
        )
        theta.append(t + dtheta)

    time = np.arange(N + 1) * dt

    # ----------------- theoretical curves ----------------- #
    # each leg's steady-state
    PHI_1 = (tau_cr * theta_cg + tau_cg * (np.pi - theta_cr)) / (tau_cr - tau_cg)
    PHI_2 = (theta_cg * tau_cr + theta_cr * tau_cg) / (tau_cr + tau_cg)
    PHI_3 = (
        np.pi
        if SOLVE_FFCR
        else (np.pi * tau_cr + theta_cr * tau_cg) / (tau_cr + tau_cg)
    )

    # t_prime, t_dprime: times we transition from legs 1-2, 2-3
    # term1 = np.pi*tau_cg + tau_cr*(np.pi+2*theta_cg-2*theta_cr)
    # term2 = 2 * (tau_cg*(np.pi-theta_cr) + tau_cr*theta_cg)
    # t_prime = (tau_cg*tau_cr)/(tau_cg-tau_cr) * np.log(term1/term2)

    # term1 = np.pi*(tau_cg+tau_cr) + 2*tau_cr*(theta_cg-theta_cr)
    # term2 = 2*(theta_cr*tau_cg+theta_cg*tau_cr) - np.pi*(tau_cg+tau_cr)
    # t_dprime = t_prime + tau_2 * np.log(term1/term2)

    term1 = np.pi * (tau_cg + tau_cr) - 2 * theta_cr * tau_cr
    term2 = 2 * tau_cg * (np.pi - theta_cr)
    t_prime = (tau_cg * tau_cr) / (tau_cg - tau_cr) * np.log(term1 / term2)

    term1 = np.pi * (tau_cg + tau_cr) - 2 * tau_cr * theta_cr
    term2 = 2 * theta_cr * tau_cg - np.pi * (tau_cg + tau_cr)
    t_dprime = t_prime + tau_2 * np.log(term1 / term2)

    # print('t\' = {}\tPhi_2 = {}'.format(t_prime,PHI_2*180/np.pi))
    # print('t\'\' = {}\tPhi_3 = {}'.format(t_dprime,PHI_3*180/np.pi))

    # computing each leg analytically
    leg1 = PHI_1 * (1 - np.exp(-time / tau_1))
    shift = theta_cr - np.pi / 2
    leg2 = PHI_2 + (shift - PHI_2) * np.exp(-(time - t_prime) / tau_2)
    leg3 = PHI_3 + (np.pi / 2 - PHI_3) * np.exp(-(time - t_dprime) / tau_cg)

    # plot each leg, along with numerical result
    if ax is not None:
        t1 = np.where(np.fabs(time - t_prime) < dt)[0][0]
        t2 = np.where(np.fabs(time - t_dprime) < dt)[0][0]
        ax.set_title(lb, x=-0.11, y=1.1)
        ax.plot(
            time * REAL_UNITS_TIME,
            np.array(theta) * 180 / np.pi,
            label="numerical",
            lw=6,
            c="black",
        )
        ax.plot(
            time[:t1] * REAL_UNITS_TIME,
            leg1[:t1] * 180 / np.pi,
            label=r"$\psi_1(t)$",
            lw=2,
            c="red",
        )
        ax.plot(
            time[t1:t2] * REAL_UNITS_TIME,
            leg2[t1:t2] * 180 / np.pi,
            label=r"$\psi_2(t)$",
            lw=2,
            color="orange",
        )
        ax.plot(
            time[t2:] * REAL_UNITS_TIME,
            leg3[t2:] * 180 / np.pi,
            label=r"$\psi_3(t)$",
            lw=2,
            color="magenta",
        )
        ax.plot(
            time * REAL_UNITS_TIME,
            np.ones(N + 1) * (theta_cr * 180 / np.pi - 90),
            ls="--",
            color="black",
            lw=3,
            label=r"$\psi_{\rm CR}-\pi/2$",
        )
        ax.plot(
            time * REAL_UNITS_TIME,
            np.ones(N + 1) * (90),
            ls="-.",
            color="black",
            lw=3,
            label=r"$\pi/2$",
        )
        ax.set_xlabel(r"$t\ [\rm min]$")
        ax.set_ylabel(r"$\psi[\degree]$")
        ax.set_ylim(0, 200)
        if plot_legend:
            ax.legend(loc=(1.1, 0.001))

    return t_dprime * REAL_UNITS_TIME
