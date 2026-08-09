#!/usr/bin/env python3
# =============================================================================
# TAB-DT: Trust-Age Weighted Bayesian Digital Twin -- full simulation study
# Paper: "Trust--Age Bayesian Digital Twins: Provable Predictive Maintenance
#         under Industrial Network Impairments"
# Physics-first protocol: every stochastic model term is validated against its
# closed-form statistic before any comparative experiment is trusted.
# =============================================================================
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (Arc, Circle, FancyArrowPatch, FancyBboxPatch,
                                Polygon, Rectangle)
from scipy.optimize import brentq
from scipy.stats import norm

rng_global = np.random.default_rng(20260805)

# ---------------------------- global parameters ------------------------------
DT      = 1.0          # sampling interval Delta t [h]
MU      = 0.02         # true degradation drift [HI/h]
SIGB    = 0.03         # Wiener diffusion coefficient [HI/sqrt(h)]
Q       = SIGB**2 * DT # process-noise variance per step
D_FAIL  = 10.0         # failure threshold on health index
X0      = 0.0
R_BASE  = 1.0          # nominal measurement-noise variance (trust = 1)
TAUS    = np.array([1.0, 0.7, 0.4])   # sensor trust scores
S       = len(TAUS)
Q_MU    = 1e-7         # drift random-walk variance (filter model)
MU0_HAT = 0.02         # fleet-mean prior drift
SIG_MU0 = 0.004        # unit-to-unit drift heterogeneity
MU_MIN  = 0.01         # physical lower support for the random drift [HI/h]
P0      = np.array([[0.25, 0.0], [0.0, SIG_MU0**2]])
ETA_STAR = Q / R_BASE  # variance-matched age-decay rate (= 9e-4 per step)
KMAX    = 1500

# ---------------------------- plotting style ---------------------------------
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "lines.linewidth": 1.45, "figure.dpi": 300, "axes.grid": True,
    "grid.alpha": 0.32, "grid.color": "#C9D2DC", "grid.linewidth": 0.45,
    "legend.framealpha": 0.96, "legend.edgecolor": "#B8C2CC",
    "mathtext.fontset": "dejavusans", "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
# High-contrast, colour-blind distinguishable palette.  Colour and marker both
# encode method identity so printed grayscale copies remain interpretable.
C = {"B1": "#6A3D9A", "B2": "#D62728", "B3": "#1565C0", "TAB": "#009E49",
     "th": "#111111", "extra": "#FF8C00"}
MRK = {"B1": "^", "B2": "s", "B3": "o", "TAB": "D"}
LBL = {"B1": "B1: open-loop DT", "B2": "B2: stale-as-fresh",
       "B3": "B3: discard-stale", "TAB": "TABDT (proposed)"}

FIGDIR = "figs"
import os
os.makedirs(FIGDIR, exist_ok=True)


def savefig(fig, name):
    fig.savefig(f"{FIGDIR}/{name}.pdf", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(f"{FIGDIR}/{name}.png", bbox_inches="tight", pad_inches=0.025,
                dpi=400)
    plt.close(fig)


def style_axes(ax, grid_axis="both"):
    """Apply one publication style to every quantitative panel."""
    ax.grid(True, axis=grid_axis)
    ax.spines["left"].set_color("#566573")
    ax.spines["bottom"].set_color("#566573")
    ax.tick_params(length=3.0, width=0.65, color="#566573")


# =============================================================================
# 1. Ground-truth generator (vectorised over N Monte-Carlo units)
# =============================================================================
def simulate_truth(N, rng, heterogeneous=True):
    """N Wiener paths with positive unit-specific random drift."""
    mu_n = rng.normal(MU, SIG_MU0, N) if heterogeneous else np.full(N, MU)
    if heterogeneous:
        invalid = mu_n < MU_MIN
        while invalid.any():
            mu_n[invalid] = rng.normal(MU, SIG_MU0, invalid.sum())
            invalid = mu_n < MU_MIN
    x = np.full(N, X0)
    X = np.empty((KMAX + 1, N)); X[0] = x
    fail_k = np.full(N, -1, dtype=int)
    alive = np.ones(N, bool)
    for k in range(1, KMAX + 1):
        x = x + (mu_n * DT + rng.normal(0.0, np.sqrt(Q), N)) * alive
        X[k] = x
        newly = alive & (x >= D_FAIL)
        fail_k[newly] = k
        alive &= ~newly
        if not alive.any():
            X = X[:k + 1]
            break
    fail_k[fail_k < 0] = X.shape[0] - 1
    return X, fail_k


# =============================================================================
# 2. Network layer: geometric delay with a finite delivery deadline.
#    A packet generated at step k by sensor s reaches the twin at step
#    k + a_{k,s}, where a ~ Geometric0(p); packets exceeding A_MAX are dropped.
#    P(a=0) = p is the on-time synchronization probability. Common uniform
#    draws couple different p values without changing the marginal law.
# =============================================================================
A_MAX = 100
DROP_AGE = A_MAX + 1


def delays_from_uniform(delay_u, p):
    """Inverse-CDF geometric delays; DROP_AGE marks a deadline loss."""
    if p >= 1.0:
        return np.zeros_like(delay_u, dtype=np.int16)
    raw = np.floor(np.log1p(-delay_u) / np.log1p(-p)).astype(np.int32)
    return np.where(raw <= A_MAX, raw, DROP_AGE).astype(np.int16)


def markov_delays(chan_u, p, mean_outage):
    """Two-state Markov (Gilbert-Elliott) link shared by each sensor stream.

    A slot is Good with stationary probability p; a packet generated at slot
    k waits for the first Good slot at or after k, so its age is the residual
    outage time.  mean_outage sets the mean Bad-run length 1/r; the choice
    mean_outage = 1/p reproduces the marginal Geometric0(p) age law while
    coupling packets that share the link.  Common uniforms in chan_u couple
    the compared outage regimes.
    """
    r = 1.0 / mean_outage                    # P(Bad -> Good)
    q = r * (1.0 - p) / p                    # P(Good -> Bad) keeps pi_G = p
    if q > 1.0:
        raise ValueError("mean_outage too short for stationary fraction p")
    K1 = chan_u.shape[0]
    good = np.empty(chan_u.shape, dtype=bool)
    good[0] = chan_u[0] < p
    for t in range(1, K1):
        good[t] = np.where(good[t - 1], chan_u[t] < 1.0 - q, chan_u[t] < r)
    nxt = np.full(chan_u.shape[1:], 10 ** 9, dtype=np.int64)
    ages = np.empty(chan_u.shape, dtype=np.int64)
    for k in range(K1 - 1, -1, -1):
        nxt = np.where(good[k], k, nxt)
        ages[k] = nxt - k
    return np.where(ages <= A_MAX, ages, DROP_AGE).astype(np.int16)


def oracle_reprocess(X, fail_k, delays, noise_std, stride=25):
    """Exact delayed-data benchmark under the twin's own model.

    At each evaluated step t the posterior is recomputed by one Kalman pass
    over every packet that has physically arrived by t, sorted by generation
    time, with exact per-sensor noise.  This is the exact conditional law the
    forward first-order correction approximates, at O(t) instead of O(1)
    cost per step.  Packets are admitted exactly when the sequential
    recursions admit them, namely when the arrival slot lies in [1, t].
    """
    K1, N = X.shape
    R_true = R_BASE / TAUS
    Y = X[:, None, :] + noise_std * np.sqrt(R_true)[None, :, None]
    gen = np.arange(K1, dtype=np.int64)[:, None, None]
    arrive = np.where(delays <= A_MAX, gen + delays, 10 ** 9)
    evals = np.arange(0, K1, stride)
    est = {k: np.full((len(evals), N), np.nan) for k in
           ("Xhat", "Muhat", "Pxx", "Pmm", "Pxm")}
    for i, t in enumerate(evals):
        z = np.tile(np.array([X0, MU0_HAT]), (N, 1))
        P = np.tile(P0, (N, 1, 1))
        for s in range(S):
            mask = (arrive[0, s] >= 1) & (arrive[0, s] <= t)
            z, P = kf_update_masked(z, P, Y[0, s], np.full(N, R_true[s]), mask)
        for k in range(1, t + 1):
            z, P = kf_predict(z, P)
            for s in range(S):
                mask = arrive[k, s] <= t
                if mask.any():
                    z, P = kf_update_masked(
                        z, P, Y[k, s], np.full(N, R_true[s]), mask)
        est["Xhat"][i] = z[:, 0]; est["Muhat"][i] = z[:, 1]
        est["Pxx"][i] = P[:, 0, 0]; est["Pmm"][i] = P[:, 1, 1]
        est["Pxm"][i] = P[:, 0, 1]
    return evals, est


def grid_rmse(rul_grid, evals, fail_k, lo=0.2, hi=0.9):
    """RMSE of RUL estimates given on the subsampled grid rows."""
    true_rul = fail_k[None, :] - evals[:, None]
    win = (evals[:, None] >= lo * fail_k[None, :]) & \
          (evals[:, None] <= hi * fail_k[None, :])
    err2 = np.where(win, (rul_grid - true_rul) ** 2, np.nan)
    return float(np.sqrt(np.nanmean(err2)))


# =============================================================================
# 3. Estimators.  State z = [x, mu]^T, A = [[1, dt], [0, 1]].
#    All four share the same prediction; they differ in how delayed packets
#    enter the update.  Every packet is used at most once.
# =============================================================================
A_MAT = np.array([[1.0, DT], [0.0, 1.0]])
QMAT  = np.array([[Q, 0.0], [0.0, Q_MU]])


def kf_predict(z, P):
    z = z @ A_MAT.T
    P = A_MAT @ P @ A_MAT.T + QMAT      # broadcasting over (N,2,2)
    return z, P


def kf_update_masked(z, P, y, Rk, mask):
    """Sequential scalar update with H = [1,0], applied where mask is True."""
    if not mask.any():
        return z, P
    Sk = P[:, 0, 0] + Rk
    K0 = P[:, 0, 0] / Sk
    K1 = P[:, 1, 0] / Sk
    innov = y - z[:, 0]
    m = mask.astype(float)
    z = z.copy()
    z[:, 0] += m * K0 * innov
    z[:, 1] += m * K1 * innov
    Kv = np.stack([K0, K1], axis=1)
    # Joseph-form covariance update preserves symmetry and positive
    # semidefiniteness under finite-precision sequential packet fusion.
    eye = np.broadcast_to(np.eye(2), P.shape).copy()
    KH = np.zeros_like(P)
    KH[:, :, 0] = Kv
    J = eye - KH
    Pj = J @ P @ np.swapaxes(J, 1, 2) + Rk[:, None, None] * (
        Kv[:, :, None] * Kv[:, None, :])
    P = np.where(mask[:, None, None], Pj, P)
    P = 0.5 * (P + np.swapaxes(P, 1, 2))
    return z, P


def run_filters(X, fail_k, p, rng=None, eta=ETA_STAR,
                methods=("B1", "B2", "B3", "TAB"),
                delay_u=None, noise_std=None, delays=None):
    """Run the requested estimators on shared truth, delays and noises."""
    K, N = X.shape[0] - 1, X.shape[1]
    if delay_u is None and delays is None:
        delay_u = rng.random((K + 1, S, N))
    if noise_std is None:
        noise_std = rng.normal(0.0, 1.0, (K + 1, S, N))
    if delays is None:
        delays = delays_from_uniform(delay_u, p)        # (K+1,S,N)
    R_true = R_BASE / TAUS
    noise = noise_std * np.sqrt(R_true)[None, :, None]
    Y = X[:, None, :] + noise

    out = {}
    for meth in methods:
        z = np.tile(np.array([X0, MU0_HAT]), (N, 1))
        P = np.tile(P0, (N, 1, 1))
        Xhat = np.empty((K + 1, N)); Xhat[0] = z[:, 0]
        Muhat = np.empty((K + 1, N)); Muhat[0] = z[:, 1]
        Pxx = np.empty((K + 1, N)); Pxx[0] = P[:, 0, 0]
        Pmm = np.empty((K + 1, N)); Pmm[0] = P[:, 1, 1]
        Pxm = np.empty((K + 1, N)); Pxm[0] = P[:, 0, 1]
        for t in range(1, K + 1):
            z, P = kf_predict(z, P)
            if meth != "B1":
                lmax = min(t, A_MAX) if meth != "B3" else 0
                for lag in range(lmax + 1):
                    k = t - lag
                    for s in range(S):
                        mask = delays[k, s] == lag
                        if not mask.any():
                            continue
                        if meth == "B3":
                            y_use = Y[k, s]
                            R_use = np.full(N, R_true[s])
                        elif meth == "B2":
                            y_use = Y[k, s]                # stale-as-fresh
                            R_use = np.full(N, R_true[s])
                        else:  # TAB-DT: R/w = R/tau_s + eta*a*R
                            y_use = Y[k, s] + z[:, 1] * lag * DT
                            w = TAUS[s] / (1.0 + TAUS[s] * eta * lag)
                            R_use = R_BASE / max(w, 1e-9) \
                                    + (lag * DT) ** 2 * P[:, 1, 1]
                        z, P = kf_update_masked(z, P, y_use, R_use, mask)
            Xhat[t] = z[:, 0]; Muhat[t] = z[:, 1]
            Pxx[t] = P[:, 0, 0]; Pmm[t] = P[:, 1, 1]
            Pxm[t] = P[:, 0, 1]
        out[meth] = dict(Xhat=Xhat, Muhat=Muhat, Pxx=Pxx, Pmm=Pmm, Pxm=Pxm)
    out["delays"] = delays
    return out
# =============================================================================
# 4. RUL inference + metrics
# =============================================================================
def rul_stats(Xhat, Muhat, Pxx, Pmm, Pxm):
    mu = np.maximum(Muhat, 1e-4)
    m = np.maximum(D_FAIL - Xhat, 1e-6)
    rul = m / mu
    var_aleat = m * SIGB**2 / mu**3                  # IG variance
    var_epist = (Pxx / mu**2 + (m / mu**2) ** 2 * Pmm
                 + 2.0 * m * Pxm / mu**3)             # full delta method
    sd = np.sqrt(np.maximum(var_aleat + var_epist, 1e-12))
    return rul, sd


def eval_run(res, X, fail_k, lo=0.2, hi=0.9):
    """RMSE and 90 % PICP over the [20%,90%] life window of each unit."""
    K1, N = X.shape
    metrics = {}
    ks = np.arange(K1)[:, None]
    true_rul = fail_k[None, :] - ks
    win = (ks >= lo * fail_k[None, :]) & (ks <= hi * fail_k[None, :])
    for meth in ("B1", "B2", "B3", "TAB"):
        if meth not in res:
            continue
        r = res[meth]
        rul, sd = rul_stats(r["Xhat"], r["Muhat"], r["Pxx"], r["Pmm"],
                            r["Pxm"])
        err = rul - true_rul
        rmse = np.sqrt(np.nanmean(np.where(win, err**2, np.nan)))
        cover = np.abs(err) <= 1.645 * sd
        picp = np.nanmean(np.where(win, cover.astype(float), np.nan))
        metrics[meth] = dict(rmse=float(rmse), picp=float(picp))
    return metrics


# =============================================================================
# 5. Theorem-1 validation: scalar known-mu KF, single sensor, Bernoulli loss
# =============================================================================
def theorem1_bound(p, Rn=R_BASE, Qn=Q):
    return (Qn + np.sqrt(Qn**2 + 4 * p * Qn * Rn)) / (2 * p)


def validate_theorem1(ps, N=4000, K=3000, rng=None):
    emp_P, emp_mse = [], []
    for p in ps:
        x = np.zeros(N); xh = np.zeros(N); Pv = np.full(N, 0.25)
        acc_P = 0.0; acc_e = 0.0; cnt = 0
        for k in range(K):
            w = rng.normal(0, np.sqrt(Q), N)
            x = x + MU * DT + w
            xh = xh + MU * DT
            Pv = Pv + Q                                   # prior
            if k > K // 2:                                # steady state only
                acc_P += Pv.mean(); acc_e += ((x - xh) ** 2).mean(); cnt += 1
            got = rng.random(N) < p
            y = x + rng.normal(0, np.sqrt(R_BASE), N)
            Kg = Pv / (Pv + R_BASE)
            xh = np.where(got, xh + Kg * (y - xh), xh)
            Pv = np.where(got, (1 - Kg) * Pv, Pv)
        emp_P.append(acc_P / cnt); emp_mse.append(acc_e / cnt)
    return np.array(emp_P), np.array(emp_mse)


# =============================================================================
# 6. Maintenance policy (Proposition 1): analytic cost rate + MC validation
# =============================================================================
def cost_rate_analytic(xi, p, cp=1.0, cf=10.0):
    Pbar = theorem1_bound(p)
    pf = norm.sf((D_FAIL - xi) / np.sqrt(Pbar))
    return (cp + (cf - cp) * pf) * MU / xi


def threshold_residual(xi, p, cp=1.0, cf=10.0):
    """First-order residual F(xi)=xi*h'(xi)-h(xi) from Proposition 1."""
    s = np.sqrt(theorem1_bound(p))
    d = cf - cp
    r = (D_FAIL - xi) / s
    h = cp + d * norm.sf(r)
    return xi * d * norm.pdf(r) / s - h


def optimal_threshold(p, cp=1.0, cf=10.0, tol=1e-10):
    """Unique interior root when it exists; otherwise compare boundaries."""
    lo = 1e-8
    if threshold_residual(D_FAIL, p, cp, cf) <= 0.0:
        candidates = np.array([lo, D_FAIL])
        costs = np.array([cost_rate_analytic(x, p, cp, cf) for x in candidates])
        return float(candidates[np.argmin(costs)])
    return float(brentq(lambda x: threshold_residual(x, p, cp, cf),
                        lo, D_FAIL, xtol=tol, rtol=tol))


def policy_mc(xi, p, N=1500, rng=None, cp=1.0, cf=10.0):
    """Single-sensor, known-mu twin operating a control-limit PM policy."""
    x = np.zeros(N); xh = np.zeros(N); Pv = np.full(N, 0.25)
    done = np.zeros(N, bool); cost = np.zeros(N); tend = np.zeros(N)
    for k in range(1, KMAX * 2):
        w = rng.normal(0, np.sqrt(Q), N)
        x = np.where(done, x, x + MU * DT + w)
        xh = np.where(done, xh, xh + MU * DT)
        Pv = np.where(done, Pv, Pv + Q)
        got = (rng.random(N) < p) & ~done
        y = x + rng.normal(0, np.sqrt(R_BASE), N)
        Kg = Pv / (Pv + R_BASE)
        xh = np.where(got, xh + Kg * (y - xh), xh)
        Pv = np.where(got, (1 - Kg) * Pv, Pv)
        fail = ~done & (x >= D_FAIL)
        pm = ~done & ~fail & (xh >= xi)
        cost[fail] = cf; cost[pm] = cp
        tend[fail | pm] = k * DT
        done |= fail | pm
        if done.all():
            break
    return cost.sum() / tend.sum()


# =============================================================================
# 7. Figure 1: architecture diagram
# =============================================================================
def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.12, 2.55))
    ax.set_xlim(0, 120); ax.set_ylim(0, 43); ax.axis("off"); ax.grid(False)
    navy, teal, amber, rose, violet = "#173B6C", "#1F8A7A", "#D98C10", "#B54A63", "#6857A6"

    def stage(x, y, w, h, title, fc, accent):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.38",
                                    fc=fc, ec=accent, lw=1.0))
        ax.add_patch(Rectangle((x, y + h - 3.5), w, 3.5, fc=accent, ec=accent))
        ax.text(x + w / 2, y + h - 1.75, title, ha="center", va="center",
                fontsize=6.4, color="white", weight="bold")

    def flow(x1, y1, x2, y2, label="", color=navy, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=9, lw=1.05, ls=ls,
                                     color=color, connectionstyle="arc3,rad=0"))
        if label:
            ax.text((x1 + x2) / 2, y1 + 1.5, label, ha="center", va="bottom",
                    fontsize=5.7, color=color)

    # 1) Physical degradation asset.
    stage(1.0, 13.0, 18.0, 23.0, "PHYSICAL ASSET", "#FFF9EC", amber)
    ax.add_patch(Circle((7.0, 26.0), 3.8, ec=amber, fc="white", lw=1.1))
    ax.add_patch(Circle((7.0, 26.0), 1.35, ec=amber, fc="#FFF0C9", lw=0.9))
    for ang in np.linspace(0, 2*np.pi, 7)[:-1]:
        ax.add_patch(Circle((7 + 2.55*np.cos(ang), 26 + 2.55*np.sin(ang)),
                            0.48, ec=amber, fc="#FFE2A3", lw=0.65))
    ax.plot([11.5, 13.3, 15.0, 17.0], [21.0, 23.0, 25.3, 30.5],
            color=rose, lw=1.35, marker="o", ms=2.2)
    ax.axhline(31.2, xmin=0.095, xmax=0.145, color=rose, ls="--", lw=0.75)
    ax.text(10.0, 17.2, "$x_{k+1}=x_k+\\mu\\Delta t+\\varepsilon_k$",
            fontsize=6.4, color="#243746")
    ax.text(10.0, 14.7, "failure at $x_k\\geq D$", fontsize=5.8, color=rose)

    # 2) Heterogeneous sensor layer.
    stage(23.0, 13.0, 17.0, 23.0, "SENSOR LAYER", "#EEF7FF", navy)
    for i, (yy, tau) in enumerate(zip([29.0, 24.3, 19.6], [1.0, 0.7, 0.4])):
        ax.add_patch(Circle((27.0, yy), 1.55, fc="white", ec=navy, lw=0.9))
        ax.add_patch(Arc((27.0, yy), 1.7, 1.7, theta1=35, theta2=325,
                         color=teal, lw=1.0))
        ax.text(33.2, yy, f"s{i+1}:  $\\tau_s={tau:.1f}$", ha="center",
                va="center", fontsize=5.8, color="#243746")
    ax.text(31.5, 15.2, "$y_k^{(s)}=x_k+v_k^{(s)}$", ha="center",
            fontsize=6.3, color=navy)

    # 3) Deadline-limited packet network.
    stage(44.0, 11.0, 20.0, 27.0, "PACKET NETWORK", "#FFF0F3", rose)
    for yy, delay in zip([29.5, 24.7, 19.9], ["$a=0$", "$a=3$", "$a>a_{\\max}$"]):
        ax.add_patch(Rectangle((47.0, yy-1.1), 2.3, 2.2, fc="white", ec=rose, lw=0.8))
        ax.plot([49.5, 58.5], [yy, yy], color=rose,
                ls="--" if "max" in delay else "-", lw=0.85)
        if "max" in delay:
            ax.plot([54.5, 56.0], [yy-1.0, yy+1.0], color=rose, lw=1.0)
            ax.plot([54.5, 56.0], [yy+1.0, yy-1.0], color=rose, lw=1.0)
        else:
            ax.add_patch(Polygon([[58.5, yy], [57.2, yy+0.75], [57.2, yy-0.75]],
                                 closed=True, fc=rose, ec=rose))
        ax.text(60.5, yy, delay, ha="center", va="center", fontsize=6.0)
    ax.text(54.0, 14.0, "$A\\sim\\mathrm{Geom}_0(p)$, deadline loss",
            ha="center", fontsize=6.0, color="#243746")

    # 4) TAB-DT inference engine with explicit mathematical stages.
    stage(68.0, 7.5, 30.0, 34.0, "TABDT BAYESIAN TWIN", "#EFFAF7", teal)
    modules = [
        ("1  AGE COMPENSATION", "$\\tilde y=y_k^{(s)}+\\hat\\mu_t a\\Delta t$"),
        ("2  TRUST AND AGE FUSION", "$w_s(a)=\\tau_s/(1+\\tau_s\\eta^*a)$"),
        ("3  POSTERIOR UPDATE", "$p(\\mathbf{z}_t\\mid\\mathcal{Y}_t),\\; \\widehat{\\mathrm{RUL}}_t\\pm1.645\\sigma_t$")]
    for j, (head, eq) in enumerate(modules):
        yy = 31.5 - 8.0*j
        ax.add_patch(FancyBboxPatch((71.0, yy-4.6), 24.0, 6.2,
                                    boxstyle="round,pad=0.25", fc="white",
                                    ec="#87BDB4", lw=0.75))
        ax.text(72.2, yy+0.4, head, fontsize=5.5, color=teal, weight="bold")
        ax.text(83.0, yy-2.2, eq, ha="center", fontsize=6.1, color="#243746")

    # 5) RUL-informed maintenance action.
    stage(102.0, 13.0, 17.0, 23.0, "MAINTENANCE", "#F4F1FC", violet)
    ax.add_patch(Arc((110.5, 26.8), 9.0, 9.0, theta1=15, theta2=165,
                     color=violet, lw=1.5))
    for a in np.linspace(25, 155, 5):
        xx = 110.5 + 4.2*np.cos(np.deg2rad(a)); yy = 26.8 + 4.2*np.sin(np.deg2rad(a))
        ax.plot([xx], [yy], marker="o", ms=1.8, color=violet)
    ax.add_patch(FancyArrowPatch((110.5, 26.8), (113.3, 29.5),
                                 arrowstyle="-|>", mutation_scale=8,
                                 lw=1.2, color=rose))
    ax.text(110.5, 20.0, "trigger if $\\hat x_t\\geq\\xi^*(p)$", ha="center",
            fontsize=6.1, color="#243746")
    ax.text(110.5, 16.5, "earlier PM as $p\\downarrow$", ha="center",
            fontsize=5.8, color=violet)

    # Main information path and lower theory-to-decision certificate path.
    flow(19.0, 25.0, 23.0, 25.0, "$x_k$")
    flow(40.0, 25.0, 44.0, 25.0, "packets")
    flow(64.0, 25.0, 68.0, 25.0, "$\\mathcal{A}_t$")
    flow(98.0, 25.0, 102.0, 25.0, "RUL posterior")
    ax.add_patch(FancyBboxPatch((41.0, 0.7), 24.0, 6.6, boxstyle="round,pad=0.3",
                                fc="#F3F6F9", ec=navy, lw=0.8))
    ax.text(53.0, 4.0, "network certificate\n$\\bar P(p)$  (Theorem 1)",
            ha="center", va="center", fontsize=5.8, color=navy, linespacing=1.25)
    ax.add_patch(FancyBboxPatch((70.0, 0.7), 24.0, 6.6, boxstyle="round,pad=0.3",
                                fc="#F4F1FC", ec=violet, lw=0.8))
    ax.text(82.0, 4.0, "PM control limit\n$\\xi^*(p)$  (Proposition 1)",
            ha="center", va="center", fontsize=5.8, color=violet, linespacing=1.25)
    flow(65.0, 4.0, 70.0, 4.0, color=violet)
    flow(94.0, 4.0, 110.5, 13.0, color=violet)
    ax.add_patch(FancyArrowPatch((110.5, 13.0), (10.0, 13.0), arrowstyle="-|>",
                                 mutation_scale=8, lw=0.9, ls="--", color="#657786",
                                 connectionstyle="arc3,rad=-0.08"))
    ax.text(24.0, 8.7, "maintenance action / renewal", fontsize=5.7,
            color="#657786", ha="center")
    savefig(fig, "fig1_architecture")


def fig_trajectory_only():
    """Regenerate Fig. 2 independently with its fixed seed."""
    rng2 = np.random.default_rng(42)
    X1, fk1 = simulate_truth(1, rng2)
    res1 = run_filters(X1, fk1, p=0.4, rng=rng2)
    k_ax = np.arange(X1.shape[0]) * DT
    fig, axs = plt.subplots(2, 1, figsize=(3.52, 2.62), sharex=True,
                            gridspec_kw=dict(hspace=0.12))
    ax = axs[0]
    ax.plot(k_ax, X1[:, 0], color=C["th"], lw=1.25, label="true $x_k$")
    xh = res1["TAB"]["Xhat"][:, 0]; sx = np.sqrt(res1["TAB"]["Pxx"][:, 0])
    ax.plot(k_ax, xh, color=C["TAB"], lw=1.35, label="TABDT $\\hat{x}_k$")
    ax.plot(k_ax, xh + 2 * sx, color=C["B3"], ls="--", lw=0.85,
            label="$\\pm2\\sigma$ limits")
    ax.plot(k_ax, xh - 2 * sx, color=C["B3"], ls="--", lw=0.85)
    ax.axhline(D_FAIL, color=C["B2"], ls="--", lw=1.0)
    ax.text(468, D_FAIL + 0.45, "failure threshold $D$", color=C["B2"],
            fontsize=6.5, ha="right")
    dly = res1["delays"][:, :, 0]
    on_time = (dly[:X1.shape[0]] == 0).any(axis=1)
    ax.plot(k_ax[on_time], np.full(on_time.sum(), -0.7), "|", color="0.4",
            ms=3, label="packets on time")
    ax.set_ylabel("health index [HI]")
    ax.legend(loc="upper left", ncol=2, borderaxespad=0.2)
    ax.set_ylim(-1.3, 16.2)
    style_axes(ax)
    ax = axs[1]
    tr = fk1[0] - np.arange(X1.shape[0])
    rul, sd = rul_stats(res1["TAB"]["Xhat"], res1["TAB"]["Muhat"],
                        res1["TAB"]["Pxx"], res1["TAB"]["Pmm"],
                        res1["TAB"]["Pxm"])
    ax.plot(k_ax, tr, color=C["th"], lw=1.25, label="true RUL")
    ax.plot(k_ax, rul[:, 0], color=C["TAB"], lw=1.35, label="$\\widehat{\\mathrm{RUL}}_k$")
    ax.plot(k_ax, rul[:, 0] + 1.645 * sd[:, 0], color=C["B3"], ls="--",
            lw=0.85, label="90% limits")
    ax.plot(k_ax, rul[:, 0] - 1.645 * sd[:, 0], color=C["B3"], ls="--", lw=0.85)
    ax.set_xlabel("time $k\\Delta t$ [h]"); ax.set_ylabel("RUL [h]")
    ax.set_ylim(0, 800); ax.legend(loc="upper right")
    style_axes(ax)
    savefig(fig, "fig2_trajectories")


# =============================================================================
# 8. Main experiment pipeline
# =============================================================================
def main():
    rng = np.random.default_rng(7)
    results = {}

    # ---------- physics validation --------------------------------------------
    Xv, fk = simulate_truth(20000, rng, heterogeneous=False)
    mean_life, std_life = fk.mean() * DT, fk.std() * DT
    ig_mean = D_FAIL / MU
    ig_std = np.sqrt(D_FAIL * SIGB**2 / MU**3)
    print(f"[VAL] mean failure time: MC={mean_life:.1f} h  IG theory={ig_mean:.1f} h")
    print(f"[VAL] std  failure time: MC={std_life:.1f} h  IG theory={ig_std:.1f} h")
    results["validation"] = dict(mc_mean=float(mean_life), ig_mean=ig_mean,
                                 mc_std=float(std_life), ig_std=float(ig_std))

    # ---------- Fig 2: illustrative trajectories ------------------------------
    fig_trajectory_only()

    # ---------- E2/E4: RMSE and PICP versus sync probability ------------------
    ps = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    NMC = 400
    rmse = {m: [] for m in LBL}; rmse_se = {m: [] for m in LBL}
    picp = {m: [] for m in LBL}; picp_se = {m: [] for m in LBL}
    Xp, fkp = simulate_truth(NMC, rng)
    K1, N = Xp.shape
    delay_u = rng.random((K1, S, N))
    noise_std = rng.normal(0.0, 1.0, (K1, S, N))
    # Retain the E2 population for the burst and oracle studies below; the
    # aliases consume no random draws, so E1-E6 outputs are unchanged.
    Xp2, fkp2, delay_u2, noise_std2 = Xp, fkp, delay_u, noise_std
    for p in ps:
        resp = run_filters(Xp, fkp, p, eta=ETA_STAR,
                           delay_u=delay_u, noise_std=noise_std)
        met = eval_run(resp, Xp, fkp)
        # per-unit RMSE for standard errors
        K1, N = Xp.shape
        ks = np.arange(K1)[:, None]
        true_rul = fkp[None, :] - ks
        win = (ks >= 0.2 * fkp[None, :]) & (ks <= 0.9 * fkp[None, :])
        for m in LBL:
            r = resp[m]
            ru, sdv = rul_stats(r["Xhat"], r["Muhat"], r["Pxx"], r["Pmm"],
                                r["Pxm"])
            err2 = np.where(win, (ru - true_rul) ** 2, np.nan)
            per_unit = np.sqrt(np.nanmean(err2, axis=0))
            covered = np.where(win, (np.abs(ru - true_rul) <= 1.645 * sdv), np.nan)
            per_unit_cover = np.nanmean(covered, axis=0)
            rmse[m].append(met[m]["rmse"])
            rmse_se[m].append(per_unit.std() / np.sqrt(N))
            picp[m].append(met[m]["picp"])
            picp_se[m].append(per_unit_cover.std() / np.sqrt(N))
        print(f"[E2] p={p:.2f} " + " ".join(
            f"{m}:{met[m]['rmse']:.1f}h" for m in LBL))
    results["rmse_vs_p"] = {m: list(map(float, rmse[m])) for m in LBL}
    results["rmse_se"] = {m: list(map(float, rmse_se[m])) for m in LBL}
    results["picp_vs_p"] = {m: list(map(float, picp[m])) for m in LBL}
    results["picp_se"] = {m: list(map(float, picp_se[m])) for m in LBL}
    results["ps"] = ps.tolist()

    fig, ax = plt.subplots(figsize=(3.52, 2.35))
    for m in LBL:
        y = np.array(rmse[m]); se = np.array(rmse_se[m])
        ax.errorbar(ps, y, yerr=1.96 * se, marker=MRK[m], ms=4.0,
                    capsize=2.2, capthick=0.8, elinewidth=0.8,
                    color=C[m], label=LBL[m])
    ax.set_xscale("log"); ax.set_xticks(ps)
    ax.set_xticklabels([".02", ".05", ".1", ".2", ".3", ".5", ".7", "1"])
    ax.set_xlabel("packet synchronization probability $p$")
    ax.set_ylabel("RUL RMSE [h]"); ax.set_ylim(48, 128)
    ax.annotate("$-27.9\\%$ vs B2\n$-14.7\\%$ vs B3",
                xy=(0.02, rmse["TAB"][0]), xytext=(0.032, 105),
                arrowprops=dict(arrowstyle="->", lw=0.75, color=C["TAB"]),
                fontsize=6.2, color=C["TAB"], ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#B8C2CC", lw=0.5))
    ax.legend(loc="center right", ncol=1, borderaxespad=0.35); style_axes(ax)
    savefig(fig, "fig3_rmse_vs_p")

    fig, ax = plt.subplots(figsize=(3.52, 2.20))
    for m in LBL:
        y = np.array(picp[m]) * 100
        se = np.array(picp_se[m]) * 100
        ax.errorbar(ps, y, yerr=1.96 * se, marker=MRK[m], ms=4.0,
                    capsize=2.2, capthick=0.8, elinewidth=0.8,
                    color=C[m], label=LBL[m])
    ax.axhline(90, color="k", ls="--", lw=0.8)
    ax.text(0.72, 91, "nominal 90%", fontsize=6.5)
    ax.set_xscale("log"); ax.set_xticks(ps)
    ax.set_xticklabels([".02", ".05", ".1", ".2", ".3", ".5", ".7", "1"])
    ax.set_xlabel("packet synchronization probability $p$")
    ax.set_ylabel("empirical 90% PICP [%]"); ax.set_ylim(88, 101.6)
    ax.legend(loc="upper right", ncol=2, borderaxespad=0.25)
    style_axes(ax)
    savefig(fig, "fig4_picp")

    # ---------- E3: Theorem-1 bound validation --------------------------------
    ps_t = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    empP, empMSE = validate_theorem1(ps_t, rng=np.random.default_rng(11))
    Pth = theorem1_bound(ps_t)
    results["thm1"] = dict(ps=ps_t.tolist(), empP=empP.tolist(),
                           empMSE=empMSE.tolist(), bound=Pth.tolist())
    fig, ax = plt.subplots(figsize=(3.52, 2.20))
    ax.loglog(ps_t, Pth, "-", color=C["th"], label="bound $\\bar{P}(p)$, Thm. 1")
    ax.loglog(ps_t, empP, "o", ms=3.6, color=C["TAB"],
              label="empirical $\\mathbb{E}[P_{k|k-1}]$")
    ax.loglog(ps_t, empMSE, "s", ms=3.4, mfc="none", color=C["B3"],
              label="empirical MSE of $\\hat{x}$")
    pref = np.sqrt(Q * R_BASE / ps_t)
    ax.loglog(ps_t, pref, ":", color="0.5", lw=0.9,
              label="$\\sqrt{QR/p}$ moderate-loss approx.")
    ax.set_xlabel("synchronization probability $p$")
    ax.set_ylabel("steady-state variance [HI$^2$]")
    ax.legend(loc="lower left", fontsize=6.0, borderaxespad=0.3,
              labelspacing=0.28, handlelength=1.6, handletextpad=0.5,
              borderpad=0.3)
    style_axes(ax); savefig(fig, "fig5_theorem1")
    print("[E3] bound/emp ratio:", np.round(Pth / empP, 3))

    # ---------- E5: sensitivity to age-decay rate eta -------------------------
    etas = np.array([0.0, ETA_STAR / 100, ETA_STAR / 10, ETA_STAR,
                     ETA_STAR * 10, ETA_STAR * 100])
    p_sens = [0.05, 0.2, 0.5]
    sens = np.zeros((len(p_sens), len(etas)))
    sens_se = np.zeros_like(sens)
    Xp, fkp = simulate_truth(400, rng)
    K1, N = Xp.shape
    delay_u_s = rng.random((K1, S, N))
    noise_std_s = rng.normal(0.0, 1.0, (K1, S, N))
    for i, p in enumerate(p_sens):
        ks = np.arange(K1)[:, None]
        true_rul = fkp[None, :] - ks
        win = (ks >= 0.2 * fkp[None, :]) & (ks <= 0.9 * fkp[None, :])
        for j, eta in enumerate(etas):
            resp = run_filters(Xp, fkp, p, eta=eta, methods=("TAB",),
                               delay_u=delay_u_s, noise_std=noise_std_s)
            r = resp["TAB"]
            ru, _ = rul_stats(r["Xhat"], r["Muhat"], r["Pxx"], r["Pmm"],
                              r["Pxm"])
            err2 = np.where(win, (ru - true_rul) ** 2, np.nan)
            per_unit = np.sqrt(np.nanmean(err2, axis=0))
            sens[i, j] = np.sqrt(np.nanmean(err2))
            sens_se[i, j] = per_unit.std() / np.sqrt(N)
        print(f"[E5] p={p}: " + " ".join(f"{v:.1f}" for v in sens[i]))
    results["sensitivity"] = dict(etas=etas.tolist(), p=p_sens,
                                  rmse=sens.tolist(), se=sens_se.tolist())
    fig, ax = plt.subplots(figsize=(3.52, 2.5))
    kappas = etas / ETA_STAR
    mk = ["o", "s", "D"]
    for i, p in enumerate(p_sens):
        ax.errorbar(kappas, sens[i], yerr=1.96 * sens_se[i], marker=mk[i],
                    ms=3.4, capsize=2, lw=1.0, elinewidth=0.7,
                    color=[C["B2"], C["B3"], C["TAB"]][i], label=f"$p={p}$")
    ax.axvline(1.0, color="k", ls="--", lw=0.8)
    ax.text(1.25, ax.get_ylim()[1] * 0.99,
            "$\\eta^{*}=Q/R$", fontsize=6.6, va="top")
    ax.set_xscale("symlog", linthresh=5e-3)
    ax.set_xlim(-3e-3, 140)
    ax.set_xticks(kappas)
    ax.set_xticklabels(["0", "$10^{-2}$", "$10^{-1}$", "1", "10", "$10^2$"])
    ax.set_xlabel("variance multiplier $\\kappa=\\eta/\\eta^{*}$")
    ax.set_ylabel("RUL RMSE [h]"); ax.legend(loc="center left")
    style_axes(ax)
    savefig(fig, "fig6_sensitivity")

    # ---------- E6: maintenance threshold (Proposition 1) ---------------------
    xis = np.linspace(6.5, 9.8, 34)
    p_cost = [0.2, 0.5, 0.9]
    cost_curves, cost_mc, xi_star = [], [], []
    for p in p_cost:
        Ca = np.array([cost_rate_analytic(x, p) for x in xis])
        cost_curves.append(Ca)
        xi_star.append(optimal_threshold(p))
        xi_pts = xis[::5]
        mc = [policy_mc(x, p, rng=np.random.default_rng(int(1000 * p)))
              for x in xi_pts]
        cost_mc.append((xi_pts.tolist(), mc))
        print(f"[E6] p={p}: xi*={xi_star[-1]:.2f}, C(xi*)="
              f"{cost_rate_analytic(xi_star[-1], p)*1000:.3f} x1e-3/h")
    results["cost"] = dict(xis=xis.tolist(), p=p_cost,
                           analytic=[c.tolist() for c in cost_curves],
                           mc=cost_mc, xi_star=xi_star)
    fig, ax = plt.subplots(figsize=(3.52, 2.30))
    cols = [C["B2"], C["B3"], C["TAB"]]
    for i, p in enumerate(p_cost):
        ax.plot(xis, cost_curves[i] * 1e3, color=cols[i], label=f"$p={p}$ (analytic)")
        xp, mc = cost_mc[i]
        ax.plot(xp, np.array(mc) * 1e3, "o", ms=3.2, mfc="none", color=cols[i])
        ax.plot(xi_star[i], cost_rate_analytic(xi_star[i], p) * 1e3,
                "*", ms=8, color=cols[i])
    ax.plot([], [], "o", mfc="none", color="0.3", label="Monte Carlo")
    ax.plot([], [], "*", ms=8, color="0.3", label="$\\xi^{*}$ (Prop. 1)")
    ax.set_xlabel("PM control limit $\\xi$ [HI]")
    ax.set_ylabel("cost rate $C(\\xi)\\times10^{3}$ [cost units/h]")
    ax.legend(loc="upper left", ncol=2, borderaxespad=0.3)
    style_axes(ax); savefig(fig, "fig7_cost")

    # ---------- headline table operating points -------------------------------
    results["headline_table"] = {
        f"p={p:.2f}": {m: dict(rmse=rmse[m][i], picp=picp[m][i]) for m in LBL}
        for i, p in enumerate(ps) if p in (0.02, 0.05, 0.20)
    }

    # ---------- E7: correlated outages (Gilbert-Elliott link) -----------------
    # Stress test outside the independent-delay design model: the timely
    # fraction is held at p = 0.05 while the mean outage length grows from
    # the marginally geometric value 1/p to fourfold that value.  Truths and
    # measurement noises are the E2 population; channel uniforms are common
    # across outage regimes.
    p_burst = 0.05
    outages = [1.0 / p_burst, 40.0, 80.0]
    K1, N = Xp2.shape
    chan_u = np.random.default_rng(20260807).random((K1, S, N))
    ks = np.arange(K1)[:, None]
    true_rul = fkp2[None, :] - ks
    win = (ks >= 0.2 * fkp2[None, :]) & (ks <= 0.9 * fkp2[None, :])
    burst = {m: [] for m in ("B2", "B3", "TAB")}
    burst_se = {m: [] for m in ("B2", "B3", "TAB")}
    burst_del, burst_age = [], []
    for mo in outages:
        delays_b = markov_delays(chan_u, p_burst, mo)
        delivered = delays_b <= A_MAX
        burst_del.append(float(delivered.mean()))
        burst_age.append(float(delays_b[delivered].mean()))
        resb = run_filters(Xp2, fkp2, p_burst, methods=("B2", "B3", "TAB"),
                           noise_std=noise_std2, delays=delays_b)
        for m in burst:
            r = resb[m]
            ru, _ = rul_stats(r["Xhat"], r["Muhat"], r["Pxx"], r["Pmm"],
                              r["Pxm"])
            err2 = np.where(win, (ru - true_rul) ** 2, np.nan)
            per_unit = np.sqrt(np.nanmean(err2, axis=0))
            burst[m].append(float(np.sqrt(np.nanmean(err2))))
            burst_se[m].append(float(per_unit.std() / np.sqrt(N)))
        print(f"[E7] outage={mo:.0f}: " + " ".join(
            f"{m}:{burst[m][-1]:.1f}h" for m in burst)
            + f" delivered={burst_del[-1]:.3f} age={burst_age[-1]:.1f}")
    results["burst"] = dict(p=p_burst, mean_outage=outages, rmse=burst,
                            se=burst_se, delivered=burst_del,
                            mean_age_delivered=burst_age,
                            channel_seed=20260807)

    # ---------- E8: exact reprocessing benchmark ------------------------------
    # The exact conditional posterior given every arrived packet, recomputed
    # by full Kalman reprocessing on a common 25-step grid, bounds what any
    # delayed-data rule can achieve under the twin's model.  p = 1 must and
    # does reproduce the sequential filter exactly.
    stride = 25
    oracle = dict(ps=[0.02, 0.05, 0.2, 1.0], stride=stride,
                  oracle_rmse=[], tab_rmse_grid=[], ratio=[])
    evals = np.arange(0, K1, stride)
    for p in oracle["ps"]:
        delays_p = delays_from_uniform(delay_u2, p)
        _, est = oracle_reprocess(Xp2, fkp2, delays_p, noise_std2,
                                  stride=stride)
        ru_o, _ = rul_stats(est["Xhat"], est["Muhat"], est["Pxx"],
                            est["Pmm"], est["Pxm"])
        rmse_o = grid_rmse(ru_o, evals, fkp2)
        rest = run_filters(Xp2, fkp2, p, methods=("TAB",),
                           delay_u=delay_u2, noise_std=noise_std2)["TAB"]
        ru_t, _ = rul_stats(rest["Xhat"], rest["Muhat"], rest["Pxx"],
                            rest["Pmm"], rest["Pxm"])
        rmse_t = grid_rmse(ru_t[evals], evals, fkp2)
        oracle["oracle_rmse"].append(rmse_o)
        oracle["tab_rmse_grid"].append(rmse_t)
        oracle["ratio"].append(rmse_t / rmse_o)
        print(f"[E8] p={p}: oracle={rmse_o:.1f}h TABDT={rmse_t:.1f}h "
              f"excess={100 * (rmse_t / rmse_o - 1):.1f}%")
    assert abs(oracle["ratio"][-1] - 1.0) < 5e-3, "p=1 oracle check failed"
    results["oracle"] = oracle

    # ---------- Fig: stress panels for E7 and E8 ------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(3.52, 2.05),
                            gridspec_kw=dict(wspace=0.46))
    ax = axs[0]
    for m in ("B2", "B3", "TAB"):
        ax.errorbar(outages, burst[m], yerr=1.96 * np.array(burst_se[m]),
                    marker=MRK[m], ms=3.6, capsize=2.0, capthick=0.8,
                    elinewidth=0.8, lw=1.3, color=C[m],
                    label={"B2": "B2", "B3": "B3", "TAB": "TABDT"}[m])
    ax.set_xscale("log", base=2)
    ax.set_xticks(outages); ax.set_xticklabels(["20", "40", "80"])
    ax.minorticks_off()
    for x, d in zip(outages, burst_del):
        ax.text(x, 52.4, f"{100 * d:.0f}%", ha="center", fontsize=5.8,
                color="0.35")
    ax.set_xlabel("mean outage [steps]")
    ax.set_ylabel("RUL RMSE [h]")
    ax.set_ylim(50, 112)
    ax.legend(loc="upper left", fontsize=6.0, borderaxespad=0.25,
              handlelength=1.5)
    style_axes(ax)
    ax = axs[1]
    exc = [100.0 * (r - 1.0) for r in oracle["ratio"]]
    ax.plot(oracle["ps"], exc, marker="D", ms=3.6, lw=1.3, color=C["TAB"])
    ax.set_xscale("log")
    ax.set_xticks(oracle["ps"])
    ax.set_xticklabels([".02", ".05", ".2", "1"])
    ax.minorticks_off()
    for x, y in zip(oracle["ps"], exc):
        ax.text(x, y + 0.65, f"{y:.1f}", ha="center", fontsize=5.8,
                color="0.25")
    ax.set_xlabel("sync. probability $p$")
    ax.set_ylabel("RMSE excess over exact [%]")
    ax.set_ylim(-0.9, 14.2)
    style_axes(ax)
    savefig(fig, "fig8_stress")

    with open("results.json", "w") as f:
        json.dump(results, f, indent=1)
    print("[DONE] results.json written")


if __name__ == "__main__":
    fig_architecture()
    main()
