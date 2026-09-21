"""
CORE: Toy one-asset heterogeneous-agent model for the entropic-wedge diagnostic (extended description below).

The household block of a one-asset Aiyagari economy at quarterly frequency,
used as a controlled laboratory for the entropic-wedge diagnostic.

Households with CRRA preferences save in a single liquid asset subject to a
zero borrowing limit, and their labour income follows a two-state Markov
chain. The module computes the stationary equilibrium with the endogenous
grid method and the lotteries of Young (2010), and then solves
perfect-foresight transition paths in which households learn at date zero the
whole path of the real rate and of the income-process deformation, so that
consumption responds to announced future changes as well as to current ones.

The deformation scales the probability of switching income state for the
households whose wealth falls in a given region of the distribution, and
together with a misspecified path of the real rate it is what the diagnostic
is allowed to correct. Regions are wealth quantiles of equal mass, and the
function qbar returns for each region the expected number of income
transitions that originate there per household and per quarter, which is the
weight that makes tilting a crowded or frequently switching region more
expensive than tilting a thin one. The Jacobian of aggregate consumption with
respect to the rate path and to the regional deformations is built column by
column, perturbing each input at each date and solving the full nonlinear
transition. The cost of a correction, its decomposition by region and the
attribution rules are implemented in entropic.py.

Notice that the income transition of period t takes place at the end of the period, 
after the savings decision, so a household belongs to a region according to t
he wealth it carries into the next period. With this convention the expectation 
in the Euler equation is well defined at each point of the grid, and the cost 
of the deformation depends on the state in which the transition actually happens.
"""

import numpy as np

# calibration
BETA = 0.97          # quarterly discount factor
GAMMA = 2.0          # CRRA
R_SS = 0.01          # quarterly real rate (4% annual)
Y = np.array([0.70, 1.30])          # income states (mean 1)
PI0 = np.array([[0.925, 0.075],
                [0.075, 0.925]])    # baseline quarterly transition matrix
NZ = 2

NA = 120
A_MIN, A_MAX = 0.0, 60.0
AGRID = A_MIN + (A_MAX - A_MIN) * np.linspace(0.0, 1.0, NA) ** 2.5

T = 40     # truncation horizon for input paths
H = 24     # number of moment horizons (dC_t, t = 0..H-1)

U_PRIME = lambda c: c ** (-GAMMA)
U_PRIME_INV = lambda w: w ** (-1.0 / GAMMA)


# tilted transition matrices, state-dependent in end-of-period assets
def tilted_Pi(delta_on_grid):
    Pi = np.empty((NA, NZ, NZ))
    scale = 1.0 + delta_on_grid                      # (NA,)
    off = PI0 * (1.0 - np.eye(NZ))                   # off-diagonal part
    Pi[:] = off[None, :, :] * scale[:, None, None]
    rowsum_off = Pi.sum(axis=2)                      # (NA,NZ)
    for z in range(NZ):
        Pi[:, z, z] = 1.0 - rowsum_off[:, z]
    return Pi


# EGM backward step with time-varying r and Pi(a')
def egm_step(c_next, Pi_t, r_t, r_next):
    mu_next = U_PRIME(c_next)  # (NA,NZ)
    # W(a',z) = beta (1+r_{t+1}) sum_{z'} Pi_t[a',z,z'] u'(c_{t+1}(a',z'))
    W = BETA * (1.0 + r_next) * np.einsum('azx,ax->az', Pi_t, mu_next)
    c_endog = U_PRIME_INV(W)    # (NA,NZ) at a'
    coh_endog = c_endog + AGRID[:, None]   # cash on hand needed
    a_endog = (coh_endog - Y[None, :]) / (1.0 + r_t)
    g = np.empty((NA, NZ))
    for z in range(NZ):
        g[:, z] = np.interp(AGRID, a_endog[:, z], AGRID,
                            left=A_MIN, right=AGRID[-1]) # borrowing constraint
        g[AGRID < a_endog[0, z], z] = A_MIN
    c = (1.0 + r_t) * AGRID[:, None] + Y[None, :] - g
    return c, g


# Young (2010) lottery for the asset choice
def young_weights(g):
    idx = np.searchsorted(AGRID, g, side='right') - 1
    idx = np.clip(idx, 0, NA - 2)
    w_hi = (g - AGRID[idx]) / (AGRID[idx + 1] - AGRID[idx])
    w_hi = np.clip(w_hi, 0.0, 1.0)
    return idx, w_hi


def forward_step(mu, g, Pi_t):
    idx, w_hi = young_weights(g)
    mu_end = np.zeros((NA, NZ))
    for z in range(NZ):
        np.add.at(mu_end[:, z], idx[:, z], mu[:, z] * (1.0 - w_hi[:, z]))
        np.add.at(mu_end[:, z], idx[:, z] + 1, mu[:, z] * w_hi[:, z])
    mu_next = np.einsum('az,azx->ax', mu_end, Pi_t)
    return mu_next, mu_end


# steady state
def steady_state(tol=1e-12, maxit=10000):
    Pi_ss = tilted_Pi(np.zeros(NA))
    c = (R_SS * AGRID[:, None] + Y[None, :]) + 1e-2   # init
    for it in range(maxit):
        c_new, g = egm_step(c, Pi_ss, R_SS, R_SS)
        err = np.max(np.abs(c_new - c))
        c = c_new
        if err < tol:
            break
    mu = np.ones((NA, NZ)) / (NA * NZ)
    for it in range(maxit):
        mu_new, mu_end = forward_step(mu, g, Pi_ss)
        err = np.max(np.abs(mu_new - mu))
        mu = mu_new
        if err < 1e-14:
            break
    _, mu_end = forward_step(mu, g, Pi_ss)
    C_ss = float(np.sum(mu * c))
    A_ss = float(np.sum(mu * AGRID[:, None]))
    return dict(c=c, g=g, mu=mu, mu_end=mu_end, C=C_ss, A=A_ss,
                Pi=Pi_ss, egm_err=err)


# regions; wealth quantiles of the end-of-period measure
def make_regions(mu_end, K): # partition the a'-grid into K wealth groups of (approximately) equal mass under the end-of-period measure
    marg = mu_end.sum(axis=1)
    cum = np.cumsum(marg)
    cuts = np.array([cum.searchsorted(j / K, side='left')
                     for j in range(1, K)])
    masks = np.zeros((K, NA), dtype=bool)
    lo = 0
    for k in range(K):
        hi = cuts[k] + 1 if k < K - 1 else NA
        masks[k, lo:hi] = True
        lo = hi
    masses = masks @ marg
    return masks, masses


def qbar(mu_end, masks):
    """Pinned prior: expected per-capita number of transitions originating
    in each region per period (eq. I.10 with dt = 1)."""
    leave = 1.0 - np.diag(PI0) 
    flow = mu_end * leave[None, :] 
    return masks @ flow.sum(axis=1)   


# transition-path solver (nonlinear, anticipated paths)
def solve_path(ss, dr_path, delta_paths, masks, extra_mask=None, Hb=12): #dr_path: (T,) deviations of r_t; delta_paths: (K,T) regional tilts. Households at t=0 learn the whole path (perfect foresight)
    K = masks.shape[0]
    r_path = R_SS + np.asarray(dr_path)
    delta_grid = np.zeros((T, NA))
    for k in range(K):
        delta_grid[:, masks[k]] += np.asarray(delta_paths)[k][:, None]
    Pis = [tilted_Pi(delta_grid[t]) for t in range(T)]
    c_next = ss['c']
    cs, gs = [None] * T, [None] * T
    for t in range(T - 1, -1, -1):
        r_next = r_path[t + 1] if t + 1 < T else R_SS
        cs[t], gs[t] = egm_step(c_next, Pis[t], r_path[t], r_next)
        c_next = cs[t]
    mu = ss['mu'].copy()
    dC = np.empty(H)
    dCb = np.empty(Hb) if extra_mask is not None else None
    if extra_mask is not None:
        Cb_ss = float(np.sum(ss['mu'][extra_mask] * ss['c'][extra_mask]))
    for t in range(T):
        if t < H:
            dC[t] = 100.0 * (np.sum(mu * cs[t]) / ss['C'] - 1.0)
        if extra_mask is not None and t < Hb:
            dCb[t] = 100.0 * (np.sum(mu[extra_mask] * cs[t][extra_mask])
                              / Cb_ss - 1.0)
        if t < T - 1:
            mu, _ = forward_step(mu, gs[t], Pis[t])
    return dC if extra_mask is None else np.concatenate([dC, dCb])


# sequence-space Jacobian by direct anticipated perturbations
def build_jacobian(ss, masks, eps_r=1e-5, eps_d=1e-4, extra_mask=None, Hb=12):
    K = masks.shape[0]
    n = T * (1 + K)
    Hm = H if extra_mask is None else H + Hb
    J = np.empty((Hm, n))
    zero_d = np.zeros((K, T))
    col = 0
    for s in range(T): # aggregate block
        dr = np.zeros(T); dr[s] = eps_r
        J[:, col] = solve_path(ss, dr, zero_d, masks,
                               extra_mask=extra_mask, Hb=Hb) / eps_r
        col += 1
    for k in range(K): # regional tilts
        for s in range(T):
            dd = np.zeros((K, T)); dd[k, s] = eps_d
            J[:, col] = solve_path(ss, np.zeros(T), dd, masks,
                                   extra_mask=extra_mask, Hb=Hb) / eps_d
            col += 1
    return J
