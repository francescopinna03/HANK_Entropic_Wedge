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
import os


def rouwenhorst(n, rho, sigma):
    p = (1.0 + rho) / 2.0
    Pi = np.array([[p, 1.0 - p], [1.0 - p, p]])
    for m in range(3, n + 1):
        Z = np.zeros((m, m))
        Z[:-1, :-1] += p * Pi
        Z[:-1, 1:] += (1.0 - p) * Pi
        Z[1:, :-1] += (1.0 - p) * Pi
        Z[1:, 1:] += p * Pi
        Z[1:-1] /= 2.0
        Pi = Z
    psi = sigma * np.sqrt(n - 1)
    z = np.linspace(-psi, psi, n)
    w = np.ones(n) / n
    for _ in range(10000):
        w_new = w @ Pi
        if np.max(np.abs(w_new - w)) < 1e-15:
            break
        w = w_new
    y = np.exp(z) / (w @ np.exp(z))
    return y, Pi


CALIBRATIONS = dict(
    lab=dict(beta=0.97, r=0.01, a_max=60.0, income=dict(kind='two_state', y=[0.70, 1.30], p_switch=0.075)),
    liquid=dict(beta=None, beta_solved=0.954320, r=0.01, a_max=40.0, wealth_target=1.04, income=dict(kind='rouwenhorst', n=7, rho=0.966, sigma=0.5)),
    total=dict(beta=None, beta_solved=0.975019, r=0.01, a_max=200.0, wealth_target=5.6, income=dict(kind='rouwenhorst', n=7, rho=0.966, sigma=0.5)),
)
CALIBRATION = os.environ.get('HANK_CALIBRATION', 'lab')
_cal = CALIBRATIONS[CALIBRATION]

# calibration
BETA = _cal['beta']  # quarterly discount factor
GAMMA = 2.0          # CRRA
R_SS = _cal['r']     # quarterly real rate (4% annual)
if _cal['income']['kind'] == 'two_state':
    Y = np.array(_cal['income']['y'])          # income states (mean 1)
    _p = _cal['income']['p_switch']
    PI0 = np.array([[1.0 - _p, _p],
                    [_p, 1.0 - _p]])    # baseline quarterly transition matrix
else:
    Y, PI0 = rouwenhorst(_cal['income']['n'], _cal['income']['rho'], _cal['income']['sigma'])
NZ = Y.size

NA = 120
A_MIN, A_MAX = 0.0, _cal['a_max']
AGRID = A_MIN + (A_MAX - A_MIN) * np.linspace(0.0, 1.0, NA) ** 2.5

T = 40     # truncation horizon for input paths
H = 24     # number of moment horizons (dC_t, t = 0..H-1)

U_PRIME = lambda c: c ** (-GAMMA)
U_PRIME_INV = lambda w: w ** (-1.0 / GAMMA)


# tilted transition matrices, state-dependent in end-of-period assets
def tilted_Pi(delta_on_grid, drift_on_grid=None):
    Pi = np.empty((NA, NZ, NZ))
    scale = 1.0 + np.asarray(delta_on_grid)                      # (NA,)
    off = PI0 * (1.0 - np.eye(NZ))                   # off-diagonal part
    Pi[:] = off[None, :, :] * (scale[:, :, None] if scale.ndim == 2 else scale[:, None, None])
    if drift_on_grid is not None:
        sgn = np.sign(np.arange(NZ)[None, :] - np.arange(NZ)[:, None])
        Pi *= 1.0 + np.asarray(drift_on_grid)[:, None, None] * sgn[None, :, :]
    rowsum_off = Pi.sum(axis=2)                      # (NA,NZ)
    for z in range(NZ):
        Pi[:, z, z] = 1.0 - rowsum_off[:, z]
    return Pi


# EGM backward step with time-varying r and Pi(a')
def egm_step(c_next, Pi_t, r_t, r_next, beta=None, x_t=1.0):
    if beta is None:
        beta = BETA
    mu_next = U_PRIME(c_next)  # (NA,NZ)
    # W(a',z) = beta (1+r_{t+1}) sum_{z'} Pi_t[a',z,z'] u'(c_{t+1}(a',z'))
    W = beta * (1.0 + r_next) * np.einsum('azx,ax->az', Pi_t, mu_next)
    c_endog = U_PRIME_INV(W)    # (NA,NZ) at a'
    coh_endog = c_endog + AGRID[:, None]   # cash on hand needed
    a_endog = (coh_endog - x_t * Y[None, :]) / (1.0 + r_t)
    g = np.empty((NA, NZ))
    for z in range(NZ):
        g[:, z] = np.interp(AGRID, a_endog[:, z], AGRID,
                            left=A_MIN, right=AGRID[-1]) # borrowing constraint
        g[AGRID < a_endog[0, z], z] = A_MIN
    c = (1.0 + r_t) * AGRID[:, None] + x_t * Y[None, :] - g
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
def steady_state(tol=1e-12, maxit=10000, beta=None):
    Pi_ss = tilted_Pi(np.zeros(NA))
    c = (R_SS * AGRID[:, None] + Y[None, :]) + 1e-2   # init
    for it in range(maxit):
        c_new, g = egm_step(c, Pi_ss, R_SS, R_SS, beta)
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
    for k in range(1, K - 1):
        if cuts[k] <= cuts[k - 1]:
            cuts[k] = cuts[k - 1] + 1
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
def solve_path(ss, dr_path, delta_paths, masks, extra_mask=None, Hb=12, dbeta_path=None, return_assets=False, return_kl=False, dx_path=None, return_consumption=False, drift_paths=None, expect=None): #dr_path: (T,) deviations of r_t; delta_paths: (K,T) regional tilts. Households at t=0 learn the whole path (perfect foresight)
    K = masks.shape[0]
    r_path = R_SS + np.asarray(dr_path)
    beta_path = BETA * (1.0 + (np.zeros(T) if dbeta_path is None else np.asarray(dbeta_path)))
    x_path = 1.0 + (np.zeros(T) if dx_path is None else np.asarray(dx_path))
    dp = np.asarray(delta_paths)
    if dp.ndim == 2:
        delta_grid = np.zeros((T, NA))
        for k in range(K):
            delta_grid[:, masks[k]] += dp[k][:, None]
    else:
        delta_grid = np.zeros((T, NA, NZ))
        for k in range(K):
            delta_grid[:, masks[k], :] += dp[k].T[:, None, :]
    if drift_paths is None:
        Pis = [tilted_Pi(delta_grid[t]) for t in range(T)]
    else:
        drift_grid = np.zeros((T, NA))
        for k in range(K):
            drift_grid[:, masks[k]] += np.asarray(drift_paths)[k][:, None]
        Pis = [tilted_Pi(delta_grid[t], drift_grid[t]) for t in range(T)]
    c_next = ss['c']
    cs, gs = [None] * T, [None] * T
    for t in range(T - 1, -1, -1):
        r_next = r_path[t + 1] if t + 1 < T else R_SS
        cs[t], gs[t] = egm_step(c_next, Pis[t], r_path[t], r_next, beta_path[t], x_path[t])
        c_next = cs[t]
    mu = ss['mu'].copy()
    mus = [None] * T
    A_path = np.empty(T)
    C_path = np.empty(T)
    kl = 0.0
    dC = np.empty(H)
    dCb = np.empty(Hb) if extra_mask is not None else None
    if extra_mask is not None:
        Cb_ss = float(np.sum(ss['mu'][extra_mask] * ss['c'][extra_mask]))
    for t in range(T):
        mus[t] = mu
        if t < H:
            dC[t] = 100.0 * (np.sum(mu * cs[t]) / ss['C'] - 1.0)
        if extra_mask is not None and t < Hb:
            dCb[t] = 100.0 * (np.sum(mu[extra_mask] * cs[t][extra_mask])
                              / Cb_ss - 1.0)
        A_path[t] = np.sum(mu * gs[t])
        C_path[t] = np.sum(mu * cs[t])
        if t < T - 1 or return_kl:
            mu_next, mu_end = forward_step(mu, gs[t], Pis[t])
            if return_kl:
                kl += float(np.sum(mu_end[:, :, None] * Pis[t] * np.log(Pis[t] / PI0[None, :, :])))
            if t < T - 1:
                mu = mu_next
    out = dC if extra_mask is None else np.concatenate([dC, dCb])
    if expect is not None:
        out = np.concatenate([out, expectation_moments(ss, mus, Pis, expect)])
    if return_consumption:
        return out, A_path, C_path
    if return_kl:
        return (out, A_path, kl) if return_assets else (out, kl)
    return (out, A_path) if return_assets else out


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


def calibrate_beta(target, lo=0.90, hi=0.999, tol=1e-6):
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if steady_state(beta=mid)['A'] < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


if BETA is None:
    BETA = _cal.get('beta_solved') or calibrate_beta(_cal['wealth_target'])


def income_forecast(Pis, t, h):
    P = np.repeat(np.eye(NZ)[None, :, :], NA, axis=0)
    for j in range(h):
        P = P @ Pis[min(t + j, len(Pis) - 1)]
    m1 = P @ Y
    m2 = P @ (Y ** 2)
    return m1, np.sqrt(np.maximum(m2 - m1 ** 2, 0.0))


def expectation_moments(ss, mus, Pis, expect):
    groups, He, h = expect['groups'], expect['He'], expect['h']
    Pi_ss = tilted_Pi(np.zeros(NA))
    m1_ss, sd_ss = income_forecast([Pi_ss], 0, h)
    out = []
    for g in groups:
        E_ss = np.sum(ss['mu'][g] * m1_ss[g]) / np.sum(ss['mu'][g])
        S_ss = np.sum(ss['mu'][g] * sd_ss[g]) / np.sum(ss['mu'][g])
        dE, dS = np.empty(He), np.empty(He)
        for t in range(He):
            m1, sd = income_forecast(Pis, t, h)
            w = mus[t][g] / np.sum(mus[t][g])
            dE[t] = 100.0 * (np.sum(w * m1[g]) / E_ss - 1.0)
            dS[t] = 100.0 * (np.sum(w * sd[g]) / S_ss - 1.0)
        out.append(dE); out.append(dS)
    return np.concatenate(out)


def income_groups(ss, n=3):
    w = ss['mu'].sum(0)
    cw = np.cumsum(w) - w
    if NZ <= n:
        parts = [[z] for z in range(NZ)]
    else:
        parts = [[z for z in range(NZ) if j / n <= cw[z] < (j + 1) / n] for j in range(n)]
    groups = []
    for zs in parts:
        g = np.zeros((NA, NZ), bool)
        g[:, zs] = True
        groups.append(g)
    return groups
