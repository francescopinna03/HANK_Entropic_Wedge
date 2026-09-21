# Entropic machinery on top of the sequence-space Jacobian.
# Blocks are ordered [agg | A_1 | ... | A_K]; all priors are diagonal. Omega_agg^{-1} = sigma_r^{-2} I_T,   Omega_k^{-1} = qbar_k I_T  (eq. I.10).

import numpy as np
from scipy import integrate, linalg
from scipy.stats import norm

from model import T, H


# blocks and projection (eqs. 33-34 of the note)

def block_slices(K):
    return [slice(0, T)] + [slice(T * (1 + k), T * (2 + k)) for k in range(K)]


def omega_inv_diag(qb, sigma_r, n_eff=1.0):
    """Diagonal of Omega^{-1}, length n = T*(1+K).
    The pinned idiosyncratic prior of eq. I.10 is per capita (Lemma 9.1). Notice that
    the population KL of a deterministic tilt is N times the individual one.
    Cross-layer comparability therefore requires the effective
    number of independent idiosyncratic histories disciplined by the
    data, n_eff:  Omega_k^{-1} = n_eff * qbar_k * I_T."""
    K = len(qb)
    d = np.empty(T * (1 + K))
    d[:T] = sigma_r ** -2
    for k in range(K):
        d[T * (1 + k):T * (2 + k)] = n_eff * qb[k]
    return d


def projection_matrix(J, oinv, rho):    #M = (Omega^{-1} + rho J'J)^{-1} rho J'  (R = rho I_H).
    n = J.shape[1]
    A = np.diag(oinv) + rho * (J.T @ J)
    return np.linalg.solve(A, rho * J.T)             # (n, H)


def quad_forms(M, oinv, K): # Q_k = M' S_k' Omega_k^{-1} S_k M for each block (agg first), Q_tot.

    Qs = []
    for sl in block_slices(K):
        Mb = M[sl, :]
        Qs.append(Mb.T @ (oinv[sl, None] * Mb))
    Q_tot = sum(Qs)
    return Qs, Q_tot


def shares_and_info(r, Qs, Q_tot):
    I_k = np.array([0.5 * r @ Q @ r for Q in Qs])
    I_tot = 0.5 * r @ Q_tot @ r
    return I_k / I_tot, I_k, I_tot


def dep(I_tot):
    return float(norm.cdf(-np.sqrt(I_tot / 2.0)))


# Imhof (1961): tail at zero of a weighted chi^2 (eq. I.13)

def imhof_tail_at_zero(lams, tol=1e-12): # Pr( sum_i lam_i Z_i^2 >= 0 ), Z_i iid N(0,1)
    lams = lams[np.abs(lams) > tol * np.max(np.abs(lams))]
    if lams.size == 0:
        return 1.0
    if np.all(lams >= 0):
        return 1.0
    if np.all(lams <= 0):
        return 0.0

    def theta(u):
        return 0.5 * np.sum(np.arctan(np.outer(u, lams)), axis=1)

    def rho(u):
        return np.exp(0.25 * np.sum(np.log1p(np.outer(u, lams) ** 2), axis=1))

    def integrand(u):
        u = np.atleast_1d(u)
        return np.sin(theta(u)) / (u * rho(u))

    val, _ = integrate.quad(lambda u: integrand(u).item(), 0.0, np.inf,
                            limit=400)
    return float(np.clip(0.5 + val / np.pi, 0.0, 1.0))


def p_value(s_obs, Q_k, Q_tot):
    lams = np.linalg.eigvalsh(Q_k - s_obs * Q_tot)
    return imhof_tail_at_zero(lams)


def mc_tail(s_obs, Q_k, Q_tot, n_draws=400_000, seed=0):
    rng = np.random.default_rng(seed)
    xi = rng.standard_normal((n_draws, Q_tot.shape[0]))
    s = np.einsum('ij,jk,ik->i', xi, Q_k, xi) / \
        np.einsum('ij,jk,ik->i', xi, Q_tot, xi)
    return float(np.mean(s >= s_obs)), s


# capacity and confounding direction (Def. 10.1)
def capacity(Q_k, Q_tot):
    w, V = linalg.eigh(Q_k, Q_tot)
    return float(w[0]), float(w[-1]), V[:, -1]


def cos_angle_qtot(r, v, Q_tot):
    num = abs(r @ Q_tot @ v)
    den = np.sqrt((r @ Q_tot @ r) * (v @ Q_tot @ v))
    return float(num / den)


# pre-data confusion matrix and resolution (Def. 10.2)
def confusion(residuals, Qs, Q_tot):
    L, B = len(residuals), len(Qs)
    S = np.empty((L, B)); P = np.empty((L, B))
    for l, r in enumerate(residuals):
        for b in range(B):
            S[l, b] = (r @ Qs[b] @ r) / (r @ Q_tot @ r)
            P[l, b] = p_value(S[l, b], Qs[b], Q_tot)
    return S, P


def resolving(P, alpha=0.05): #resolution at level alpha iff argmin_b P[l,:] = l and P[l,l] <= alpha for every injection l (injection l targets block l by construction).
    ok = True
    for l in range(P.shape[0]):
        if np.argmin(P[l]) != l or P[l, l] > alpha:
            ok = False
    return ok


# pairwise attribution (Oss. 10.6). contrasts s_k - s_j under the null
def pairwise_pvals(r, Qs, Q_tot):
    """pi[k,j] = Pr_null( s_k(xi)-s_j(xi) >= s_k(r)-s_j(r) ), Imhof at zero
    of (Q_k - Q_j) - d_obs * Q_tot.  Diagonal set to 0."""
    B = len(Qs)
    s = np.array([(r @ Q @ r) / (r @ Q_tot @ r) for Q in Qs])
    pi = np.zeros((B, B))
    for k in range(B):
        for j in range(B):
            if j == k:
                continue
            d_obs = s[k] - s[j]
            lams = np.linalg.eigvalsh((Qs[k] - Qs[j]) - d_obs * Q_tot)
            pi[k, j] = imhof_tail_at_zero(lams)
    return pi, s


def attribute(r, Qs, Q_tot): #Attribution score per block. worst pairwise p-value; winner is the block whose dominance over every rival is most uniformly anomalous
    pi, s = pairwise_pvals(r, Qs, Q_tot)
    B = len(Qs)
    score = np.array([max(pi[k, j] for j in range(B) if j != k)
                      for k in range(B)])
    return int(np.argmin(score)), score, s
