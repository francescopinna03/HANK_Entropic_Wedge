import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from model import *


def frozen_setup(ss, extra_mask, Hb=12):
    idx, w_hi = young_weights(ss['g'])
    hagg = 100.0 * ss['c'] / ss['C']
    Cb = float(np.sum(ss['mu'][extra_mask] * ss['c'][extra_mask]))
    hb = 100.0 * ss['c'] * extra_mask[:, None] / Cb
    gbar = np.concatenate([np.full(H, 100.0), np.full(Hb, 100.0)])
    return dict(idx=idx, w_hi=w_hi, hagg=hagg, hb=hb, gbar=gbar, Hb=Hb, mu0=ss['mu'], mu_end=ss['mu_end'])


def lottery_forward(mu, st):
    mu_end = np.zeros((NA, NZ))
    for z in range(NZ):
        np.add.at(mu_end[:, z], st['idx'][:, z], mu[:, z] * (1.0 - st['w_hi'][:, z]))
        np.add.at(mu_end[:, z], st['idx'][:, z] + 1, mu[:, z] * st['w_hi'][:, z])
    return mu_end


def lottery_backward(Vend, st, tilt=False):
    V = np.empty((NA, NZ))
    for z in range(NZ):
        lo = Vend[st['idx'][:, z], z]
        hi = Vend[st['idx'][:, z] + 1, z]
        w = st['w_hi'][:, z]
        if tilt:
            V[:, z] = logsumexp(np.stack([lo, hi]), axis=0, b=np.stack([1.0 - w, w]))
        else:
            V[:, z] = (1.0 - w) * lo + w * hi
    return V


def period_functional(zeta, st, t):
    Hb = st['Hb']
    out = np.zeros((NA, NZ))
    if t < H:
        out += zeta[t] * st['hagg']
    if t < Hb:
        out += zeta[H + t] * st['hb']
    return out


def dual_value_and_kernels(zeta, st, tilt_lottery=False):
    U = np.zeros((NA, NZ))
    Us = [None] * (T + 1)
    Js = [None] * T
    Us[T] = U
    for t in range(T - 1, -1, -1):
        J = np.empty((NA, NZ))
        for z in range(NZ):
            J[:, z] = logsumexp(Us[t + 1], axis=1, b=PI0[z][None, :])
        Js[t] = J
        U = period_functional(zeta, st, t) + lottery_backward(J, st, tilt_lottery)
        Us[t] = U
    Lam = float(np.sum(st['mu0'] * Us[0])) - float(zeta @ st['gbar'])
    return Lam, Us, Js


def forward_under_tilt(zeta, st, Us, Js, tilt_lottery=False):
    Hb = st['Hb']
    mu = st['mu0'].copy()
    g = np.zeros(H + Hb)
    kl = 0.0
    for t in range(T):
        if t < H:
            g[t] = np.sum(mu * st['hagg'])
        if t < Hb:
            g[H + t] = np.sum(mu * st['hb'])
        if tilt_lottery:
            mu_end = np.zeros((NA, NZ))
            for z in range(NZ):
                lo = Js[t][st['idx'][:, z], z]
                hi = Js[t][st['idx'][:, z] + 1, z]
                w = st['w_hi'][:, z]
                m = logsumexp(np.stack([lo, hi]), axis=0, b=np.stack([1.0 - w, w]))
                qlo = (1.0 - w) * np.exp(lo - m)
                qhi = w * np.exp(hi - m)
                kl += np.sum(mu[:, z] * (qlo * np.log(np.maximum(qlo, 1e-300) / np.maximum(1.0 - w, 1e-300)) + qhi * np.log(np.maximum(qhi, 1e-300) / np.maximum(w, 1e-300))))
                np.add.at(mu_end[:, z], st['idx'][:, z], mu[:, z] * qlo)
                np.add.at(mu_end[:, z], st['idx'][:, z] + 1, mu[:, z] * qhi)
        else:
            mu_end = lottery_forward(mu, st)
        Q = np.empty((NA, NZ, NZ))
        for z in range(NZ):
            Q[:, z, :] = PI0[z][None, :] * np.exp(Us[t + 1] - Js[t][:, z][:, None])
        kl += float(np.sum(mu_end[:, :, None] * Q * np.log(Q / PI0[None, :, :])))
        mu = np.einsum('az,azx->ax', mu_end, Q)
    return g - st['gbar'], float(kl), mu


def solve_exact(delta_y, st, tilt_lottery=False, zeta0=None, tol=1e-10):
    n = delta_y.size
    def f(zeta):
        Lam, Us, Js = dual_value_and_kernels(zeta, st, tilt_lottery)
        g, kl, _ = forward_under_tilt(zeta, st, Us, Js, tilt_lottery)
        return Lam - zeta @ delta_y, g - delta_y
    z0 = np.zeros(n) if zeta0 is None else zeta0
    res = minimize(f, z0, jac=True, method='L-BFGS-B', options=dict(maxiter=5000, ftol=1e-15, gtol=tol))
    zeta = res.x
    Lam, Us, Js = dual_value_and_kernels(zeta, st, tilt_lottery)
    g, kl, _ = forward_under_tilt(zeta, st, Us, Js, tilt_lottery)
    W = float(zeta @ delta_y - Lam)
    return dict(W=W, KL=kl, zeta=zeta, feasibility=float(np.max(np.abs(g - delta_y))), nit=int(res.nit), Us=Us, Js=Js)


def exact_kernel_factors(sol, st):
    out = np.empty((T, NA, NZ))
    for t in range(T):
        for z in range(NZ):
            leave_q = 0.0
            for zp in range(NZ):
                if zp != z:
                    leave_q = leave_q + PI0[z, zp] * np.exp(sol['Us'][t + 1][:, zp] - sol['Js'][t][:, z])
            out[t, :, z] = leave_q / (1.0 - PI0[z, z]) - 1.0
    return out


def forward_with_delta(delta_state, masks, st):
    K = masks.shape[0]
    Hb = st['Hb']
    mu = st['mu0'].copy()
    g = np.zeros(H + Hb)
    kl = 0.0
    for t in range(T):
        if t < H:
            g[t] = np.sum(mu * st['hagg'])
        if t < Hb:
            g[H + t] = np.sum(mu * st['hb'])
        dgrid = np.zeros((NA, NZ))
        for k in range(K):
            dgrid[masks[k], :] += delta_state[k, :, t][None, :]
        Pi = tilted_Pi(dgrid)
        mu_end = lottery_forward(mu, st)
        kl += float(np.sum(mu_end[:, :, None] * Pi * np.log(Pi / PI0[None, :, :])))
        mu = np.einsum('az,azx->ax', mu_end, Pi)
    return g - st['gbar'], float(kl)


def frozen_jacobian(masks, st, eps=1e-4):
    K = masks.shape[0]
    Hm = H + st['Hb']
    J = np.empty((Hm, K * NZ * T))
    g0, _ = forward_with_delta(np.zeros((K, NZ, T)), masks, st)
    for k in range(K):
        for z in range(NZ):
            for s in range(T):
                d = np.zeros((K, NZ, T)); d[k, z, s] = eps
                g, _ = forward_with_delta(d, masks, st)
                J[:, (k * NZ + z) * T + s] = (g - g0) / eps
    return J


def second_order_weights(masks, st):
    K = masks.shape[0]
    leave = 1.0 - np.diag(PI0)
    w = masks @ (st['mu_end'] * (leave / (1.0 - leave))[None, :])
    return w


def quadratic_restricted(delta_y, J, w_blocks, blocks, ridge=1e-12):
    d = np.concatenate([np.full(T, w_blocks[k, z]) for (k, z) in blocks])
    cols = np.hstack([J[:, (k * NZ + z) * T:(k * NZ + z + 1) * T] for (k, z) in blocks])
    G = cols @ (cols / d[None, :]).T
    u = (cols / d[None, :]).T @ np.linalg.solve(G + ridge * np.eye(G.shape[0]), delta_y)
    return float(0.5 * u @ (d * u)), u


def exact_restricted(delta_y, masks, st, blocks, u0=None, maxiter=30, tol=1e-9, eps=1e-5):
    K = masks.shape[0]
    nb = len(blocks)
    w = second_order_weights(masks, st)
    D = np.concatenate([np.full(T, w[k, z]) for (k, z) in blocks])
    active = np.ones(delta_y.size, dtype=bool)
    active[0] = False
    active[H] = False
    def unpack(x):
        d = np.zeros((K, NZ, T))
        for i, (k, z) in enumerate(blocks):
            d[k, z] = x[i * T:(i + 1) * T]
        return d
    x = np.zeros(nb * T) if u0 is None else u0.copy()
    hist = []
    for it in range(maxiter):
        g0, kl0 = forward_with_delta(unpack(x), masks, st)
        r = (delta_y - g0)[active]
        J = np.empty((active.sum(), nb * T))
        gk = np.empty(nb * T)
        for j in range(nb * T):
            xp = x.copy(); xp[j] += eps
            g, kl = forward_with_delta(unpack(xp), masks, st)
            J[:, j] = (g - g0)[active] / eps
            gk[j] = (kl - kl0) / eps
        JD = J / D[None, :]
        lam = np.linalg.solve(JD @ J.T, r + JD @ gk)
        step = (J.T @ lam - gk) / D
        x = x + step
        hist.append((kl0, float(np.abs(r).max()), float(np.abs(step).max())))
        if np.abs(r).max() < tol and np.abs(step).max() < 1e-7:
            break
    g, kl = forward_with_delta(unpack(x), masks, st)
    return dict(W=kl, delta=unpack(x), u=x, feasibility=float(np.max(np.abs((g - delta_y)[active]))), nit=it + 1, hist=hist)


def kl_by_block(delta_state, masks, st):
    K = masks.shape[0]
    mu = st['mu0'].copy()
    out = np.zeros((K, NZ))
    for t in range(T):
        dgrid = np.zeros((NA, NZ))
        for k in range(K):
            dgrid[masks[k], :] += delta_state[k, :, t][None, :]
        Pi = tilted_Pi(dgrid)
        mu_end = lottery_forward(mu, st)
        rowkl = np.sum(Pi * np.log(Pi / PI0[None, :, :]), axis=2)
        for k in range(K):
            out[k] += np.sum(mu_end[masks[k], :] * rowkl[masks[k], :], axis=0)
        mu = np.einsum('az,azx->ax', mu_end, Pi)
    return out


def channel_share(delta_state, k):
    dL, dH = delta_state[k, 0], delta_state[k, 1]
    p = 0.5 * (dL + dH)
    m = 0.5 * (dL - dH)
    return float(m @ m / (m @ m + p @ p))
