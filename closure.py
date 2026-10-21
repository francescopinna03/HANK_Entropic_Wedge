import numpy as np
from model import *


def ge_outputs(ss, dr, dbeta, delta, masks, extra_mask=None, Hb=12):
    return solve_path(ss, dr, delta, masks, extra_mask=extra_mask, Hb=Hb,
                      dbeta_path=dbeta, return_assets=True)


def pe_jacobians(ss, masks, extra_mask=None, Hb=12, eps_r=1e-5, eps_b=1e-5, eps_d=1e-4):
    K = masks.shape[0]
    zr = np.zeros(T)
    zd = np.zeros((K, T))
    y0, A0 = ge_outputs(ss, zr, zr, zd, masks, extra_mask, Hb)
    Hm = y0.size
    Jy = dict(r=np.empty((Hm, T)), b=np.empty((Hm, T)), d=np.empty((Hm, K * T)))
    JA = dict(r=np.empty((T, T)), b=np.empty((T, T)), d=np.empty((T, K * T)))
    for s in range(T):
        dr = np.zeros(T); dr[s] = eps_r
        y, A = ge_outputs(ss, dr, zr, zd, masks, extra_mask, Hb)
        Jy['r'][:, s] = (y - y0) / eps_r
        JA['r'][:, s] = (A - A0) / eps_r
        db = np.zeros(T); db[s] = eps_b
        y, A = ge_outputs(ss, zr, db, zd, masks, extra_mask, Hb)
        Jy['b'][:, s] = (y - y0) / eps_b
        JA['b'][:, s] = (A - A0) / eps_b
    for k in range(K):
        for s in range(T):
            dd = np.zeros((K, T)); dd[k, s] = eps_d
            y, A = ge_outputs(ss, zr, zr, dd, masks, extra_mask, Hb)
            Jy['d'][:, k * T + s] = (y - y0) / eps_d
            JA['d'][:, k * T + s] = (A - A0) / eps_d
    return Jy, JA, y0, A0


def aggregate_pe(Jy, JA, Kc, K8=8):
    g = K8 // Kc
    Jyd = np.empty((Jy['d'].shape[0], Kc * T))
    JAd = np.empty((T, Kc * T))
    for k in range(Kc):
        Jyd[:, k*T:(k+1)*T] = sum(Jy['d'][:, j*T:(j+1)*T] for j in range(g*k, g*(k+1)))
        JAd[:, k*T:(k+1)*T] = sum(JA['d'][:, j*T:(j+1)*T] for j in range(g*k, g*(k+1)))
    return dict(r=Jy['r'], b=Jy['b'], d=Jyd), dict(r=JA['r'], b=JA['b'], d=JAd)


def ge_jacobian(Jy, JA, with_micro):
    dr_db = -np.linalg.solve(JA['r'], JA['b'])
    dr_dd = -np.linalg.solve(JA['r'], JA['d'])
    rows_b = 100.0 * dr_db[:H]
    rows_d = 100.0 * dr_dd[:H]
    if with_micro:
        rows_b = np.vstack([rows_b, Jy['b'][H:] + Jy['r'][H:] @ dr_db])
        rows_d = np.vstack([rows_d, Jy['d'][H:] + Jy['r'][H:] @ dr_dd])
    return np.hstack([rows_b, rows_d]), dr_db, dr_dd


def ge_transition(ss, dbeta, delta, masks, JA_r, extra_mask=None, Hb=12, tol=1e-10, maxit=60):
    zr = np.zeros(T)
    y, A = ge_outputs(ss, zr, dbeta, delta, masks, extra_mask, Hb)
    f = A - ss['A']
    Binv = np.linalg.inv(JA_r)
    dr = np.zeros(T)
    it = 0
    while np.max(np.abs(f)) > tol and it < maxit:
        step = -Binv @ f
        dr = dr + step
        y, A = ge_outputs(ss, dr, dbeta, delta, masks, extra_mask, Hb)
        f_new = A - ss['A']
        df = f_new - f
        Binv = Binv + np.outer(step - Binv @ df, step @ Binv) / (step @ Binv @ df)
        f = f_new
        it += 1
    m = 100.0 * dr[:H]
    if extra_mask is not None:
        m = np.concatenate([m, y[H:]])
    return m, dr, y, it, float(np.max(np.abs(f)))


def monitors(JA_r, Jt):
    sA = np.linalg.svd(JA_r, compute_uv=False)
    sJ = np.linalg.svd(Jt, compute_uv=False)
    return dict(sigma_min_market=float(sA[-1]), sigma_max_market=float(sA[0]),
                cond_market=float(sA[0] / sA[-1]), sigma_min_Jt=float(sJ[-1]),
                sigma_max_Jt=float(sJ[0]), cond_Jt=float(sJ[0] / sJ[-1]))


def cosine(u, v):
    return float(u @ v / np.sqrt((u @ u) * (v @ v)))
