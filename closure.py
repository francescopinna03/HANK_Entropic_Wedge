import numpy as np
from model import *


KAPPA_Y = 0.05
PHI_PI = 1.5


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


def hank_outputs(ss, dr, dx, dbeta, delta, masks, extra_mask=None, Hb=12, drift=None):
    return solve_path(ss, dr, delta, masks, extra_mask=extra_mask, Hb=Hb,
                      dbeta_path=dbeta, dx_path=dx, return_consumption=True, drift_paths=drift)


def consumption_jacobians(ss, masks, extra_mask=None, Hb=12, eps_r=1e-5, eps_b=1e-5, eps_d=1e-4, eps_x=1e-5, with_drift=False):
    K = masks.shape[0]
    zr = np.zeros(T)
    zd = np.zeros((K, T))
    y0, A0, C0 = hank_outputs(ss, zr, zr, zr, zd, masks, extra_mask, Hb)
    Hm = y0.size
    nd = (2 if with_drift else 1) * K * T
    Jy = dict(r=np.empty((Hm, T)), b=np.empty((Hm, T)), d=np.empty((Hm, nd)), x=np.empty((Hm, T)))
    JC = dict(r=np.empty((T, T)), b=np.empty((T, T)), d=np.empty((T, nd)), x=np.empty((T, T)))
    for s in range(T):
        dr = np.zeros(T); dr[s] = eps_r
        y, A, C = hank_outputs(ss, dr, zr, zr, zd, masks, extra_mask, Hb)
        Jy['r'][:, s] = (y - y0) / eps_r
        JC['r'][:, s] = (C - C0) / eps_r
        dx = np.zeros(T); dx[s] = eps_x
        y, A, C = hank_outputs(ss, zr, dx, zr, zd, masks, extra_mask, Hb)
        Jy['x'][:, s] = (y - y0) / eps_x
        JC['x'][:, s] = (C - C0) / eps_x
        db = np.zeros(T); db[s] = eps_b
        y, A, C = hank_outputs(ss, zr, zr, db, zd, masks, extra_mask, Hb)
        Jy['b'][:, s] = (y - y0) / eps_b
        JC['b'][:, s] = (C - C0) / eps_b
    for k in range(K):
        for s in range(T):
            dd = np.zeros((K, T)); dd[k, s] = eps_d
            y, A, C = hank_outputs(ss, zr, zr, zr, dd, masks, extra_mask, Hb)
            Jy['d'][:, k * T + s] = (y - y0) / eps_d
            JC['d'][:, k * T + s] = (C - C0) / eps_d
            if with_drift:
                dm = np.zeros((K, T)); dm[k, s] = eps_d
                y, A, C = hank_outputs(ss, zr, zr, zr, zd, masks, extra_mask, Hb, dm)
                Jy['d'][:, (K + k) * T + s] = (y - y0) / eps_d
                JC['d'][:, (K + k) * T + s] = (C - C0) / eps_d
    return Jy, JC, y0, C0


def aggregate_consumption(Jy, JC, Kc, K8=8):
    g = K8 // Kc
    nb = Jy['d'].shape[1] // (K8 * T)
    Jyd = np.empty((Jy['d'].shape[0], nb * Kc * T))
    JCd = np.empty((T, nb * Kc * T))
    for b in range(nb):
        for k in range(Kc):
            cols = [(b * K8 + j) * T for j in range(g*k, g*(k+1))]
            Jyd[:, (b*Kc+k)*T:(b*Kc+k+1)*T] = sum(Jy['d'][:, c:c+T] for c in cols)
            JCd[:, (b*Kc+k)*T:(b*Kc+k+1)*T] = sum(JC['d'][:, c:c+T] for c in cols)
    return dict(r=Jy['r'], b=Jy['b'], d=Jyd, x=Jy['x']), dict(r=JC['r'], b=JC['b'], d=JCd, x=JC['x'])


def hank_operators(ss, JC, kappa_y=KAPPA_Y, phi_pi=PHI_PI):
    F = np.eye(T, k=1)
    Y_ss = ss['C']
    B = ss['A']
    P = np.linalg.solve(np.eye(T) - BETA * F, kappa_y * np.eye(T)) / Y_ss
    R = (phi_pi * np.eye(T) - F) @ P
    G = JC['r'] - B * JC['x']
    HU = np.eye(T) - JC['x'] - G @ R
    return dict(P=P, R=R, G=G, HU=HU, F=F, Y_ss=Y_ss, B=B, kappa_y=kappa_y, phi_pi=phi_pi)


def hank_jacobian(ss, Jy, JC, with_micro, kappa_y=KAPPA_Y, phi_pi=PHI_PI):
    op = hank_operators(ss, JC, kappa_y, phi_pi)
    Y_ss, B, R, G = op['Y_ss'], op['B'], op['R'], op['G']
    HUinv = np.linalg.inv(op['HU'])
    dY = dict(e=HUinv @ G, d=HUinv @ JC['d'], b=HUinv @ JC['b'])
    dr = dict(e=R @ dY['e'] + np.eye(T), d=R @ dY['d'], b=R @ dY['b'])
    dx = {k: dY[k] - B * dr[k] for k in dY}
    direct = dict(e=0.0, d=Jy['d'][H:])
    rows = {}
    for k in ['e', 'd']:
        rows[k] = 100.0 * dY[k][:H] / Y_ss
        if with_micro:
            rows[k] = np.vstack([rows[k], direct[k] + Jy['r'][H:] @ dr[k] + Jy['x'][H:] @ dx[k]])
    return np.hstack([rows['e'], rows['d']]), op, dict(dY=dY, dr=dr, dx=dx)


def hank_transition(ss, eps, delta, masks, JC, extra_mask=None, Hb=12, dbeta=None, kappa_y=KAPPA_Y, phi_pi=PHI_PI, tol=1e-10, maxit=80, drift=None):
    op = hank_operators(ss, JC, kappa_y, phi_pi)
    Y_ss, B, F, G = op['Y_ss'], op['B'], op['F'], op['G']
    db = np.zeros(T) if dbeta is None else dbeta

    def paths(Y, pi):
        r = R_SS + phi_pi * pi + eps - F @ pi
        X = Y - r * B
        return r, X

    def residual(Y, pi):
        r, X = paths(Y, pi)
        y, A, C = hank_outputs(ss, r - R_SS, X - 1.0, db, delta, masks, extra_mask, Hb, drift)
        gY = Y - C
        gpi = pi - kappa_y * np.log(Y / Y_ss) - BETA * (F @ pi)
        return np.concatenate([gY, gpi]), y, r, C

    Y = np.full(T, Y_ss)
    pi = np.zeros(T)
    f, y, r, C = residual(Y, pi)
    dr_dpi = phi_pi * np.eye(T) - F
    JY = np.block([[np.eye(T) - JC['x'], -G @ dr_dpi],
                   [-kappa_y / Y_ss * np.eye(T), np.eye(T) - BETA * F]])
    Binv = np.linalg.inv(JY)
    z = np.concatenate([Y, pi])
    it = 0
    while np.max(np.abs(f)) > tol and it < maxit:
        step = -Binv @ f
        z = z + step
        Y, pi = z[:T], z[T:]
        f_new, y, r, C = residual(Y, pi)
        df = f_new - f
        Binv = Binv + np.outer(step - Binv @ df, step @ Binv) / (step @ Binv @ df)
        f = f_new
        it += 1
    m = y[:H]
    if extra_mask is not None:
        m = np.concatenate([m, y[H:]])
    return m, r - R_SS, pi, Y, it, float(np.max(np.abs(f)))


def determinacy_monitor(op):
    s = np.linalg.svd(op['HU'], compute_uv=False)
    return dict(phi_pi=float(op['phi_pi']), kappa_y=float(op['kappa_y']),
                taylor_principle=bool(op['phi_pi'] > 1.0),
                sigma_min_HU=float(s[-1]), sigma_max_HU=float(s[0]), cond_HU=float(s[0] / s[-1]))


def conditioned_hank_jacobian(ss, Jy, JC, with_micro, kappa_y=KAPPA_Y, phi_pi=PHI_PI):
    Jt, op, maps = hank_jacobian(ss, Jy, JC, with_micro, kappa_y, phi_pi)
    Gr = np.vstack([Jy['r'][:H], Jy['r'][H:]]) if with_micro else Jy['r'][:H]
    cols_e = Jt[:, :T] - Gr @ maps['dr']['e']
    cols_d = Jt[:, T:] - Gr @ maps['dr']['d']
    return np.hstack([cols_e, cols_d]), op, maps
