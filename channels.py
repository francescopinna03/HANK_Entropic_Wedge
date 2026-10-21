import numpy as np
from model import *
from closure import ge_outputs, ge_jacobian


def qbar_states(mu_end, masks):
    leave = 1.0 - np.diag(PI0)
    return masks @ (mu_end * leave[None, :])


def pe_jacobians_states(ss, masks, extra_mask=None, Hb=12, eps_r=1e-5, eps_b=1e-5, eps_d=1e-4):
    K = masks.shape[0]
    zr = np.zeros(T)
    zd = np.zeros((K, NZ, T))
    y0, A0 = ge_outputs(ss, zr, zr, zd, masks, extra_mask, Hb)
    Hm = y0.size
    Jy = dict(r=np.empty((Hm, T)), b=np.empty((Hm, T)), d=np.empty((Hm, K * NZ * T)))
    JA = dict(r=np.empty((T, T)), b=np.empty((T, T)), d=np.empty((T, K * NZ * T)))
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
        for z in range(NZ):
            for s in range(T):
                dd = np.zeros((K, NZ, T)); dd[k, z, s] = eps_d
                y, A = ge_outputs(ss, zr, zr, dd, masks, extra_mask, Hb)
                c = (k * NZ + z) * T + s
                Jy['d'][:, c] = (y - y0) / eps_d
                JA['d'][:, c] = (A - A0) / eps_d
    return Jy, JA, y0, A0


def aggregate_states(Jy, JA, Kc, K8=8):
    g = K8 // Kc
    def agg(M):
        out = np.zeros((M.shape[0], Kc * NZ * T))
        for k in range(Kc):
            for z in range(NZ):
                for j in range(g * k, g * (k + 1)):
                    out[:, (k*NZ+z)*T:(k*NZ+z+1)*T] += M[:, (j*NZ+z)*T:(j*NZ+z+1)*T]
        return out
    return dict(r=Jy['r'], b=Jy['b'], d=agg(Jy['d'])), dict(r=JA['r'], b=JA['b'], d=agg(JA['d']))


def kept_blocks(qb_states, tol=1e-12):
    return [(k, z) for k in range(qb_states.shape[0]) for z in range(NZ) if qb_states[k, z] > tol]


def select_columns(Jd, blocks):
    return np.hstack([Jd[:, (k*NZ+z)*T:(k*NZ+z+1)*T] for (k, z) in blocks])


def pe_matrix(JyK, blocks):
    return np.hstack([JyK['r'], select_columns(JyK['d'], blocks)])


def ge_matrix(JyK, JAK, blocks, with_micro):
    Jt, _, _ = ge_jacobian(JyK, JAK, with_micro)
    return np.hstack([Jt[:, :T], select_columns(Jt[:, T:], blocks)])


def region_groups(blocks, Kc):
    return [[1 + i for i, (k, z) in enumerate(blocks) if k == kk] for kk in range(Kc)]


def channel_split(u, blocks, Kc):
    out = []
    for kk in range(Kc):
        idx = {z: i for i, (k, z) in enumerate(blocks) if k == kk}
        if len(idx) < NZ:
            out.append(None)
            continue
        dL = u[T*(1+idx[0]):T*(2+idx[0])]
        dH = u[T*(1+idx[1]):T*(2+idx[1])]
        p = 0.5 * (dL + dH)
        m = 0.5 * (dL - dH)
        out.append(float(m @ m / (m @ m + p @ p)))
    return out
