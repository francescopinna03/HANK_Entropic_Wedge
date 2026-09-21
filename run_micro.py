#Resolution analysis with AUGMENTED moments; 24 aggregate dC_t plus 12 distributional dCb_t (bottom-50% by beginning-of-period wealth)

import numpy as np, json, warnings, time, os
warnings.filterwarnings('ignore')
from model import *
from entropic import *

NE = 8000
SIGMA_R = 0.0025
ALPHA = 0.05
DEC = 0.85
AMP = 0.15
HB = 12
DR_INJ = -0.005 * 0.8 ** np.arange(T)

t_all = time.time()
ss = steady_state()
results = {}

mu_a = ss['mu'].sum(1)  # beginning-of-period wealth marginal
cum = np.cumsum(mu_a)
em = cum <= 0.5 + 1e-12  # bottom-50% by wealth
results['bottom_mass'] = float(mu_a[em].sum())
results['bottom_cells'] = int(em.sum())
print('bottom-50%% mask: %d grid points, mass %.4f'
      % (em.sum(), mu_a[em].sum()))

masks8, masses8 = make_regions(ss['mu_end'], 8)
qb8 = qbar(ss['mu_end'], masks8)
t0 = time.time()
J8m = build_jacobian(ss, masks8, extra_mask=em, Hb=HB)   # (H+HB, T*9)
print('augmented J8 built in %.1fs, shape %s' % (time.time() - t0, J8m.shape))
np.save('J8m.npy', J8m)
Hm = H + HB

def level(Kc):
    g = 8 // Kc
    masksK = np.array([masks8[g*k:g*(k+1)].any(0) for k in range(Kc)])
    qbK = np.array([qb8[g*k:g*(k+1)].sum() for k in range(Kc)])
    JK = np.empty((Hm, T * (1 + Kc)))
    JK[:, :T] = J8m[:, :T]
    for k in range(Kc):
        cols = sum(J8m[:, T*(1+j):T*(2+j)] for j in range(g*k, g*(k+1)))
        JK[:, T*(1+k):T*(2+k)] = cols
    return masksK, qbK, JK

def injections(masksK):
    Kc = masksK.shape[0]
    rs = [solve_path(ss, DR_INJ, np.zeros((Kc, T)), masksK,
                     extra_mask=em, Hb=HB)]
    for k in range(Kc):
        dd = np.zeros((Kc, T)); dd[k] = AMP * DEC ** np.arange(T)
        rs.append(solve_path(ss, np.zeros(T), dd, masksK,
                             extra_mask=em, Hb=HB))
    return np.array(rs)

masks4, _ = make_regions(ss['mu_end'], 4)
m4, qb4, J4m = level(4)
assert np.array_equal(masks4, m4)
res4 = injections(m4)
oinv4 = omega_inv_diag(qb4, SIGMA_R, NE)

def avg_fit(rho, J, oinv, res):
    M = projection_matrix(J, oinv, rho)
    return np.mean([1 - np.linalg.norm(r - J@(M@r))/np.linalg.norm(r)
                    for r in res])

lo, hi = -4.0, 10.0
for _ in range(70):
    mid = 0.5*(lo+hi)
    if avg_fit(10**mid, J4m, oinv4, res4) < 0.95: lo = mid
    else: hi = mid
RHO = 10**hi
print('rho calibrated at K=4 (augmented moments): %.4g' % RHO)
results['NE'], results['rho'], results['sigma_r'] = NE, float(RHO), SIGMA_R
results['alpha'], results['amp'], results['Hb'] = ALPHA, AMP, HB

def analyze(Kc):
    masksK, qbK, JK = level(Kc)
    oinv = omega_inv_diag(qbK, SIGMA_R, NE)
    M = projection_matrix(JK, oinv, RHO)
    Qs, Qt = quad_forms(M, oinv, Kc)
    assert np.allclose(sum(Qs), Qt)
    res = injections(masksK)
    fits = [1 - np.linalg.norm(r - JK@(M@r))/np.linalg.norm(r) for r in res]
    caps = [capacity(Q, Qt) for Q in Qs]
    lmax = np.array([c[1] for c in caps]); lmin = np.array([c[0] for c in caps])
    B = Kc + 1
    S = np.empty((B, B)); P = np.empty((B, B)); U = np.empty((B, B))
    att_pw = np.empty(B, int)
    for l in range(B):
        r = res[l]
        for b in range(B):
            S[l, b] = (r @ Qs[b] @ r) / (r @ Qt @ r)
            P[l, b] = p_value(S[l, b], Qs[b], Qt)
        U[l] = S[l] / lmax
        att_pw[l] = attribute(r, Qs, Qt)[0]
    att_u = U.argmax(1)
    margin = np.array([np.sort(U[l])[-1] - np.sort(U[l])[-2] for l in range(B)])
    ok_u = bool(np.all(att_u == np.arange(B)))
    ok_pw = bool(np.all(att_pw == np.arange(B)))
    p_true = np.array([P[l, l] for l in range(B)])
    info = []
    for l in range(B):
        s_, I_k, I_tot = shares_and_info(res[l], Qs, Qt)
        info.append(dict(I=float(I_tot), dep=dep(I_tot)))
    V = [c[2] for c in caps]
    cosV = [[cos_angle_qtot(V[a], V[b], Qt) for b in range(B)] for a in range(B)]
    return dict(K=Kc, qb=qbK.tolist(), fits=[float(f) for f in fits],
                lmax=lmax.tolist(), lmin=lmin.tolist(),
                S=S.tolist(), P=P.tolist(), U=U.tolist(),
                att_u=att_u.tolist(), att_pw=att_pw.tolist(),
                margin=margin.tolist(), ok_u=ok_u, ok_pw=ok_pw,
                p_true=p_true.tolist(), info=info, cosV=cosV), res

levels = {}
for Kc in [1, 2, 4, 8]:
    t0 = time.time()
    levels[Kc], resK = analyze(Kc)
    L = levels[Kc]
    print('K=%d: util-attr %s (ok=%s, min margin %.4f) | pairwise %s (ok=%s) '
          '| p_true max %.4g | fits %.3f-%.3f  [%.1fs]'
          % (Kc, L['att_u'], L['ok_u'], min(L['margin']), L['att_pw'],
             L['ok_pw'], max(L['p_true']), min(L['fits']), max(L['fits']),
             time.time() - t0))
    if Kc == 4:
        np.save('res4m.npy', resK)
    if Kc == 8:
        np.save('res8m.npy', resK)

results['levels'] = {str(k): v for k, v in levels.items()}

agg = json.load(open('results.json'))
print('\n--- resolution verdict (utilization attribution), per level ---')
print('%-4s %-22s %-22s' % ('K', 'agg-only moments', 'augmented (+bottom50)'))
cmp_table = {}
for Kc in [1, 2, 4, 8]:
    a = agg['levels'][str(Kc)]; m = levels[Kc]
    sa = '%s  margin %.3g' % ('OK ' if a['ok_u'] else 'FAIL', min(a['margin']))
    sm = '%s  margin %.3g' % ('OK ' if m['ok_u'] else 'FAIL', min(m['margin']))
    print('%-4d %-22s %-22s' % (Kc, sa, sm))
    cmp_table[str(Kc)] = dict(agg_ok=a['ok_u'], agg_margin=min(a['margin']),
                              micro_ok=m['ok_u'], micro_margin=min(m['margin']),
                              agg_ptrue=max(a['p_true']),
                              micro_ptrue=max(m['p_true']))
results['comparison'] = cmp_table

json.dump(results, open('results_micro.json', 'w'), indent=1)
print('TOTAL %.1fs' % (time.time() - t_all))
