#Full resolution analysis over the nested dictionary family K in {1,2,4,8}.
import numpy as np, json, warnings, time
warnings.filterwarnings('ignore')
from model import *
from entropic import *

NE = 8000 
SIGMA_R = 0.0025 
RHO = None 
ALPHA = 0.05
DEC = 0.85 
AMP = 0.15
DR_INJ = -0.005 * 0.8 ** np.arange(T)

t_all = time.time()
ss = steady_state()
results = {}

# octile dictionary: masks, qbar, Jacobian
masks8, masses8 = make_regions(ss['mu_end'], 8)
qb8 = qbar(ss['mu_end'], masks8)
t0 = time.time()
J8 = build_jacobian(ss, masks8) 
print('J8 built in %.1fs' % (time.time() - t0))

def level(Kc):
    g = 8 // Kc
    masksK = np.array([masks8[g*k:g*(k+1)].any(0) for k in range(Kc)])
    qbK = np.array([qb8[g*k:g*(k+1)].sum() for k in range(Kc)])
    JK = np.empty((H, T * (1 + Kc)))
    JK[:, :T] = J8[:, :T]
    for k in range(Kc):
        cols = sum(J8[:, T*(1+j):T*(2+j)] for j in range(g*k, g*(k+1)))
        JK[:, T*(1+k):T*(2+k)] = cols
    return masksK, qbK, JK

# consistency: K=4 from sums == K=4 built directly
masks4, masses4 = make_regions(ss['mu_end'], 4)
m4s, qb4s, J4s = level(4)
assert np.array_equal(masks4, m4s)
J4_direct = np.load('J4.npy') if __import__('os').path.exists('J4.npy') \
    else build_jacobian(ss, masks4)
gap = np.max(np.abs(J4s - J4_direct)) / np.max(np.abs(J4_direct))
print('nesting identity J4(sum of octile cols) vs direct: rel gap %.2e' % gap)
results['nesting_gap'] = float(gap)

def injections(masksK): # injections per level (nonlinear)
    Kc = masksK.shape[0]
    rs = [solve_path(ss, DR_INJ, np.zeros((Kc, T)), masksK)]
    for k in range(Kc):
        dd = np.zeros((Kc, T)); dd[k] = AMP * DEC ** np.arange(T)
        rs.append(solve_path(ss, np.zeros(T), dd, masksK))
    return np.array(rs)

res4 = injections(masks4)
oinv4 = omega_inv_diag(qb4s, SIGMA_R, NE)
lo, hi = -4.0, 10.0
def avg_fit(rho, J, oinv, res):
    M = projection_matrix(J, oinv, rho)
    return np.mean([1 - np.linalg.norm(r - J@(M@r))/np.linalg.norm(r)
                    for r in res])
for _ in range(70):
    mid = 0.5*(lo+hi)
    if avg_fit(10**mid, J4s, oinv4, res4) < 0.95: lo = mid
    else: hi = mid
RHO = 10**hi
print('rho calibrated at K=4: %.4g' % RHO)
results['NE'], results['rho'], results['sigma_r'] = NE, float(RHO), SIGMA_R
results['alpha'], results['amp'] = ALPHA, AMP

def analyze(Kc): # pre-level analysis
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
    PI = np.empty((B, B, B)); att_pw = np.empty(B, int)
    for l in range(B):
        r = res[l]
        for b in range(B):
            S[l, b] = (r @ Qs[b] @ r) / (r @ Qt @ r)
            P[l, b] = p_value(S[l, b], Qs[b], Qt)
        U[l] = S[l] / lmax
        k_star, score, _ = attribute(r, Qs, Qt)
        att_pw[l] = k_star
        PI[l] = pairwise_pvals(r, Qs, Qt)[0]
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
    return dict(K=Kc, masses=(masksK @ ss['mu_end'].sum(1)).tolist(),
                qb=qbK.tolist(), fits=[float(f) for f in fits],
                lmax=lmax.tolist(), lmin=lmin.tolist(),
                S=S.tolist(), P=P.tolist(), U=U.tolist(),
                att_u=att_u.tolist(), att_pw=att_pw.tolist(),
                margin=margin.tolist(), ok_u=ok_u, ok_pw=ok_pw,
                p_true=p_true.tolist(), info=info, cosV=cosV), res, (Qs, Qt, M, JK)

levels = {}
for Kc in [1, 2, 4, 8]:
    t0 = time.time()
    levels[Kc], resK, objs = analyze(Kc)
    L = levels[Kc]
    print('K=%d: util-attribution %s (ok=%s, min margin %.4f) | pairwise %s '
          '(ok=%s) | p_true max %.4g | fits %.3f-%.3f  [%.1fs]'
          % (Kc, L['att_u'], L['ok_u'], min(L['margin']), L['att_pw'],
             L['ok_pw'], max(L['p_true']), min(L['fits']), max(L['fits']),
             time.time() - t0))
    if Kc == 4:
        Qs4, Qt4, M4, J4l = objs; res4_saved = resK

results['levels'] = {str(k): v for k, v in levels.items()}

l, b = 0, 3 
r = res4_saved[l]
s_obs = (r @ Qs4[b] @ r) / (r @ Qt4 @ r)
p_im = p_value(s_obs, Qs4[b], Qt4)
p_mc, _ = mc_tail(s_obs, Qs4[b], Qt4, n_draws=400_000)
print('Imhof validation: s=%.4f p_Imhof=%.5f p_MC=%.5f' % (s_obs, p_im, p_mc))
results['imhof_check'] = dict(s=float(s_obs), p_imhof=float(p_im), p_mc=float(p_mc))

c3 = {}
for fac in [0.1, 1.0, 10.0]:
    Mf = projection_matrix(J4l, omega_inv_diag(np.array(levels[4]['qb']), SIGMA_R, NE), RHO * fac)
    Qf, Qtf = quad_forms(Mf, omega_inv_diag(np.array(levels[4]['qb']), SIGMA_R, NE), 4)
    r0 = res4_saved[0]
    sh = [float((r0 @ Q @ r0) / (r0 @ Qtf @ r0)) for Q in Qf]
    ft = 1 - np.linalg.norm(r0 - J4l @ (Mf @ r0)) / np.linalg.norm(r0)
    c3[str(fac)] = dict(shares=sh, fit=float(ft))
results['c3'] = c3

conc = [] #concordance
for l in range(5):
    eta = np.zeros(T * 5)
    if l == 0:
        eta[:T] = DR_INJ
    else:
        eta[T*l:T*(l+1)] = AMP * DEC ** np.arange(T)
    e = np.linalg.norm(res4_saved[l] - J4l @ eta) / np.linalg.norm(res4_saved[l])
    conc.append(float(e))
results['concordance'] = conc
print('linearization rel err per injection:', np.round(conc, 4))

cap_ne = {}
for ne in [1, 100, 1000, 8000, 100000]:
    oi = omega_inv_diag(np.array(levels[4]['qb']), SIGMA_R, ne)
    lo2, hi2 = -4.0, 10.0
    for _ in range(70):
        mid = 0.5*(lo2+hi2)
        if avg_fit(10**mid, J4l, oi, res4_saved) < 0.95: lo2 = mid
        else: hi2 = mid
    Mn = projection_matrix(J4l, oi, 10**hi2)
    Qn, Qtn = quad_forms(Mn, oi, 4)
    cap_ne[str(ne)] = dict(lmax=[capacity(Q, Qtn)[1] for Q in Qn],
                           rho=float(10**hi2))
results['cap_ne'] = cap_ne

results['ss'] = dict(C=ss['C'], A=ss['A'],
                     mass_constraint=float(ss['mu'][0, :].sum()))
results['masses8'] = masses8.tolist()
json.dump(results, open('results.json', 'w'), indent=1)
np.save('J8.npy', J8)
np.save('res4_final.npy', res4_saved)
print('TOTAL %.1fs' % (time.time() - t_all))
