import numpy as np, json, warnings, time
warnings.filterwarnings('ignore')
from model import *
from entropic import *
from closure import *

NE = 8000
SIGMA_B = 0.0025
ALPHA = 0.05
DEC = 0.85
AMP = 0.15
HB = 12
DB_INJ = 0.005 * 0.8 ** np.arange(T)

t_all = time.time()
ss = steady_state()
results = dict(NE=NE, sigma_b=SIGMA_B, alpha=ALPHA, amp=AMP, dec=DEC, B=ss['A'])

mu_a = ss['mu'].sum(1)
em = np.cumsum(mu_a) <= 0.5 + 1e-12
masks8, masses8 = make_regions(ss['mu_end'], 8)
qb8 = qbar(ss['mu_end'], masks8)

t0 = time.time()
Jy8, JA8, y0, A0 = pe_jacobians(ss, masks8, extra_mask=em, Hb=HB)
print('PE Jacobians at K=8 built in %.1fs' % (time.time() - t0))
np.save('Jy8_ge.npy', np.hstack([Jy8['r'], Jy8['b'], Jy8['d']]))
np.save('JA8_ge.npy', np.hstack([JA8['r'], JA8['b'], JA8['d']]))
results['pe_baseline_gap'] = dict(y=float(np.abs(y0).max()), A=float(np.abs(A0 - ss['A']).max()))


def level(Kc, with_micro):
    g = 8 // Kc
    masksK = np.array([masks8[g*k:g*(k+1)].any(0) for k in range(Kc)])
    qbK = np.array([qb8[g*k:g*(k+1)].sum() for k in range(Kc)])
    JyK, JAK = aggregate_pe(Jy8, JA8, Kc)
    Jt, dr_db, dr_dd = ge_jacobian(JyK, JAK, with_micro)
    return masksK, qbK, Jt, JAK['r'], dr_db, dr_dd


def injections(masksK, with_micro):
    Kc = masksK.shape[0]
    extra = em if with_micro else None
    out = []
    m, dr, y, it, res = ge_transition(ss, DB_INJ, np.zeros((Kc, T)), masksK, JA8['r'], extra_mask=extra, Hb=HB)
    out.append((m, dr, it, res))
    for k in range(Kc):
        dd = np.zeros((Kc, T)); dd[k] = AMP * DEC ** np.arange(T)
        m, dr, y, it, res = ge_transition(ss, np.zeros(T), dd, masksK, JA8['r'], extra_mask=extra, Hb=HB)
        out.append((m, dr, it, res))
    return out


def avg_fit(rho, J, oinv, res):
    M = projection_matrix(J, oinv, rho)
    return np.mean([1 - np.linalg.norm(r - J@(M@r))/np.linalg.norm(r) for r in res])


def calibrate_rho(J, oinv, res):
    lo, hi = -4.0, 10.0
    for _ in range(70):
        mid = 0.5*(lo+hi)
        if avg_fit(10**mid, J, oinv, res) < 0.95: lo = mid
        else: hi = mid
    return 10**hi


def analyze(Kc, with_micro, rho):
    masksK, qbK, JK, JAr, dr_db, dr_dd = level(Kc, with_micro)
    oinv = omega_inv_diag(qbK, SIGMA_B, NE)
    M = projection_matrix(JK, oinv, rho)
    Qs, Qt = quad_forms(M, oinv, Kc)
    assert np.allclose(sum(Qs), Qt)
    inj = injections(masksK, with_micro)
    res = np.array([x[0] for x in inj])
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
    info = []
    for l in range(B):
        _, _, I_tot = shares_and_info(res[l], Qs, Qt)
        info.append(dict(I=float(I_tot), dep=dep(I_tot)))
    conc = []
    for l in range(B):
        eta = np.zeros(T * (1 + Kc))
        if l == 0:
            eta[:T] = DB_INJ
        else:
            eta[T*l:T*(l+1)] = AMP * DEC ** np.arange(T)
        conc.append(float(np.linalg.norm(res[l] - JK @ eta) / np.linalg.norm(res[l])))
    out = dict(K=Kc, with_micro=with_micro, rho=float(rho),
               masses=(masksK @ ss['mu_end'].sum(1)).tolist(), qb=qbK.tolist(),
               fits=[float(f) for f in fits], lmax=lmax.tolist(), lmin=lmin.tolist(),
               S=S.tolist(), P=P.tolist(), U=U.tolist(),
               att_u=att_u.tolist(), att_pw=att_pw.tolist(), margin=margin.tolist(),
               ok_u=bool(np.all(att_u == np.arange(B))), ok_pw=bool(np.all(att_pw == np.arange(B))),
               p_true=[float(P[l, l]) for l in range(B)], info=info, linearization=conc,
               broyden=[dict(it=int(x[2]), resid=float(x[3])) for x in inj],
               monitors=monitors(JAr, JK))
    return out, res, inj, (Qs, Qt, M, JK)


for with_micro in [False, True]:
    tag = 'augmented' if with_micro else 'rate_only'
    masks4, qb4, J4, JAr4, _, _ = level(4, with_micro)
    inj4 = injections(masks4, with_micro)
    res4 = np.array([x[0] for x in inj4])
    rho = calibrate_rho(J4, omega_inv_diag(qb4, SIGMA_B, NE), res4)
    print('[%s] rho calibrated at K=4: %.4g' % (tag, rho))
    levels = {}
    for Kc in [1, 2, 4, 8]:
        t0 = time.time()
        levels[str(Kc)], res, inj, objs = analyze(Kc, with_micro, rho)
        L = levels[str(Kc)]
        print('[%s] K=%d: util-attr %s (ok=%s, min margin %.3g) | pairwise %s (ok=%s) | p_true max %.3g | lmax %s | fits %.3f-%.3f | broyden it %s [%.1fs]'
              % (tag, Kc, L['att_u'], L['ok_u'], min(L['margin']), L['att_pw'], L['ok_pw'],
                 max(L['p_true']), np.round(L['lmax'], 3).tolist(), min(L['fits']), max(L['fits']),
                 [b['it'] for b in L['broyden']], time.time() - t0))
    results[tag] = dict(rho=float(rho), levels=levels)

masks4, _ = make_regions(ss['mu_end'], 4)
m_b, dr_b, y_b, it_b, res_b = ge_transition(ss, DB_INJ, np.zeros((4, T)), masks4, JA8['r'], extra_mask=em, Hb=HB)
dd = np.tile(AMP * DEC ** np.arange(T), (4, 1))
m_u, dr_u, y_u, it_u, res_u = ge_transition(ss, np.zeros(T), dd, masks4, JA8['r'], extra_mask=em, Hb=HB)
quart = []
for k in range(4):
    d1 = np.zeros((4, T)); d1[k] = AMP * DEC ** np.arange(T)
    m_k, dr_k, y_k, _, _ = ge_transition(ss, np.zeros(T), d1, masks4, JA8['r'], extra_mask=em, Hb=HB)
    quart.append(dict(dr_bp=(1e4 * dr_k[:H]).tolist(), dCb=y_k[H:].tolist(), dC=y_k[:H].tolist(),
                      cos_r_with_beta=cosine(dr_k[:H], dr_b[:H]), cos_cb_with_beta=cosine(y_k[H:], y_b[H:])))
results['confounding'] = dict(
    beta=dict(dr_bp=(1e4 * dr_b[:H]).tolist(), dCb=y_b[H:].tolist(), dC=y_b[:H].tolist()),
    uniform=dict(dr_bp=(1e4 * dr_u[:H]).tolist(), dCb=y_u[H:].tolist(), dC=y_u[:H].tolist()),
    cos_r=cosine(dr_b[:H], dr_u[:H]), cos_cb=cosine(y_b[H:], y_u[H:]),
    cos_all=cosine(m_b, m_u), quartiles=quart,
    agg_identity_gap=float(np.abs(y_b[:H] - 100.0 * dr_b[:H] * ss['A'] / ss['C']).max()),
    linear_vs_nonlinear_dr=float(np.linalg.norm(dr_b - level(4, True)[4] @ DB_INJ) / np.linalg.norm(dr_b)))
print('confounding in GE: cos(r) beta vs uniform churn %.3f | cos(bottom-half C) %.3f | cos(all moments) %.3f'
      % (results['confounding']['cos_r'], results['confounding']['cos_cb'], results['confounding']['cos_all']))
print('aggregate identity gap %.1e | linear vs nonlinear dr %.2e'
      % (results['confounding']['agg_identity_gap'], results['confounding']['linear_vs_nonlinear_dr']))
results['ss'] = dict(C=ss['C'], A=ss['A'], mass_constraint=float(ss['mu'][0, :].sum()), r=R_SS)
results['masses8'] = masses8.tolist()
json.dump(results, open('results_ge.json', 'w'), indent=1)
print('TOTAL %.1fs' % (time.time() - t_all))
