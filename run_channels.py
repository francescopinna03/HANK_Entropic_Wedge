import numpy as np, json, warnings, time
warnings.filterwarnings('ignore')
from model import *
from entropic import *
from closure import ge_transition
from channels import *

NE = 8000
SIGMA = 0.0025
DEC = 0.85
AMP = 0.15
HB = 12
DR_INJ = -0.005 * 0.8 ** np.arange(T)
DB_INJ = 0.005 * 0.8 ** np.arange(T)
PATH = AMP * DEC ** np.arange(T)

t_all = time.time()
ss = steady_state()
mu_a = ss['mu'].sum(1)
em = np.cumsum(mu_a) <= 0.5 + 1e-12
masks8, _ = make_regions(ss['mu_end'], 8)
qs8 = qbar_states(ss['mu_end'], masks8)

t0 = time.time()
Jy8, JA8, y0, A0 = pe_jacobians_states(ss, masks8, extra_mask=em, Hb=HB)
print('state-specific Jacobians at K=8 built in %.1fs, %d columns' % (time.time() - t0, Jy8['d'].shape[1] + 2 * T))
results = dict(NE=NE, sigma=SIGMA, amp=AMP, dec=DEC, baseline_gap=float(np.abs(y0).max()))

masks4, _ = make_regions(ss['mu_end'], 4)
d2 = np.zeros((4, T)); d2[1] = PATH
d3 = np.zeros((4, NZ, T)); d3[1, 0] = PATH; d3[1, 1] = PATH
results['symmetric_reconstruction_gap'] = float(np.abs(solve_path(ss, np.zeros(T), d2, masks4) - solve_path(ss, np.zeros(T), d3, masks4)).max())
Jy4, _ = aggregate_states(Jy8, JA8, 4)
sym_cols = sum(Jy4['d'][:H, (1*NZ+z)*T:(1*NZ+z+1)*T] for z in range(NZ))
d2c = np.zeros((4, T)); d2c[1, 5] = 1e-4
ref = (solve_path(ss, np.zeros(T), d2c, masks4) - solve_path(ss, np.zeros(T), np.zeros((4, T)), masks4)) / 1e-4
results['symmetric_column_gap'] = float(np.abs(sym_cols[:, 5] - ref).max() / np.abs(ref).max())
print('symmetric deformation reconstructed: path gap %.1e, Jacobian column gap %.1e'
      % (results['symmetric_reconstruction_gap'], results['symmetric_column_gap']))


def injections(masksK, blocks, closure, with_micro):
    Kc = masksK.shape[0]
    extra = em if with_micro else None
    def run(dr_or_db, dd):
        if closure == 'pe':
            return solve_path(ss, dr_or_db, dd, masksK, extra_mask=extra, Hb=HB)
        return ge_transition(ss, dr_or_db, dd, masksK, JA8['r'], extra_mask=extra, Hb=HB)[0]
    agg = DR_INJ if closure == 'pe' else DB_INJ
    out = [run(agg, np.zeros((Kc, NZ, T)))]
    for (k, z) in blocks:
        dd = np.zeros((Kc, NZ, T)); dd[k, z] = PATH
        out.append(run(np.zeros(T), dd))
    chan = []
    for k in range(Kc):
        if sum(1 for (kk, z) in blocks if kk == k) < NZ:
            continue
        for name, sign in [('persistence', 1.0), ('mean', -1.0)]:
            dd = np.zeros((Kc, NZ, T)); dd[k, 0] = PATH; dd[k, 1] = sign * PATH
            chan.append((k, name, run(np.zeros(T), dd)))
    return np.array(out), chan


def calibrate_rho(J, oinv, res):
    lo, hi = -4.0, 10.0
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        M = projection_matrix(J, oinv, 10 ** mid)
        f = np.mean([1 - np.linalg.norm(r - J @ (M @ r)) / np.linalg.norm(r) for r in res])
        if f < 0.95: lo = mid
        else: hi = mid
    return 10 ** hi


def attribution(res, Qs, Qt):
    caps = np.array([capacity(Q, Qt)[1] for Q in Qs])
    U = np.array([[(r @ Q @ r) / (r @ Qt @ r) for Q in Qs] for r in res]) / caps
    att = U.argmax(1)
    margin = np.array([np.sort(u)[-1] - np.sort(u)[-2] for u in U])
    return caps, U, att, margin


def analyze(Kc, closure, with_micro):
    JyK, JAK = aggregate_states(Jy8, JA8, Kc)
    g = 8 // Kc
    masksK = np.array([masks8[g*k:g*(k+1)].any(0) for k in range(Kc)])
    qsK = np.array([qs8[g*k:g*(k+1)].sum(0) for k in range(Kc)])
    blocks = kept_blocks(qsK)
    if closure == 'pe':
        J = pe_matrix(JyK, blocks)
    else:
        J = ge_matrix(JyK, JAK, blocks, with_micro)
    if not with_micro:
        J = J[:H]
    nb = len(blocks)
    oinv = omega_inv_diag(np.array([qsK[k, z] for (k, z) in blocks]), SIGMA, NE)
    res, chan = injections(masksK, blocks, closure, with_micro)
    rho = calibrate_rho(J, oinv, res)
    M = projection_matrix(J, oinv, rho)
    Qs, Qt = quad_forms(M, oinv, nb)
    assert np.allclose(sum(Qs), Qt)
    caps, U, att, margin = attribution(res, Qs, Qt)
    groups = region_groups(blocks, Kc)
    Qr = [Qs[0]] + [sum(Qs[b] for b in grp) for grp in groups]
    target_region = [0] + [1 + k for (k, z) in blocks]
    capr, Ur, attr, marginr = attribution(res, Qr, Qt)
    chan_res = np.array([c[2] for c in chan]) if chan else np.zeros((0, J.shape[0]))
    chan_out = []
    if len(chan):
        _, Uc, attc, marginc = attribution(chan_res, Qr, Qt)
        for i, (k, name, r) in enumerate(chan):
            u = M @ r
            share_mean = channel_split(u, blocks, Kc)[k]
            I_tot = 0.5 * r @ Qt @ r
            chan_out.append(dict(region=int(k), channel=name, att_region=int(attc[i]), ok=bool(attc[i] == 1 + k),
                                 margin=float(marginc[i]), mean_share_of_correction=share_mean,
                                 I=float(I_tot), dep=dep(I_tot)))
    info = [dict(I=float(0.5 * r @ Qt @ r), dep=dep(0.5 * r @ Qt @ r)) for r in res]
    return dict(K=Kc, closure=closure, with_micro=with_micro, rho=float(rho),
                blocks=[[int(k), int(z)] for (k, z) in blocks], qb=[float(qsK[k, z]) for (k, z) in blocks],
                state_caps=caps.tolist(), state_att=att.tolist(), state_ok=bool(np.all(att == np.arange(nb + 1))),
                state_margin=margin.tolist(),
                region_caps=capr.tolist(), region_att=attr.tolist(),
                region_ok=bool(np.all(attr == np.array(target_region))), region_margin=marginr.tolist(),
                channels=chan_out, info=info)


for closure in ['pe', 'ge']:
    for with_micro in [False, True]:
        tag = '%s_%s' % (closure, 'augmented' if with_micro else 'aggregate')
        results[tag] = {}
        for Kc in [1, 2, 4, 8]:
            t0 = time.time()
            L = analyze(Kc, closure, with_micro)
            results[tag][str(Kc)] = L
            ch = ' '.join('%d%s:%s' % (c['region'] + 1, c['channel'][0], 'ok' if c['ok'] else 'X') for c in L['channels'])
            print('[%s] K=%d blocks=%d | state ok=%s min margin %.3g | region ok=%s min margin %.3g | channels %s [%.1fs]'
                  % (tag, Kc, len(L['blocks']), L['state_ok'], min(L['state_margin']), L['region_ok'],
                     min(L['region_margin']), ch, time.time() - t0))

json.dump(results, open('results_channels.json', 'w'), indent=1)
print('TOTAL %.1fs' % (time.time() - t_all))
