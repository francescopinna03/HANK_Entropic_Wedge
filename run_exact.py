import numpy as np, json, time, warnings
warnings.filterwarnings('ignore')
from scipy.stats import norm
from model import *
from exact import *

NE = 8000
DEC = 0.85
AMPS = [0.05, 0.15, 0.30, 0.60]
K = 4

t_all = time.time()
ss = steady_state()
mu_a = ss['mu'].sum(1)
em = np.cumsum(mu_a) <= 0.5 + 1e-12
st = frozen_setup(ss, em)
masks, _ = make_regions(ss['mu_end'], K)
w = second_order_weights(masks, st)
blocks = [(k, z) for k in range(K) for z in range(NZ) if w[k, z] > 1e-12]
t0 = time.time()
Jf = frozen_jacobian(masks, st)
np.save('Jf4.npy', Jf)
print('frozen-policy Jacobian built in %.1fs' % (time.time() - t0))


def dep(Wpc):
    return float(norm.cdf(-np.sqrt(NE * Wpc / 2.0)))


def make_injection(kind, k, amp):
    path = amp * DEC ** np.arange(T)
    d = np.zeros((K, NZ, T))
    if kind == 'persistence':
        d[k, 0] = path; d[k, 1] = path
    elif kind == 'mean':
        d[k, 0] = path; d[k, 1] = -path
    elif kind == 'low':
        d[k, 0] = path
    elif kind == 'uniform':
        d[:, 0] = path; d[:, 1] = path
    return d


cases = [('low', 0), ('persistence', 1), ('mean', 1), ('persistence', 2), ('mean', 2), ('persistence', 3), ('mean', 3), ('uniform', 0)]
results = dict(NE=NE, K=K, blocks=[[int(k), int(z)] for (k, z) in blocks], second_order_weights=w.tolist(), cases=[])
for kind, k in cases:
    for amp in AMPS:
        t0 = time.time()
        d = make_injection(kind, k, amp)
        dy, kl_inj = forward_with_delta(d, masks, st)
        Wq, u = quadratic_restricted(dy, Jf, w, blocks)
        er = exact_restricted(dy, masks, st, blocks, u0=u)
        ef = solve_exact(dy, st)
        row = dict(kind=kind, region=int(k), amp=amp, norm_dy=float(np.linalg.norm(dy)),
                   KL_injected=float(kl_inj), W_quadratic=float(Wq), W_exact_restricted=float(er['W']),
                   W_exact_full=float(ef['W']), feas_restricted=er['feasibility'], feas_full=ef['feasibility'],
                   nit_restricted=er['nit'], nit_full=ef['nit'],
                   dep_injected=dep(kl_inj), dep_quadratic=dep(Wq), dep_exact_restricted=dep(er['W']), dep_exact_full=dep(ef['W']),
                   kl_by_block_exact=kl_by_block(er['delta'], masks, st).tolist(),
                   kl_by_block_injected=kl_by_block(d, masks, st).tolist())
        if kind in ('persistence', 'mean') :
            dq = np.zeros((K, NZ, T))
            for i, (kk, z) in enumerate(blocks):
                dq[kk, z] = u[i * T:(i + 1) * T]
            row['mean_share_quadratic'] = channel_share(dq, k)
            row['mean_share_exact'] = channel_share(er['delta'], k)
        if kind == 'uniform' and amp == 0.15:
            el = solve_exact(dy, st, tilt_lottery=True, tol=1e-7)
            row['W_exact_full_lottery_tilt'] = float(el['W'])
        results['cases'].append(row)
        print('%-11s Q%d amp %.2f | KL_inj %.3e | quad %.3e | exact_r %.3e | exact_full %.3e | ratio exact_r/quad %.4f | full/restr %.3f | DEP inj %.3f quad %.3f full %.3f%s [%.0fs]'
              % (kind, k + 1, amp, kl_inj, Wq, er['W'], ef['W'], er['W'] / Wq, ef['W'] / er['W'], dep(kl_inj), dep(Wq), dep(ef['W']),
                 (' | mean share quad %.2f exact %.2f' % (row['mean_share_quadratic'], row['mean_share_exact'])) if 'mean_share_exact' in row else '',
                 time.time() - t0), flush=True)
        json.dump(results, open('results_exact.json', 'w'), indent=1)
print('TOTAL %.0fs' % (time.time() - t_all))
