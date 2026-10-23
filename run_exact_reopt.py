import numpy as np, json, time, sys, os, warnings
warnings.filterwarnings('ignore')
from scipy.stats import norm
from model import *
from exact import *
from channels import pe_jacobians_states

NE = 8000
DEC = 0.85
AMPS = [0.05, 0.15, 0.30, 0.60]
K = 4
OUT = 'results_exact_reopt.json'
RESUME = '--resume' in sys.argv

t_all = time.time()
ss = steady_state()
mu_a = ss['mu'].sum(1)
em = np.cumsum(mu_a) <= 0.5 + 1e-12
st = frozen_setup(ss, em)
masks, _ = make_regions(ss['mu_end'], K)
w = second_order_weights(masks, st)
blocks = [(k, z) for k in range(K) for z in range(NZ) if w[k, z] > 1e-12]
t0 = time.time()
Jy, JA, y0, A0 = pe_jacobians_states(ss, masks, extra_mask=em)
Jr = Jy['d']
print('re-optimising Jacobian at K=4 built in %.1fs' % (time.time() - t0))
fr = lambda d: forward_reopt(d, masks, ss, em)


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
if RESUME and os.path.exists(OUT):
    results = json.load(open(OUT))
else:
    results = dict(NE=NE, K=K, blocks=[[int(k), int(z)] for (k, z) in blocks], second_order_weights=w.tolist(), cases=[])
done = {(c['kind'], c['region'], c['amp']) for c in results['cases']}
for kind, k in cases:
    for amp in AMPS:
        if (kind, k, amp) in done:
            continue
        if RESUME and time.time() - t_all > 240:
            break
        t0 = time.time()
        d = make_injection(kind, k, amp)
        dy, kl_inj = fr(d)
        Wq, u = quadratic_restricted(dy, Jr, w, blocks)
        er = exact_restricted(dy, masks, st, blocks, u0=u, forward=fr)
        dyf, klf = forward_with_delta(d, masks, st)
        ef = exact_restricted(dyf, masks, st, blocks)
        row = dict(kind=kind, region=int(k), amp=amp, norm_dy=float(np.linalg.norm(dy)), norm_dy_frozen=float(np.linalg.norm(dyf)),
                   KL_injected=float(kl_inj), W_quadratic=float(Wq), W_exact_reopt=float(er['W']), W_exact_frozen=float(ef['W']),
                   feas_reopt=er['feasibility'], feas_frozen=ef['feasibility'], nit_reopt=er['nit'], nit_frozen=ef['nit'],
                   dep_injected=dep(kl_inj), dep_quadratic=dep(Wq), dep_exact_reopt=dep(er['W']), dep_exact_frozen=dep(ef['W']),
                   shadow_prices=er['prices'].tolist(), moments_reopt=dy.tolist())
        if kind in ('persistence', 'mean'):
            dq = np.zeros((K, NZ, T))
            for i, (kk, z) in enumerate(blocks):
                dq[kk, z] = u[i * T:(i + 1) * T]
            row['mean_share_quadratic'] = channel_share(dq, k)
            row['mean_share_exact_reopt'] = channel_share(er['delta'], k)
            row['mean_share_exact_frozen'] = channel_share(ef['delta'], k)
        results['cases'].append(row)
        print('%-11s Q%d amp %.2f | KL_inj %.3e | quad %.3e | exact_reopt %.3e | exact_frozen %.3e | reopt/quad %.4f | reopt/frozen %.2f | DEP inj %.3f reopt %.3f%s [%.0fs]'
              % (kind, k + 1, amp, kl_inj, Wq, er['W'], ef['W'], er['W'] / Wq, er['W'] / ef['W'], dep(kl_inj), dep(er['W']),
                 (' | mean share quad %.2f reopt %.2f frozen %.2f' % (row['mean_share_quadratic'], row['mean_share_exact_reopt'], row['mean_share_exact_frozen'])) if 'mean_share_exact_reopt' in row else '',
                 time.time() - t0), flush=True)
        json.dump(results, open(OUT, 'w'), indent=1)
print('TOTAL %.0fs' % (time.time() - t_all))
