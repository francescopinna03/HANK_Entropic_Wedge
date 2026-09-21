# Economic descriptors of the toy economy, used in the note on the code.

import json, os, warnings
import numpy as np
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from model import *
from plotstyle import BLU, GRI, ROS, legend_below, save

os.makedirs('figs', exist_ok=True)

DEC, AMP, HB = 0.85, 0.15, 12
DR_INJ = -0.005 * 0.8 ** np.arange(T)

ss = steady_state()
mu, c, mu_end = ss['mu'], ss['c'], ss['mu_end']
out = {}

# steady state
eps = 1e-4
mpc = np.empty_like(c)
for z in range(NZ):
    # a transfer eps raises cash on hand by eps, i.e. assets by eps/(1+r)
    mpc[:, z] = (np.interp(AGRID + eps / (1.0 + R_SS), AGRID, c[:, z])
                 - c[:, z]) / eps
out['steady_state'] = dict(
    C=ss['C'], A=ss['A'], A_over_quarterly_income=ss['A'] / float(Y.mean()),
    mass_constraint=float(mu[0].sum()),
    mass_constraint_low_high=[float(x) for x in mu[0]],
    avg_quarterly_mpc=float(np.sum(mu * mpc)))

# quartiles
# beginning-of-period quartiles for MPC and wealth shares

ma = mu.sum(1); cum = np.cumsum(ma)
cuts = [0] + [int(np.searchsorted(cum, q)) + 1 for q in (0.25, 0.5, 0.75)] + [NA]
W = float(np.sum(mu * AGRID[:, None]))
m4, _ = make_regions(mu_end, 4)
qb4 = qbar(mu_end, m4)
quart = []
for k in range(4):
    sl = slice(cuts[k], cuts[k + 1])
    mk = float(mu[sl].sum())
    ek = mu_end[m4[k]]
    idx = np.where(m4[k])[0]
    quart.append(dict(
        mass_bop=mk,
        mpc=float(np.sum(mu[sl] * mpc[sl]) / mk),
        wealth_share=float(np.sum(mu[sl] * AGRID[sl, None]) / W),
        mass_eop=float(ek.sum()),
        high_state_share_eop=float(ek[:, 1].sum() / ek.sum()),
        a_range_eop=[float(AGRID[idx[0]]), float(AGRID[idx[-1]])],
        qbar=float(qb4[k])))
out['quartiles'] = quart

# responses to the injections
resp = {}
for k in range(4):
    dd = np.zeros((4, T)); dd[k] = AMP * DEC ** np.arange(T)
    resp['tilt_A%d' % (k + 1)] = solve_path(ss, np.zeros(T), dd, m4)
resp['rate_cut'] = solve_path(ss, DR_INJ, np.zeros((4, T)), m4)
out['responses_pp'] = {key: v.tolist() for key, v in resp.items()}

# rate / uniform churn confounding
m1, _ = make_regions(mu_end, 1)
em = np.cumsum(ma) <= 0.5 + 1e-12   # bottom-50% as in run_micro.py
rr = solve_path(ss, DR_INJ, np.zeros((1, T)), m1, extra_mask=em, Hb=HB)
dd = np.zeros((1, T)); dd[0] = AMP * DEC ** np.arange(T)
ru = solve_path(ss, np.zeros(T), dd, m1, extra_mask=em, Hb=HB)
cosang = lambda x, y: float(x @ y / np.linalg.norm(x) / np.linalg.norm(y))
out['confounding'] = dict(
    cos_aggregate_only=cosang(rr[:H], ru[:H]),
    cos_bottom50_block=cosang(rr[H:], ru[H:]),
    cos_augmented=cosang(rr, ru),
    rate_cut_aggregate=rr[:H].tolist(), uniform_churn_aggregate=ru[:H].tolist(),
    rate_cut_bottom50=rr[H:].tolist(), uniform_churn_bottom50=ru[H:].tolist())

json.dump(out, open('results_economics.json', 'w'), indent=1)

# figure F0
tq = np.arange(H)
fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.1))
styles = [BLU, '#4f81bd', ROS, '#d99694']
for k in range(4):
    hs = quart[k]['high_state_share_eop']
    ax[0].plot(tq, resp['tilt_A%d' % (k + 1)][:H], color=styles[k],
               label='$Q_%d$, %d%% a reddito alto' % (k + 1, round(100 * hs)))
ax[0].axhline(0, color='k', lw=0.5)
ax[0].set_xlabel('trimestri'); ax[0].set_ylabel('consumo aggregato, p.p.')
ax[0].set_title('(a) churn in un quartile di ricchezza')
legend_below(ax[0], ncol=2, gap=0.27)
ax[1].plot(tq, rr[:H], color=GRI, ls='--', label='taglio del tasso')
ax[1].plot(tq, ru[:H], color=BLU, label='aumento uniforme del churn')
ax[1].axhline(0, color='k', lw=0.5)
ax[1].set_xlabel('trimestri'); ax[1].set_ylabel('consumo aggregato, p.p.')
ax[1].set_title('(b) taglio del tasso e churn uniforme')
legend_below(ax[1], ncol=1, gap=0.27)
fig.tight_layout(w_pad=2.0)
save(fig, 'F0_economia')

print(json.dumps(out['steady_state'], indent=1))
for k, q in enumerate(quart):
    print('Q%d' % (k + 1), {a: (round(b, 3) if isinstance(b, float) else
                               [round(x, 3) for x in b]) for a, b in q.items()})
print({a: round(b, 3) for a, b in out['confounding'].items()
       if a.startswith('cos')})
