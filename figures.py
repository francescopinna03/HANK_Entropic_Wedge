import numpy as np, json, warnings, os
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from model import *
from entropic import *
from plotstyle import BLU, GRI, ROS, VER, legend_below, save

os.makedirs('figs', exist_ok=True)

R = json.load(open('results.json'))
Rm = json.load(open('results_micro.json'))
RHO, NE, SR = R['rho'], R['NE'], R['sigma_r']
ss = steady_state()
masks8, _ = make_regions(ss['mu_end'], 8)
qb8 = qbar(ss['mu_end'], masks8)
J8 = np.load('J8.npy')

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

m4, qb4, J4 = level(4)
res4 = np.load('res4_final.npy')
tq = np.arange(H)

fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.1))
cols = [J4[:, 0]] + [J4[:, T*(1+k)] for k in range(4)]
labs = ['tasso $r$', '$\\delta_{A_1}$', '$\\delta_{A_2}$',
        '$\\delta_{A_3}$', '$\\delta_{A_4}$']
palette = [GRI, BLU, '#4f81bd', ROS, '#d99694']
for c, lb, co in zip(cols, labs, palette):
    cn = c / np.max(np.abs(c))
    ax[0].plot(tq, cn, label=lb, color=co,
               ls='--' if lb.startswith('tasso') else '-')
ax[0].axhline(0, color='k', lw=0.5)
ax[0].set_xlabel('trimestri'); ax[0].set_ylabel('risposta normalizzata')
ax[0].set_title('(a) colonne fake news, impulso in $t=0$')
legend_below(ax[0], ncol=3, gap=0.27)

m1, _, _ = level(1)
r_agg = solve_path(ss, -0.005 * 0.8 ** np.arange(T), np.zeros((1, T)), m1)
dd = np.zeros((1, T)); dd[0] = 0.15 * 0.85 ** np.arange(T)
r_uni = solve_path(ss, np.zeros(T), dd, m1)
ca = float(r_agg @ r_uni / np.linalg.norm(r_agg) / np.linalg.norm(r_uni))
sgn = -1.0 if ca < 0 else 1.0
ax[1].plot(tq, r_agg / np.max(np.abs(r_agg)), color=GRI, ls='--',
           label='iniezione sul tasso $r$')
ax[1].plot(tq, sgn * r_uni / np.max(np.abs(r_uni)), color=BLU,
           label='tilt uniforme ($K=1$), segno invertito')
ax[1].axhline(0, color='k', lw=0.5)
ax[1].set_xlabel('trimestri'); ax[1].set_ylabel('residuo normalizzato')
ax[1].set_title('(b) confondimento sui soli momenti aggregati')
ax[1].text(0.97, 0.95, '$|\\cos\\angle| = %.3f$' % abs(ca), transform=ax[1].transAxes,
           ha='right', va='top', fontsize=8.5)
legend_below(ax[1], ncol=1, gap=0.27)
fig.tight_layout(w_pad=2.0)
save(fig, 'F1_fakenews')
print('F1 done; cos(r, uniform tilt) =', round(ca, 4))

oinv4 = omega_inv_diag(qb4, SR, NE)
M4 = projection_matrix(J4, oinv4, RHO)
Qs4, Qt4 = quad_forms(M4, oinv4, 4)
b = 3
r0 = res4[0]
s_obs = float(r0 @ Qs4[b] @ r0 / (r0 @ Qt4 @ r0))
lmin_b, lmax_b, _ = capacity(Qs4[b], Qt4)
sg = np.linspace(1e-4, lmax_b * 0.999, 80)
p_curve = np.array([p_value(s, Qs4[b], Qt4) for s in sg])
p_obs = p_value(s_obs, Qs4[b], Qt4)
_, draws = mc_tail(s_obs, Qs4[b], Qt4, n_draws=400_000)
ds = np.sort(draws)
emp_s = ds
emp_p = 1.0 - np.arange(1, len(ds) + 1) / len(ds)

fig, ax = plt.subplots(figsize=(4.4, 3.1))
ax.plot(sg, p_curve, color=BLU, label='legge esatta (Imhof)')
sub = slice(0, len(emp_s), 400)
ax.plot(emp_s[sub], np.maximum(emp_p[sub], 1/len(ds)), color=ROS, ls=':',
        lw=1.2, label='coda empirica Monte Carlo ($4\\times10^5$ estrazioni)')
ax.axvline(s_obs, color=GRI, lw=0.8, ls='--')
ax.scatter([s_obs], [p_obs], color='k', zorder=5, s=14)
ax.annotate('$s_{oss}=%.4f$\n$p=%.4f$' % (s_obs, p_obs),
            xy=(s_obs, p_obs), xytext=(s_obs * 1.6, p_obs * 2.0), fontsize=8)
ax.set_yscale('log'); ax.set_ylim(1e-7, 1.6)
ax.set_xlim(-0.001, 1.15 * sg[np.argmax(p_curve < 1e-7)] if np.any(p_curve < 1e-7) else sg[-1])
ax.set_xlabel('quota $s$'); ax.set_ylabel('$p_k(s)=\\Pr(s_k(\\hat r)\\geq s)$')
ax.set_title('null direzionale, blocco $A_3$, iniezione aggregata ($K=4$)')
legend_below(ax, ncol=1, gap=0.24)
fig.tight_layout()
save(fig, 'F2_imhof')
print('F2 done; s_obs=%.4f p=%.5f' % (s_obs, p_obs))

S = np.array(R['levels']['4']['S']); P = np.array(R['levels']['4']['P'])
labs5 = ['agg', '$A_1$', '$A_2$', '$A_3$', '$A_4$']
cmap = LinearSegmentedColormap.from_list('w2b', ['#ffffff', BLU])
fig, ax = plt.subplots(1, 2, figsize=(6.6, 3.0))
im0 = ax[0].imshow(S, cmap=cmap, vmin=0, vmax=1)
ax[0].set_title('(a) quote $S_{\\ell k}=s_k(r^{(\\ell)})$')
Plog = np.maximum(P, 1e-16)
im1 = ax[1].imshow(-np.log10(Plog), cmap=cmap, vmin=0, vmax=16)
ax[1].set_title('(b) $-\\log_{10}$ p-value $P_{\\ell k}$')
for a, Mx, fmt in [(ax[0], S, '%.2f'), (ax[1], P, None)]:
    a.set_xticks(range(5)); a.set_xticklabels(labs5)
    a.set_yticks(range(5)); a.set_yticklabels(labs5)
    a.set_xlabel('blocco $k$'); a.set_ylabel('iniezione $\\ell$')
    for i in range(5):
        a.add_patch(plt.Rectangle((i-.5, i-.5), 1, 1, fill=False,
                                  edgecolor=ROS, lw=1.4))
        for j in range(5):
            v = Mx[i, j]
            if fmt:
                txt = '%.2f' % v
            elif v < 1e-12:
                txt = '$<\\!10^{-12}$'
            elif v < 1e-3:
                txt = '%.0e' % v
            else:
                txt = '%.3f' % v
            dark = (S[i, j] if fmt else min(-np.log10(max(v, 1e-16))/16, 1)) > 0.55
            a.text(j, i, txt, ha='center', va='center', fontsize=6.5,
                   color='white' if dark else 'black')
fig.tight_layout(w_pad=2.0)
save(fig, 'F3_confusione')
print('F3 done')

Ks = [1, 2, 4, 8]
agg_m = [R['levels'][str(k)] for k in Ks]
mic_m = [Rm['levels'][str(k)] for k in Ks]
fig, ax = plt.subplots(figsize=(4.6, 2.9))
x = np.arange(len(Ks)); w = 0.36
for off, src, lab, hatch in [(-w/2, agg_m, 'soli aggregati', ''),
                             (w/2, mic_m, 'aumentati', '//')]:
    h = [max(min(m['margin']), 1e-6) for m in src]
    col = [VER if m['ok_u'] else ROS for m in src]
    bars = ax.bar(x + off, h, w, color=col, hatch=hatch, alpha=0.85,
                  edgecolor='white', label=lab)
    for xi, m, hh in zip(x + off, src, h):
        ax.text(xi, hh * 1.25, 'OK' if m['ok_u'] else 'NO', ha='center',
                fontsize=7, color=VER if m['ok_u'] else ROS, weight='bold')
ax.set_yscale('log'); ax.set_ylim(5e-7, 4)
ax.set_xticks(x); ax.set_xticklabels(['$K=%d$' % k for k in Ks])
ax.set_ylabel('margine minimo di attribuzione')
ax.set_title('risoluzione della famiglia annidata per insieme dei momenti')
leg = [Patch(facecolor=GRI, alpha=.85, label='soli aggregati ($H=24$)'),
       Patch(facecolor=GRI, alpha=.85, hatch='//',
             label='aumentati ($H=24+12$)'),
       Patch(facecolor=VER, label='partizione risolta'),
       Patch(facecolor=ROS, label='attribuzione errata')]
legend_below(ax, ncol=2, handles=leg, gap=0.16)
fig.tight_layout()
save(fig, 'F4_risoluzione')
print('F4 done')
print('all figures in figs/')
