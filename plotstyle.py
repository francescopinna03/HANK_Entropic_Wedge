# Common style for all figures of the repository.
# Text and mathematics share the same serif font, titles have one size and
# weight, legends sit below the axes so that no curve can cross them, and PDF
# files carry no creation date, so that a replication reproduces them byte
# for byte.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLU, GRI, ROS, VER = '#1f4e79', '#7f7f7f', '#a63d40', '#2e6e4e'

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif',
    'font.size': 9, 'axes.titlesize': 9, 'axes.titleweight': 'normal',
    'axes.titlepad': 6, 'axes.labelsize': 9, 'legend.fontsize': 7.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 150, 'axes.linewidth': 0.7, 'lines.linewidth': 1.4,
    'pdf.fonttype': 42})


def legend_below(ax, ncol, handles=None, gap=0.30):
    # legend centred under the axes, outside the plotting area
    kw = dict(loc='upper center', bbox_to_anchor=(0.5, -gap), ncol=ncol,
              frameon=False, handlelength=2.2, columnspacing=1.2)
    if handles is not None:
        kw['handles'] = handles
    return ax.legend(**kw)


def save(fig, stem):
    # PDF without creation date (deterministic) and PNG for quick viewing
    fig.savefig('figs/%s.pdf' % stem, bbox_inches='tight',
                metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig('figs/%s.png' % stem, bbox_inches='tight')
    plt.close(fig)
