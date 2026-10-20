"""Shared ALICE plotting settings for primary-pruning QA."""
def apply_style():
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import mplhep as hep
    plt.style.use(hep.styles.ALICE)
    for key in mpl.rcParams.keys():
        if key.startswith('legend.'):
            mpl.rcParams[key] = mpl.rcParamsDefault[key]
    plt.rcParams.update({'figure.max_open_warning': 0})
    plt.rcParams.update({'figure.figsize': (16, 10), 'font.size': 20,
                         'axes.labelsize': 20, 'xtick.labelsize': 20,
                         'ytick.labelsize': 20, 'legend.fontsize': 20,
                         'legend.title_fontsize': 20, 'lines.linewidth': 2.5,
                         'patch.linewidth': 2.5, 'axes.linewidth': 1.5})


def cutoff_colors(count):
    import matplotlib.pyplot as plt
    import numpy as np
    return plt.get_cmap('viridis')(np.linspace(.08, .92, count))

LOSS_TARGETS = (0.001, 0.005, 0.01, 0.03, 0.05)
