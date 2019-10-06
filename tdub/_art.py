import matplotlib

matplotlib.use("pdf")

def setup_style():
    matplotlib.rcParams["axes.labelsize"] = 14
    matplotlib.rcParams["font.size"] = 12
    matplotlib.rcParams["xtick.top"] = True
    matplotlib.rcParams["ytick.right"] = True
    matplotlib.rcParams["xtick.direction"] = "in"
    matplotlib.rcParams["ytick.direction"] = "in"
    matplotlib.rcParams["xtick.labelsize"] = 12
    matplotlib.rcParams["ytick.labelsize"] = 12
    matplotlib.rcParams["xtick.minor.visible"] = True
    matplotlib.rcParams["ytick.minor.visible"] = True
    matplotlib.rcParams["xtick.major.width"] = 0.8
    matplotlib.rcParams["xtick.minor.width"] = 0.8
    matplotlib.rcParams["xtick.major.size"] = 7.0
    matplotlib.rcParams["xtick.minor.size"] = 4.0
    matplotlib.rcParams["xtick.major.pad"] = 1.5
    matplotlib.rcParams["xtick.minor.pad"] = 1.4
    matplotlib.rcParams["ytick.major.width"] = 0.8
    matplotlib.rcParams["ytick.minor.width"] = 0.8
    matplotlib.rcParams["ytick.major.size"] = 7.0
    matplotlib.rcParams["ytick.minor.size"] = 4.0
    matplotlib.rcParams["ytick.major.pad"] = 1.5
    matplotlib.rcParams["ytick.minor.pad"] = 1.4
    matplotlib.rcParams["legend.frameon"] = False
    matplotlib.rcParams["legend.numpoints"] = 1
    matplotlib.rcParams["legend.fontsize"] = 11
    matplotlib.rcParams["legend.handlelength"] = 1.5
    matplotlib.rcParams["axes.formatter.limits"] = [-4, 4]
    matplotlib.rcParams["axes.formatter.use_mathtext"] = True
