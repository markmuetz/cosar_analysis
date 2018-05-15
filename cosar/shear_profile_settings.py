from omnium.analyser_setting import AnalyserSetting

full_settings = AnalyserSetting(dict(
    TROPICS_SLICE = slice(48, 97),
    NH_TROPICS_SLICE = slice(73, 97),
    SH_TROPICS_SLICE = slice(48, 72),
    USE_SEEDS = True,
    RANDOM_SEEDS = [391137, 725164,  12042, 707637, 106586],
    # RANDOM_SEEDS = [391137],
    # CLUSTERS = list(range(5, 21)),
    CLUSTERS = [10],
    # CLUSTERS = [5, 10, 15, 20]
    # CLUSTERS = [11],
    # CLUSTERS = [5, 10, 15, 20]
    DETAILED_CLUSTER = 11,
    N_PCA_COMPONENTS = None,
    EXPL_VAR_MIN = 0.9,
    CAPE_THRESH = 100,
    SHEAR_PERCENTILE = 75,
    INTERACTIVE = False,
    FIGDIR = 'fig',
    PLOT_EGU_FIGS = False,
    NUM_EGU_SAMPLES = 10000,
))
