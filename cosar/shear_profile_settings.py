from omnium.analyser_setting import AnalyserSetting
import cosar

full_settings = AnalyserSetting(cosar, dict(
    TROPICS_SLICE = slice(48, 97),
    NH_TROPICS_SLICE = slice(73, 97),
    SH_TROPICS_SLICE = slice(48, 72),
    USE_SEEDS = True,
    RANDOM_SEEDS = [391137, 725164,  12042, 707637, 106586],
    # RANDOM_SEEDS = [391137],
    CLUSTERS = list(range(5, 21)),
    # CLUSTERS = [11],
    # CLUSTERS = [5, 10, 15, 20]
    # CLUSTERS = [11],
    # CLUSTERS = [5, 10, 15, 20]
    DETAILED_CLUSTER = 11,
    N_PCA_COMPONENTS = None,
    EXPL_VAR_MIN = 0.9,
    CAPE_THRESH = 100,
    SHEAR_PRESS_THRESH_HPA = 500,
    SHEAR_PERCENTILE = 75,
    INTERACTIVE = False,
    PLOT_EGU_FIGS = False,
    NUM_EGU_SAMPLES = 10000,
    NUM_PRESSURE_LEVELS=20,
    INDEX_850HPA=-4,
    FAVOUR_LOWER_TROP=True,
))
