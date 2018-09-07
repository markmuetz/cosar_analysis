from omnium import AnalysisSettings


production_settings = AnalysisSettings(dict(
    TROPICS_SLICE = slice(48, 97),
    NH_TROPICS_SLICE = slice(73, 97),
    SH_TROPICS_SLICE = slice(48, 72),
    USE_SEEDS = True,
    RANDOM_SEEDS = [391137, 725164,  12042, 707637, 106586],
    CLUSTERS = list(range(5, 21)),
    DETAILED_CLUSTER = 10,
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
    FAVOUR_FACTOR=4,
    FAVOUR_INDEX=10,
    FILTERS=('cape', 'shear'),
    LOC='tropics',
))

# Various sensitivity tests on CAPE_THRESH, SHEAR_PERCENTILE and FAVOUR_FACTOR.
# Can be run with e.g. omnium run -s production_higher_CAPE ...
production_higher_CAPE = production_settings.copy()
production_higher_CAPE.set('CAPE_THRESH', 125)
production_lower_CAPE = production_settings.copy()
production_lower_CAPE.set('CAPE_THRESH', 75)

production_higher_shear = production_settings.copy()
production_higher_shear.set('SHEAR_PERCENTILE', 85)
production_lower_shear = production_settings.copy()
production_lower_shear.set('SHEAR_PERCENTILE', 65)

production_higher_favour = production_settings.copy()
production_higher_favour.set('FAVOUR_FACTOR', 5)
production_lower_favour = production_settings.copy()
production_lower_favour.set('FAVOUR_FACTOR', 3)

# Actually 23.75N - 23.75S.
production_23_tropics = production_settings.copy()
production_23_tropics.set('TROPICS_SLICE', slice(53, 92))

test_settings = AnalysisSettings(dict(
    TROPICS_SLICE = slice(48, 97),
    NH_TROPICS_SLICE = slice(73, 97),
    SH_TROPICS_SLICE = slice(48, 72),
    USE_SEEDS = True,
    RANDOM_SEEDS = [391137],
    CLUSTERS = [5, 10, 15, 20],
    DETAILED_CLUSTER = 10,
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
    FILTERS=('cape', 'shear'),
    LOC='tropics',
))
