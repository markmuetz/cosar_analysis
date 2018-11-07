from omnium import AnalysisSettings


production_settings = AnalysisSettings(dict(
    # 23.75N - 23.75S.
    TROPICS_SLICE = slice(53, 92),
    # 23.75N - 0N
    NH_TROPICS_SLICE = slice(73, 92),
    # 0N - 23.75S
    SH_TROPICS_SLICE = slice(53, 72),
    # Random seeds to use in kmeans alg.
    RANDOM_SEEDS = [391137, 725164,  12042, 707637, 106586],
    # n_clusters to analyse.
    CLUSTERS = list(range(5, 21)),
    # Cluster to analyse in detail.
    DETAILED_CLUSTER = 10,
    # n PCA components to use, can be set manually. 
    # If None, will be set by EXPL_VAR_MIN.
    N_PCA_COMPONENTS = None,
    # Minimum explained variance for PCA.
    EXPL_VAR_MIN = 0.9,
    # Threshold settings to use. CAPE threshold in J/kg.
    CAPE_THRESH = 100,
    SHEAR_PRESS_THRESH_HPA = 500,
    SHEAR_PERCENTILE = 75,
    # Description of input data.
    NUM_PRESSURE_LEVELS=20,
    INDEX_850HPA=-4,
    # Favour lower trop settings.
    FAVOUR_LOWER_TROP=True,
    FAVOUR_FACTOR=4,
    # where to apply favouring to.
    FAVOUR_INDEX=10,
    # which filters to apply.
    FILTERS=('cape', 'shear'),
    # N.B. using other than 'tropics' untested.
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

# These are the old settings - with tropics as 30N - 30S.
production_30_tropics = production_settings.copy()
production_30_tropics.set('TROPICS_SLICE', slice(48, 97))

# No favour lower troposphere.
production_no_favour_lower = production_settings.copy()
production_no_favour_lower.set('FAVOUR_LOWER_TROP', False)

# Reduced number of clusters for test.
test_settings = production_settings.copy()
test_settings.set('CLUSTERS', [5, 10, 15, 20])
