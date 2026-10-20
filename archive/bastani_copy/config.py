"""
Calibration configs for the synthetic refugee-matching environment.

US  = replication of Bastani, Bastani & McLaughlin (2026), Appendix EC.2
      (marginals taken from Bansak et al. 2018 SM).
DE  = German setting (16 Länder, Königsteiner quotas, IAB-BAMF-SOEP 2016 marginals).

Every value marked ASSUMED is not taken from a source and must be checked /
replaced (ideally from the descriptive table of the Strasser Ceballos & Kern SOEP sample).
"""
import numpy as np

# --------------------------------------------------------------------------
# US setting (Bastani et al. 2026, Table EC.1 / Fig. EC.1)
# --------------------------------------------------------------------------
US_ORIGINS = ["Burma", "Iraq", "Bhutan", "Somalia", "Afghanistan",
              "DRC", "Iran", "Eritrea", "Ukraine", "Syria",
              "Sudan", "Ethiopia", "Moldova", "Other"]
_us_origin_p = np.array([0.23, 0.20, 0.13, 0.11, 0.07] + [26 / 900] * 9)

US = dict(
    name="US",
    n_locations=43,
    loc_p=None,  
    loc_p_range=(0.006, 0.062),
    loc_names=None,
    age_bins=[(18, 30, 0.44), (30, 40, 0.28), (40, 50, 0.16), (50, 60, 0.12)],
    p_male=0.53,
    education=dict(levels=["none", "little", "secondary", "advanced", "university"],
                   p=[0.18, 0.39, 0.21, 0.10, 0.12]),
    p_language=0.43,            # speaks English
    language_name="english",
    p_free_case=0.28,
    origins=US_ORIGINS,
    origin_p=_us_origin_p / _us_origin_p.sum(),
    years=[2011, 2012, 2013, 2014, 2015, 2016],
    year_p=[4 / 23] * 5 + [3 / 23],
    month_p=[6 / 69] * 9 + [5 / 69] * 3,
    # outcome calibration
    target_rate_all=0.23, 
    target_rate_free=0.34,     
    loc_effect="beta12",      
    test_only_free=True,
    n_train=33000,
    n_test=1000,
)

# --------------------------------------------------------------------------
# German setting
# --------------------------------------------------------------------------
DE_LAENDER = ["NRW", "Bayern", "BW", "Niedersachsen", "Hessen", "Berlin",
              "Sachsen", "RLP", "SH", "Brandenburg", "Sachsen-Anhalt",
              "Hamburg", "Thueringen", "MV", "Saarland", "Bremen"]
# BAMF, Erstverteilung (EASY quotas, Königsteiner Schlüssel), accessed 2026
DE_QUOTA = np.array([20.96090, 15.68563, 13.12509, 9.40263, 7.40812, 5.20075,
                     4.94788, 4.80876, 3.39588, 3.01025, 2.69612,
                     2.63402, 2.61509, 1.96964, 1.18749, 0.95455]) / 100

DE = dict(
    name="DE",
    n_locations=16,
    loc_p=DE_QUOTA / DE_QUOTA.sum(),
    loc_names=DE_LAENDER,

    age_bins=[(18, 30, 0.58), (30, 40, 0.24), (40, 50, 0.12), (50, 64, 0.06)],
    p_male=0.73,               # "almost three quarters" of adults are men

    education=dict(levels=["none", "primary", "middle", "upper_secondary", "university"],
                   p=[0.09, 0.10, 0.28, 0.40, 0.13]),
    p_language=0.16,           # speaks German well / very well (2016)
    language_name="german",
    p_free_case=None,         

    origins=["Syria", "Afghanistan", "Iraq", "Other"],
    origin_p=np.array([0.415, 0.098, 0.10, 0.387]),
  
    years=[2013, 2014, 2015, 2016, 2017, 2018],
    year_p=[0.05, 0.10, 0.40, 0.35, 0.06, 0.04],
    month_p=[1 / 12] * 12,
   
    target_rate_all=0.15,
    target_rate_free=None,
    
    loc_effect="proportional",
    loc_rel_spread=(0.74, 1.00),
    test_only_free=False,
    n_train=33000,
    n_test=1000,
)


def build_config(cfg):
    cfg = dict(cfg)
    if cfg["loc_p"] is None:
        lo, hi = cfg["loc_p_range"]
        p = np.geomspace(lo, hi, cfg["n_locations"])
        cfg["loc_p"] = p / p.sum()
    if cfg["loc_names"] is None:
        cfg["loc_names"] = [f"L{i:02d}" for i in range(cfg["n_locations"])]
    return cfg


CONFIGS = {"US": US, "DE": DE}
