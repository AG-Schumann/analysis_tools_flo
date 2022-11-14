import numpy as np

# default variables
position_gate = 2.95 
position_cathode = 42.06
S1_correction_window = (4, 40)


sigma_position_gate = .2
sigma_position_cathode = .3



bw_drift = 2
bw_drift_aa = 1 # aa = above anode
bw_s1_dt = 2
bw_s1_area = 20
bw_s2_area = 200

bins_drift = np.arange(-bw_drift/2, 60+bw_drift/2, bw_drift)
bins_drift = np.arange(-bw_drift/2, 60+bw_drift/2, bw_drift)


default_bins = {
    "drifttime_all": bins_drift,
}
default_bins["drifttime_below_anode"] = bins_drift[bins_drift > position_gate]
default_bins["drifttime_above_anode"] = np.arange(-bw_drift_aa/2, position_gate+bw_drift_aa*2/3, bw_drift_aa)
default_bins["drifttime_fine"] = np.arange(-bw_drift_aa/2, 40+bw_drift_aa*2/3, bw_drift_aa)

default_bins["drifttime"] = np.arange(position_gate+1, position_cathode-1, 2)
default_bins["full_range_s1"] = np.arange(-20*bw_s1_dt/2, 100+2/3*bw_s1_dt, bw_s1_dt)
default_bins["area_S2"] = np.arange(-bw_s2_area/2, 8000+bw_s2_area*3/2, bw_s2_area)
default_bins["area_S1"] = np.arange(-bw_s1_area/2, 1000+bw_s1_area*3/2, bw_s1_area)
