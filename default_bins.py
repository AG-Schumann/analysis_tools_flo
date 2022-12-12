import numpy as np



def make_bins(x0, x1, bw):
    '''    
    returns bins such that the first bin center is x0
    (x0 is NOT the left side of the first bin!!!)
    '''
    return(np.arange(x0-bw/2, x1+bw, bw))






tpc_corrections = (3.40, 42.24, 69, 5, 40)


default_bins = {}
default_bw = {}


# default variables
position_gate = tpc_corrections[0]
position_cathode = tpc_corrections[1]
S1_correction_window = tpc_corrections[-2:]

sigma_position_gate = .2
sigma_position_cathode = .3


# bins for cathode search
default_bw["search_gate"] = .5
default_bw["search_cath"] = 1
default_bins["search_gate"] = make_bins( 1.5, 10, default_bw["search_gate"])
default_bins["search_cath"] = make_bins(32,   52, default_bw["search_gate"])

default_bins["decay_time"] = make_bins(0, 2500, 100)




bw_drift = 2
bw_drift_aa = 1 # aa = above anode

# this allows us to have the bin boundries exactly at the s1 correction window border
bw_s1_dt = np.diff(tpc_corrections[-2:])/20

bw_s1_area = 10
bw_s2_area = 200

bins_drift = np.arange(-bw_drift/2, 60+bw_drift/2, bw_drift)
default_bins["cdt"] = np.arange(0, 42, 2)
default_bins["cdt_extended"] = np.arange(-10, 62, 2)

default_bins["drifttime_all"] = bins_drift
default_bins["drifttime_below_anode"] = bins_drift[bins_drift > position_gate]
default_bins["drifttime_above_anode"] = np.arange(-bw_drift_aa/2, position_gate+bw_drift_aa*2/3, bw_drift_aa)
default_bins["drifttime_fine"] = np.arange(-bw_drift_aa/2, 40+bw_drift_aa*2/3, bw_drift_aa)

default_bins["drifttime"] = np.arange(position_gate+1, position_cathode-1, 2)
default_bins["area_S2"] = np.arange(-bw_s2_area/2, 8000+bw_s2_area*3/2, bw_s2_area)
default_bins["area_S1"] = np.arange(-bw_s1_area/2, 400+bw_s1_area*3/2, bw_s1_area)


default_bins["full_range_s1"] = np.arange(min(S1_correction_window) - 2 * bw_s1_dt, max(S1_correction_window) + 2 * bw_s1_dt, bw_s1_dt)
default_bins["max_range_s1"] = np.arange(min(S1_correction_window) - 20 * bw_s1_dt, max(S1_correction_window) + 50 * bw_s1_dt, bw_s1_dt)