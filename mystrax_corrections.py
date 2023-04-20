import numpy as np
import flo_functions as ff






def calc_drift_velocity(info):
    
    dft_gate = info["dft_gate"]
    dft_cath = info["dft_cath"]
    dist_gate_cath = info["dist_gate_cath"]
    
    v_dft = dist_gate_cath / (dft_cath - dft_gate)
    
    return(v_dft)

def correct_drifttime(
    ds,
    info,
    tpc_info,
    id_bool,
):
    
    if not isinstance(id_bool, np.ndarray):
        id_bool = np.array([True] * len(ds))
    dft = ds[id_bool]["drifttime"]
    
    dft_gate = info["dft_gate"]
    dft_cath = info["dft_cath"]
    dist_gate_cath = info["dist_gate_cath"]
    dist_anode_gate = info["dist_anode_gate"]
    
    dft_c = (dft - dft_gate)
    
    v_dft = calc_drift_velocity(info)
    
    z = dft_c*v_dft*(dft_c>0) + dft_c*(dft_c<0)
    
    ds["drifttime_corrected"][id_bool] = dft_c
    ds["z"][id_bool] = z
    
# S2 correction function

def correct_S2(
    ds,
    info,
    tpc_info,
    id_bool,
    
):
    if not isinstance(id_bool, np.ndarray):
        id_bool = np.array([True] * len(ds))
    
    lifetime = info["electron_lifetime"]
    
    # print(f"elt: {lifetime:.1f} µs")
    
    dft_c = ds[id_bool]["drifttime_corrected"]
    # if not np.any(dft_c > 0):
        # raise ValueError("S2 correction requires corrected drifttime")
        
    
    id_bool = id_bool & (ds["drifttime_corrected"] > 0)
    dft_c = ds[id_bool]["drifttime_corrected"]
    exp_dft_over_lft = np.exp(dft_c/ lifetime)
    
    
    for field_i in [2, 3, 5, 7]:
        s2 = ds[id_bool]["areas"][:, field_i]
        cs2 = s2 * exp_dft_over_lft
    
        ds["areas_corrected"][id_bool, field_i] = cs2
    
    
    
# S1 correction function
    
    
def correct_S1(
    ds,
    info,
    tpc_info,
    id_bool,
):
    if not isinstance(id_bool, np.ndarray):
        id_bool = np.array([True] * len(ds))
    id_bool = id_bool & (ds["drifttime_corrected"] >= 0)
    
    
    
    
    dft_correct_for = (tpc_info["dft_cath"]-tpc_info["dft_gate"])/2
    
    dftc = ds[id_bool]["drifttime_corrected"]
    # if not np.any(dftc > 0):
        # raise ValueError("S1 correction requires corrected drifttime")
    
    f = ff.poly_1
    
    
    
    for field_i,info_i in zip([0, 1], info):
        s1 = ds[id_bool]["areas"][:, field_i]
        m, c = info_i["m"], info_i["c"]
        cs1 = s1 * f(dft_correct_for, m = m, c = c)  / f(dftc, m = m, c = c)

        ds["areas_corrected"][id_bool, field_i] = cs1
        
    ds["areas_corrected"][id_bool, 6] = ds["areas_corrected"][id_bool, 0] + ds["areas_corrected"][id_bool, 1]

    
    
    
    
    
# Summary of all functions
    
corrections = {
    "gate_cathode": correct_drifttime,
    "electron_lifetime": correct_S2,
    "S1_correction": correct_S1,
}
