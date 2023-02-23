import numpy as np







def calc_drift_velocity(info):
    
    dft_gate = info["dft_gate"]
    dft_cath = info["dft_cath"]
    dist_gate_cath = info["dist_gate_cath"]
    
    v_dft = dist_gate_cath / (dft_cath - dft_gate)
    
    return(v_dft)

def correct_drifttime(
    ds,
    info,
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
    
    

def correct_S2(
    ds,
    info,
    id_bool,
):
    if not isinstance(id_bool, np.ndarray):
        id_bool = np.array([True] * len(ds))
    
    lifetime = info["electron_lifetime"]
    
    
    id_bool = id_bool & (ds["drifttime_corrected"] > 0)
    exp_dft_over_lft = np.exp(ds[id_bool]["drifttime_corrected"]/ lifetime)
    
    
    for field_i in [2, 3, 5, 7]:
        s2 = ds["areas"][id_bool, field_i]
        cs2 = s2 * exp_dft_over_lft
    
        ds["areas_corrected"][id_bool, field_i] = cs2
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# Summary of all functions
    
corrections = {
    "gate_cathode": correct_drifttime,
    "electron_lifetime": correct_S2,
}
