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
        id_bool = np.ones(len(ds))
    dft = ds[id_bool]["drifttime"]
    
    dft_gate = info["dft_gate"]
    dft_cath = info["dft_cath"]
    dist_gate_cath = info["dist_gate_cath"]
    dist_anode_gate = info["dist_anode_gate"]
    
    dft_c = (dft - dft_gate)
    
    v_dft = calc_drift_velocity(info)
    
    z = dft_c*v_dft*(dft_c>0) + dft_c*(dft_c<0)
    
    ds["z"][id_bool] = z