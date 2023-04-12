import numpy as np
import nestpy
from flo_fancy import *


nc = nestpy.NESTcalc(nestpy.VDetector())







def get_quanta_for_E_dft(
    decay = 0,
    drift_field = 500.,
    deltaT_ns = 1500.,
    E = False,
    verbose = False,
    **kwargs
):
    '''
    returns the number of quanta as tuple
    
    
    parameters:
      *  specify either
          - decay: (0: total, 1: first decay, 2: second decay)
          or E to pass the Energy of the decay
      * deltaT_ns: decay time of decays (plural!)
            This wrapper-fuction also accepts numpy arrays for this value!
    
    '''
    # WEIRD VALUES IF WE USE PRECISE VALUES!!
#     Es = [41.5569, 32.1516, 9.4053]
    Es =   [41.5,    32.1,    9.4]
    if not isinstance(E, bool) and isinstance(E, (float, int)):
        qp(f"E given: {E} keV", end = "\n", verbose = verbose)
    elif not isinstance(decay, bool) and decay in [0,1,2]:
        E = Es[decay]
        qp(f"decay given: {decay} => {E} keV", end = "\n", verbose = verbose)
    
    else:
        raise ValueError("E and decay not specified properly")
    
    
    if isinstance(deltaT_ns, (float, int)):
        qp(f"deltaT_ns: {deltaT_ns:.1f} ns", end = "\n", verbose = verbose)
        deltaT_ns = np.array([deltaT_ns])

    else:
        qp(f"deltaT_ns: {len(deltaT_ns)} entries", end = "\n", verbose = verbose)
        
    qp(f"drift_field: {drift_field:.1f} V/cm", end = "\n", verbose = verbose)
    qp(f"Energy: {E:.2f} keV", end = "\n", verbose = verbose)
    
    yields = [
        nc.GetYieldKr83m(
            energy = E,
            drift_field = drift_field,
            deltaT_ns = deltaT_ns_i,
            **kwargs
        ) for deltaT_ns_i in deltaT_ns
    ]
    
    results = np.array([
        np.array([y.PhotonYield for y in yields]),
        np.array([y.ElectronYield for y in yields]),
    ])
    
    return(results)


def get_all_quanta_for_dct(deltaT_ns, for_straxbra = True, **kwargs):
    '''
    returns the number of quanta for all six/eight signal types (S1/S2: split / unsplit / total)
    has same structure as straxbra areas field
    
    leave parameter for_straxbra at True to get an array that krs["areas_corrected"] can be divided by
    
    
    see get_quanta_for_E_dft for (default) parameters
    
    '''


    out = [-1]*8
    quantas = [
        get_quanta_for_E_dft(
            decay = decay,
            deltaT_ns = deltaT_ns,
            **kwargs)
        for decay in [1, 2, 0]
    ]
    
    # use this structure to be compatible with _summary plugins of straxbra
    
    # S1 signals
    out[0] = quantas[0][0]
    out[1] = quantas[1][0]
    # S2 signals
    out[2] = quantas[0][1]
    out[3] = quantas[1][1]
    # unsplit Signals (S1 and S2)
    out[4] = quantas[2][0]
    out[5] = quantas[2][1]
    # total signals (S1 and S2)
    out[6] = quantas[2][0]
    out[7] = quantas[2][1]
    
    
    if for_straxbra is True:
        out = np.array(out).T
    else:
        _ = out.pop(4)
        _ = out.pop(4)
        
    
    return(out)







