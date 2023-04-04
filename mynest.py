import numpy as np
import nestpy
from flo_fancy import *


nc = nestpy.NESTcalc(nestpy.VDetector())



def get_quanta_for_E_dft(
    decay = 0,
    drift_field = 500,
    deltaT_ns = 1500,
    E = False,
    verbose = False,
    **kwargs
):
    '''
    returns the number of quanta as tuple
    
    
    parameters:
        specify either
          - decay: (0: total, 1: first decay, 2: second decay)
          or E to pass the Energy of the decay
          
    
    '''
    Es = [41.5569, 32.1516, 9.4053]
    if decay in [0,1,2]:
        E = Es[decay]
        qp(f"decay given: {decay} => {E} keV", end = "\n", verbose = verbose)
    elif not isinstance(E, bool):
        qp(f"E given: {E} keV", end = "\n", verbose = verbose)
    else:
        raise ValueError("E and decay not specified properly")
    
    
    yields = nc.GetYieldKr83m(
        energy = E,
        drift_field = drift_field,
        deltaT_ns = deltaT_ns,
        **kwargs
    )
    qp(f"""    Energy:      {E} keV
    drift-field: {drift_field} V/cm
    deltaT_ns:   {deltaT_ns} ns
Results:
                n_photons:   {yields.PhotonYield:.1f}
                n_electrons: {yields.ElectronYield:.1f}
""", verbose = verbose)
    
    return(yields.PhotonYield, yields.ElectronYield)