import numpy as np
import pandas as pd




fit_pars = pd.DataFrame({
    "label": [f"Θ{i}" for i in range(5)],
    "label_tex": [f"\\Theta_{i}" for i in range(5)],
    "unit": ["1/(µm e)", "kV/cm", "kV/cm", "PE/(kV/cm µm)", "kV/cm"],
    "desc": ["charge gain factor", "slope in charge gain", "threshold of charge mult.", "S2 gain factor", "threshold of S2"],
    "fit":  [0.8, 242, 725, 16.6, 412],
    "sfit": [.1, 45, 48, 1.1, 412],
})

fit_10 = np.array([1.15, 561, 586, 13.3, 399])
sfit_10 = np.array([0.15, 119, 47, 0.4, 7])


len_desc = max([len(x) for x in fit_pars["desc"].values])


pars_calc_g = dict(
    E = 5.41e6,
    W = 15.6,
    f_ion = 4.15e-2,
    e_LC = 0.212,
    e_Q = 0.323,
    f_LXE = 1.07,
    e_dy = 0.75,
)


def calc_N_e__g_conversion(
    E = pars_calc_g["E"],
    W = pars_calc_g["W"],
    f_ion = pars_calc_g["f_ion"],
    e_LC = pars_calc_g["e_LC"],
    e_Q = pars_calc_g["e_Q"],
    f_LXE = pars_calc_g["f_LXE"],
    e_dy = pars_calc_g["e_dy"],
):
    return(
        E/W,
        W/(E * f_ion * e_LC * e_Q * f_LXE * e_dy)
    )


N_e, g = calc_N_e__g_conversion()


pars_calc_g__flo = dict(E = 42e3, W = 13.3, e_LC = .14)
N_e__flo, g__flo = calc_N_e__g_conversion(**pars_calc_g__flo)


def E_r_calc(r, V_A, d_w, V_surface_1kV = 243.6):
    # 2 pi is already in E_lookup
    E = V_surface_1kV*V_A*(d_w/2) / r * (r>=(d_w/2))
    return(E)

def calc_r_max(V_A, d_w = 10, V_surface_1kV = 243.6, treshold = 300):
    return(
        V_surface_1kV*V_A*(d_w/2) / treshold
    )


def dNe_calc(Ne, E, dr, p0, p1, p2):
    # added the (E >= p2) as there is no amplification if E < p2
    if (E > p2):
        exp = np.exp(-p1/(E-p2))
    else:
        return(0)
    

    dNe = Ne * p0 * exp * dr
    dNe * (dNe > 0)
    return(dNe)
                           
def dNg_calc(Ne, E, dr, p3, p4):
    if E > p4:
        dNg = Ne * p3 * (E - p4) * dr
    else:
        return(0)
    return(dNg)





# The actual simulation
def calc_one_voltage(dV, p0, p1, p2, p3, p4, Ne0 = 1, dr = .05, d_w = 10, V_surface_1kV = 243.6, r_max = False, treshold = 300):
    '''
    calculates everything for one voltage
    use return_value["PE"] to get _PE
    
    params:
        dV:
            anode voltage [kV]
        p0 -p4
            see fit_description
        for constant number of electrons: set p0 to 0
        Ne0:
            number of initial electrons
        dr (0.1 µm):
            stepsize [µm]
        d_w (10 µm):
            wire diameter [µm]
        V_surface_1kV (243.6 kV):
            field strength of wire at surface (in [kV/cm])
        r_max (False):
            maximum distance of wire to calculate field [µm]
            if set to False: use calc_r_max(...)
        threshold (300 kV/cm):
            threshold for r_max caclulation if r_max is False
    '''
    
    if r_max is False:
        r_max = calc_r_max(dV, d_w = 10, V_surface_1kV= V_surface_1kV, treshold = treshold)
        r_max = np.round(r_max/dr)*dr
    r_sim = np.arange(r_max, d_w/2, -dr)

    
    E_sim = E_r_calc(r_sim, dV, d_w)
    
    Ne_sim = np.zeros(len(E_sim))
    Ng_sim = np.zeros(len(E_sim))
    dNe_sim = np.zeros(len(E_sim))
    dNg_sim = np.zeros(len(E_sim))
    
    
    Ne = Ne0*1
    Ng = 0
    for i_sim, (r, E) in enumerate(zip(r_sim, E_sim)):
        
        dNe = dNe_calc(Ne, E, dr, p0, p1, p2)
        Ne += dNe

        dNg = dNg_calc(Ne, E, dr, p3, p4)
        Ng += dNg
        
        
        dNe_sim[i_sim] = dNe
        Ne_sim[i_sim] = Ne
        Ng_sim[i_sim] = Ng
        dNg_sim[i_sim] = dNg

    return({
        "N_e": Ne,
        "PE": Ng,
        "r_sim": r_sim,
        "E_sim": E_sim,
        "N_e_sim": Ne_sim,
        "N_g_sim": Ng_sim,
        "dN_e_sim": dNe_sim,
        "dN_g_sim": dNg_sim,
    })



def PE_factory(Ne0 = 1, dr = .05, d_w = 10, r_max = False):
    '''
    creates a function that requires
        dVs, p0, p1, p2, p3, p4
    Returns only PE as an array for the voltages
    
    default parameters:
        Ne0 = 1, dr = .1, d_w = 10, r_max = False
        are used in calc_one_voltage(...)
        
    use this when you  want to fit the entire function
    '''
    
    def f(dVs, p0, p1, p2, p3, p4):
        return(np.array([
            calc_one_voltage(
                # paramters to fit
                dV, p0, p1, p2, p3, p4, 
                # constant parameters
                Ne0 = Ne0, dr = dr, d_w = 10, r_max = r_max,
            )["PE"]
            for dV in dVs
        ]))

    return(f)

def PE_factory_constant_Ne(Ne0 = 1, dr = .05, d_w = 10, r_max = False):
    '''
    creates a function assumes a constant number of electrons and equires only
        dVs, p3, p4
    Returns only PE as an array for the voltages
    
    
    default parameters:
        Ne0 = 1, dr = .1, d_w = 10, r_max = False
        are used in calc_one_voltage(...)
        
    use this when you  want to fit only the photn creation part and assume no change in the number of electrons
    '''
    
    def f(dVs, p3, p4):
        return(np.array([
            calc_one_voltage(
                # paramters to fit
                dV, p3 = p3, p4 = p4, 
                p0 = 0, p1 = 1, p2 = 1,
                Ne0 = Ne0, dr = dr, d_w = 10, r_max = r_max
            )["PE"]
            for dV in dVs
        ]))
    
    return(f)
