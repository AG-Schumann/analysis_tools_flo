import numpy as np
import pandas as pd



fit_10 = np.array([1.15, 561, 586, 13.3, 399])
sfit_10 = np.array([0.15, 119, 47, 0.4, 7])

fit_10_el = fit_10 * 1
sfit_10_el = sfit_10 * 1
fit_10_el[3] = 1.68e-2
sfit_10_el[3] = 0.51e-2




fit_5_10 = np.array([0.8, 242, 725, 16.6, 412])
sfit_5_10 = np.array([.1, 45, 48, 1.1, 10])




fit_pars = pd.DataFrame({
    "label": [f"Θ{i}" for i in range(5)],
    "label_tex": [f"\\Theta_{i}" for i in range(5)],
    "unit": ["1/(µm e)", "kV/cm", "kV/cm", "PE/(kV/cm µm)", "kV/cm"],
    "desc": ["charge gain factor", "slope in charge gain", "threshold of charge mult.", "S2 gain factor", "threshold of S2"],
    "fit":  fit_10,
    "sfit": sfit_10,
    "fit_el_gain": fit_10_el,
    "sfit_el_gain": sfit_10_el,
    
})



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
        W/(E * f_ion * e_LC * e_Q * f_LXE * e_dy)
    )


N_e = 14000 # from paper, sec 2.2
g = calc_N_e__g_conversion()


pars_calc_g__flo = dict(E = 42e3, W = 13.3, e_LC = .14)
N_e__flo = 1000
g__flo = calc_N_e__g_conversion(**pars_calc_g__flo)


def E_r_calc(r, V_A, d_w, V_surface_1kV = 243.6):
    # 2 pi is already in E_lookup
    E = V_surface_1kV*V_A*(d_w/2) / r * (r>=(d_w/2))
    return(E)

def calc_r_max(V_A, d_w = 10, V_surface_1kV = 243.6, treshold = 300):
    return(
        V_surface_1kV*V_A*(d_w/2) / treshold
    )



# The iterative functions are here
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
    
    
    
    
def calc_shadowing_correction_factor(V_A, N_steps = 10000, d_w=10, V_surface_1kV=243.6, treshold=400, return_all = False):
    '''
    Gives the mean shadowing correction factor for anode voltages.
    '''
    if isinstance(V_A, (np.ndarray, list, tuple)):
        return(
            np.array([
                calc_shadowing_correction_factor(V_Ai, N_steps = 10000, d_w=10, V_surface_1kV=243.6, treshold=400, return_all = return_all)
                for V_Ai in V_A
            ])
        )
    
    r_max = calc_r_max(V_A=V_A, d_w=d_w, V_surface_1kV=V_surface_1kV, treshold=treshold)
    r_range = np.linspace(r_max, d_w/2, N_steps)

    shaddowing = 1/(1-(np.arcsin(d_w/2/r_range)/np.pi))
    mean_shadowing = np.mean(shaddowing)
    if return_all is True:
        return({
            "mean_factor": mean_shadowing,
            "all_factors": shaddowing,
            "r_range": r_range,
            "r_max": r_max,
            "V_A": V_A,
            "N_steps": N_steps,
            "d_w": d_w,
            "V_surface_1kV": V_surface_1kV,
            "treshold": treshold,
        })
    
    return(mean_shadowing)




def calc_el_gain_band(dVs, Ne0 = 1, dr = .05, d_w = 10, r_max = False):
    '''
    calculates for a given list of V_anode:
        the el-gain and the band of uncertaintie of elenas fit paramters
        ignores the thresholds as increase the bands as they correlate with the gains
          but we have no covariance matrix
    '''
    
    pars = fit_pars["fit_el_gain"].values*1
    s_pars = fit_pars["sfit_el_gain"].values*1
    
    parset = dict(
        p0 = pars[0],
        p1 = pars[1],
        p2 = pars[2],
        p3 = pars[3],
        p4 = pars[4],
    )
    s_pars = dict(
        p0 = s_pars[0],
        p1 = s_pars[1],
        p2 = pars[2],
        p3 = s_pars[3],
        p4 = pars[4],
    )
    
    
    def f(dVs, parset):
        return(
            np.array([
                calc_one_voltage(
                    # paramters to fit
                    dV,
                    # constant parameters
                    Ne0 = Ne0, dr = dr, d_w = d_w, r_max = False,
                    # the parameters
                    **parset
                )["PE"]
                for dV in dVs
            ])
        )
        
    values = f(dVs, parset)
    s_values = f(dVs, s_pars)

    return(values, s_values)
