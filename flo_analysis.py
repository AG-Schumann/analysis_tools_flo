import numpy as np
import scipy.optimize
import pandas as pd
from scipy.optimize import curve_fit
import flo_histograms as fhist
import matplotlib.pyplot as plt
import decimal


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

default_bins["drifttime"] = np.arange(position_gate+1, position_cathode-1, 1)
default_bins["full_range_s1"] = np.arange(-20*bw_s1_dt/2, 100+2/3*bw_s1_dt, bw_s1_dt)
default_bins["area_S2"] = np.arange(-bw_s2_area/2, 8000+bw_s2_area*3/2, bw_s2_area)
default_bins["area_S1"] = np.arange(-bw_s1_area/2, 1000+bw_s1_area*3/2, bw_s1_area)




# default_corrections = {
    # 'S1':{
        # 'a': [1.5466517373651314, 1.0412855975322826, 0.518685272788315],
        # 'b': [200.9746348880132, 145.43723496246653, 54.98796099205108],
    # },
    # "sS1":{
        # 'a': [0.019741374740314042, 0.015930507676975354, 0.007051187861560944],
        # 'b': [0.4605871081950396, 0.3783252192920378, 0.16472936074754693],
    # }
# }
    
def np_array_all(*args):
    out = [-1] * len(args)
    for i, arg in enumerate(args):
        out[i] = np.array(arg)
    return(out)

def get_g1g2(a, c, cov, W = 13.7):
    '''
    returns g1, g2, sg1, sg2
    '''
    var_a = cov[0, 0]
    var_c = cov[1, 1]
    cov_ca = cov[1,0]
    
    g1 = -c/a * W/1000
    sg1 = ((c/a**2)**2*var_a
            + (1/a)**2*var_c
            + (c/a**2)*(1/a)*cov_ca
        )**.5*W/1000
        
    g2 = c * W/1000
    sg2 = var_c**.5 * W/1000
    
    return(g1, g2, sg1, sg2)



def config_to_dict(config_df):
    '''
    converts a config pandas dataframe into a dictionary.
    
    call with object from context.show_config(...):
    config_df = context.show_config("peaks")
    '''
    config = config_df.to_dict(orient = "recods")

    def choose_value(setting_dict):
        if setting_dict["current"] != '<OMITTED>':
            return(setting_dict["current"])
        else:
            return(setting_dict["default"])

    return({
        x["option"]:choose_value(x)
        for x in config
    })


def get_latex_fraction(x, prefix = "", suffix = "", prefix_if_zero = False, suffix_if_zero = False):
    '''
    requries decimals package
    
    
    returns the value x as a nice latex fraction
    (you need to put $ around this output)
    
    
    x:
        value to calculate fraction of
    prefix, suffix:
        prefix and suffix (liek a unit)
    prefix_if_zero, suffix_if_zero:
        wheter to show the prrefix or suffix if the value is zero
    
    '''
    
    
    
    nom, den = decimal.Decimal(x).as_integer_ratio()
    str_latex = ""
    
    if nom < 0:
        str_latex += "-"
    
    nom = np.abs(nom)
    
    if nom == 0:
        str_latex += "0"
    else:
        if (nom == 1) & (den == 1):
            if suffix == "":
                str_latex += "1"
        elif (den == 1):
            str_latex += f"{nom}"
        else:
            str_latex += f"\\frac{{{nom}}}{{{den}}}"
    
    
    
    if prefix_if_zero is False and x == 0:
        prefix = ""
    
    if suffix_if_zero is False and x == 0:
        suffix = ""
    
    
    return(f"{prefix}{str_latex}{suffix}")
 



    
def fwhm(x, y, c = 0, scale = .5):
    '''
    returns the full width half maximum info of x-y points:
     * width
     * central position
     * used threshold
     
     
    takes:
     * x, y: x and y of the data
     * c: constant offset of y ("baseline level")
     * scale: relative height to find (default: 0.5)
    
    '''
    
    def get_intercept(x, y, thr):
        fit = np.polyfit(x, y, 1)
        return((thr-fit[1])/fit[0])



    max_y = max(y)
    idx_max = np.argmax(y)
    loc_max = x[idx_max]
    thr = (max_y*scale)
    idx_low = np.nonzero(y < thr)[0]
    
    idx_l = idx_low[max(np.nonzero(idx_low < idx_max)[0])]+np.array([0,1])
    idx_r = idx_low[min(np.nonzero(idx_low > idx_max)[0])]+np.array([-1,0])



    l = get_intercept(x[idx_l], y[idx_l], thr)
    r = get_intercept(x[idx_r], y[idx_r], thr)

    width = r-l
    center = (l+r)/2
    

    return(width, center, thr+c)

    
    
def exp_decay(t, A, tau, C):
    return(A*np.exp(-t/tau)+C)


def exp_decay_zero(t, A, tau, C):
    return(A*np.exp(-t/tau))

    

def fit_exp_decay(decays, bins = None, p0 = [100, 222, 0], ax = False):
    '''
fits an exponential decay 'A*exp(-t/tau) + C'
to a histogram of 'decays'
plots into ax if ax is not False

parameters:

  decays:
    the individual decay times (a histogram will be created from those)
  bins:
    the bins information to bin the data. Used in np.histogram(x, bins)
    if left at default (None), uses steps of 50 from 25 to 2525
  p0:
    Starting parameters of fit default = [100, 222, 0]
    (OK for Krypton decays)
  ax:
    the Axis element to plot into, default = False, so no plot
  
returns:
    a tuple of the fit result, its uncertainties and a chi2 tuple
        
    '''
   
    if bins is None:
        bw = 50
        bins = np.arange(bw/2, 2500+bw/2, bw)
    
    
    y, bins = np.histogram(decays, bins = bins)
    x = fhist.get_bin_centers(bins)
    sy = y**.5
    
    start_fit = np.argmax(y)
    
    yf, xf, syf = y[start_fit:], x[start_fit:], sy[start_fit:]
    yf, xf, syf = fhist.remove_zero(yf, xf, syf)
    
    def exp_decay(t, A, tau, C):
        return(A*np.exp(-t/tau)+C)
    
    
    fit, cov = curve_fit(
        exp_decay,
        xf, yf, 
        absolute_sigma=True,
        sigma=syf,
        p0 = p0
    )
    sfit = np.diag(cov)**.5
    chi2 = fhist.chi_sqr(exp_decay, xf, yf, syf, *fit)
    
    
    
    
    
    
    if ax is not False:
        xc = np.linspace(x[0], x[-1], 1000)
        ycf = exp_decay(xc, *fit)
        
        
        plt_ = ax.plot(xf, yf, "x" , label = f"data (N = {np.sum(y, dtype = int)})")[0]
        fhist.errorbar(ax, x, y, sy, color = plt_.get_color())
        
        ax.plot(xc, ycf, label = f"fit: $chi^2_{{red}}={chi2[3]}$")
        
        ax.set_xlabel("decay time / ns")
        ax.set_ylabel("counts")
        
        fhist.addlabel(ax, f"$\\tau = ({fit[1]:.1f}\\pm{sfit[1]:.1f})$ ns")
        fhist.addlabel(ax, f"$A = {fit[0]:.1f}\\pm{sfit[0]:.1f}$")
        fhist.addlabel(ax, f"$C = {fit[2]:.1f}\\pm{sfit[2]:.1f}$")
        
        ax.set_ylim(top = max(y))
        ax.legend(loc = "upper right")
    return((fit, sfit, chi2))
    



def correct_s2(kr, lifetime):
    # replace by np.exp((kr["time_drift"]-t_ref)/lifetime)
    kr["cS2"] = kr["area_s2"] * np.exp(kr["time_drift"]/lifetime)
    kr["cS21"] = kr["area_s21"] * np.exp(kr["time_drift"]/lifetime)
    kr["cS22"] = kr["area_s22"] * np.exp(kr["time_drift"]/lifetime)
    return(None)
    
    
def lin(x, a, c):
    return(a*x + c)
    
def corr_lin(x, y, a, c, t_start = position_gate, t_end = position_cathode):
    
    t0 = (t_start + t_end)/2
    
    y_corr = y * 1
    y_calc = y / lin(x, a, c)*lin(t0, a, c)
    
    idx_corr = np.nonzero((x >= t_start) & (x <= t_end))[0]
    y_corr[idx_corr] = y_calc[idx_corr]
    
    
    return(y_corr)
    
def correct_s1(kr, s1_corr_pars):
    kr["cS1"]  = corr_lin(kr["time_drift"], kr["area_s1"], *s1_corr_pars["s1"])
    kr["cS11"] = corr_lin(kr["time_drift"], kr["area_s11"], *s1_corr_pars["s11"])
    kr["cS12"] = corr_lin(kr["time_drift"], kr["area_s12"], *s1_corr_pars["s12"])
    
    
    return(None)
    

def plot_binned(ax, x, y, bins = 10, marker = ".", label = "", eb_1 = False, eb_2 = False, x_max_plot=False, n_counts_min=2, nresults_min=2, *args, **kwargs):
    
    cbc, cmedian, cmd_sd, cmd_unc = get_binned_data(x, y, bins=bins, n_counts_min=n_counts_min, nresults_min=nresults_min, *args, **kwargs)
    
    if x_max_plot is not False:
        _, cbc_p, cmedian_p, cmd_sd_p, cmd_unc_p = fhist.remove_zero(cbc <= x_max_plot, cbc, cmedian, cmd_sd, cmd_unc)
    else:
        cbc_p, cmedian_p, cmd_sd_p, cmd_unc_p = cbc, cmedian, cmd_sd, cmd_unc
        
        
    if ax is not False:
        _ = ax.plot(cbc_p, cmedian_p, ".", label = label, marker= marker, *args, **kwargs)[0]
        if eb_1 is True:
            fhist.errorbar(ax, cbc_p, cmedian_p, cmd_unc_p, color = _.get_color())
        if eb_2 is True:
            fhist.errorbar(ax, cbc_p, cmedian_p, cmd_sd_p, color = _.get_color(), alpha= .25)
    
    return(cbc, cmedian, cmd_sd, cmd_unc)



def get_binned_data(dt, area, bins, n_counts_min=2, nresults_min=2, *args, **kwargs):
    '''
    helper function for get_e_lifetime_from_run
    
    bins area by dt and calculates median statistics
    
    returns bin-center, median, spread and uncertainty for all bins
        
    '''
    

    binned_data = fhist.binning(dt, area, bins)
    bc = []
    md = []

    for bc_, values in binned_data.items():
        if len(values) >= n_counts_min:
            bc.append(bc_)
            md.append(fhist.median(values))
    
    if len(bc) >= nresults_min:
        median, md_sd, md_unc = zip(*md)
    
        return(np.array(bc), np.array(median), np.array(md_sd), np.array(md_unc))
    else:
        return(np.array([]), np.array([]), np.array([]), np.array([]))


def get_e_lifetime_from_run(kr, ax = False, bins = None, field = "area_s2", *args, **kwargs):
    '''
calculates the electron lifetime of a run based on the uncorrected S2 area and the drift time


parameters:
kr: the filtered dataset of sp_krypton
ax: where a plot should be added to (on default: no plot)
bins: the bins for the S2 binning, by default: 5 to 41 in steps of 2

modifies:
corrects the S2 area (kr["cs2"]) automatically


returns:
the lifetime plus uncertainty (in  µs)



'''
    
    
    if bins is None:
        bins = default_bins["drifttime"]

    bc, median, md_sd, md_unc = get_binned_data(kr["time_drift"], kr[field], bins)
    
    
    
    # select what info to use for fit
    xf, yf, syf = bc, median, md_unc
    
    p0 = [median[0], 100, 0]

    fit, cov = curve_fit(
        exp_decay_zero,
        xf, yf,
        absolute_sigma=True,
        sigma=syf,
        p0 = p0,
        bounds = (0, np.inf)
    )



    sfit = np.diag(cov)**.5
    chi2 = fhist.chi_sqr(exp_decay, xf, yf, syf, *fit)

    xc = np.linspace(bc[0], bc[-1], 1000)
    ycf = exp_decay(xc, *fit)



    correct_s2(kr, fit[1])
    cbc, cmedian, cmd_sd, cmd_unc = get_binned_data(kr["time_drift"], kr["cS2"], bins)





    if ax is not False:
        plt_data = ax.plot(xf, yf, ".", label = f"raw S2 area\n(median $\\pm$ uncertainty)")[0]
        fhist.errorbar(ax, xf, yf, syf, color = plt_data.get_color())
        ax.plot(xc, ycf, label = f"fit $\\chi^2_{{red}}= {chi2[3]}$")
        fhist.addlabel(ax, f"$\\tau = ({fit[1]:.1f}\\pm{sfit[1]:.1f})$ ns")
        fhist.addlabel(ax, f"$A = {fit[0]:.1f}\\pm{sfit[0]:.1f}$")
        fhist.addlabel(ax, f"$C$ fixed to 0")


        plt_data_c = ax.plot(cbc, cmedian, "x", label = f"cS2")[0]
        fhist.errorbar(ax, cbc, cmedian, cmd_unc, color = plt_data_c.get_color())

        ax.set_xlabel("drifttime / µs")
        ax.set_ylabel("S2 Area / PE")
        
        ax.legend(loc = "lower left")

    return({
        "e-lifetime": (fit[1], sfit[1]),
        "cS2_0": (fit[0], sfit[0]),
        "chi2": (chi2[2], chi2[3])
    })
    
    
    

    
def fit_s1_field(kr, field, bins_dt, ax = False,
    t_start = position_gate, t_end = position_cathode,
    cut_after_cathode = 10,
    *args, **kwargs
):
    '''
    performs a S1 calibration on "field" (s1, s11 or s12)
    
    cut_at_cathod: remove values after cut_after_cathode µs after cahode (default = 10)
    (just for plotting)
    
    '''
    
    
    t0 = (t_start + t_end)/2
    
    
    kr_ = kr.copy()
    t_max = False
    if cut_after_cathode is not False:
        t_max = position_cathode + cut_after_cathode
        # print(f"cutting dt at {t_max:.2f}")
        
    
    
    
    x, y, sdy, sy = plot_binned(
        ax, kr["time_drift"], kr[f"area_{field}"], label = f"uncorrected (n={len(kr)})", bins=bins_dt,
        eb_1 = True, eb_2 = True,
        x_max_plot = t_max,
        *args, **kwargs
    )
    
    
    
    _, xf, yf, syf = fhist.remove_zero((x >= t_start) & (x <= t_end), x, y, sy)
    
    
    
    
    fit, cov = curve_fit(
        lin,
        xf, yf,
        absolute_sigma=True,
        sigma=syf,
    )



    sfit = list(np.diag(cov)**.5)
    
    fit = list(map(float, fit))
    sfit = list(map(float, sfit))
    
    chi2 = fhist.chi_sqr(lin, xf, yf, syf, *fit)

    xc = np.linspace(t_start, t_end, 1000)
    ycf = lin(xc, *fit)
    
    corr_pars = (*fit, t_start, t_end)

    if ax is not False:
        ax.set_title(field.upper())
        ax.plot(xc, ycf, label = f"$\\chi^2_{{red}} = {chi2[3]}$")
        

        for par, p, sp in zip(["a", "c"], fit, sfit):
            fhist.add_fit_parameter(ax, par, p, sp, fmt = ".2f")


        ax.set_xlabel("drifttime /  µs")
        ax.set_ylabel("signal area /  PE")

        ax.legend(loc = "lower right")
        
        # correcting (just for plotting)
        kr[f"c{field.upper()}"] = corr_lin(kr["time_drift"], kr[f"area_{field}"], *corr_pars)
        
        xc, yc, sdyc, syc = plot_binned(
            ax, kr["time_drift"], kr[f"c{field.upper()}"], label = f"corrected (n={len(kr)})", bins=bins_dt,
            eb_1 = True, eb_2 = False,
            x_max_plot = t_max,

        )
        
        
    return(corr_pars, (fit, sfit, chi2))


    
    
    
    
    
    
    
    
    
    
def get_peaks(data, signals, split = -1):
    '''
    d: filtered sp_krypton data
    signals: list of signals: [s11, s12, s21, s22, s1, s2]
    split: whether signals need to have split S2s: yes(1), no(0), irrelevant(-1)
    '''

    if split == 0:
        d = data[~data["s2_split"]]
    elif split == 1:
        d = data[data["s2_split"]]
    else:
        d = data


    a, w = np.array([]), np.array([])
    for s in signals:
        a = np.append(a, d[f"area_{s}"])
        w = np.append(w, d[f"width_{s}"])

    return(w, a)
    
    
    
    
    
    

    
def plot_signals(data, fname = False, title = ""):
    '''
    takes a dataset (data) and plots all individuals signals as 2d histogram (S11, S12, ....),
    saves grafic into {fname}.png
    
    '''
    
    plt.rcParams["figure.figsize"] = (16,12)
    fig, axs = plt.subplots(2, 3)
    plt.suptitle(f"{title}")
    axs = axs.reshape(-1)



    calls = [
        ("all first S1", ["s11"], -1),
        ("all second S1", ["s12"], -1),
        ("all unsplit S2", ["s21"], 0),
        ("all split first S2", ["s21"], 1),
        ("all split second S2", ["s22"], 1),

    ]



    for ax, (title, signals, split) in zip(axs, calls):

        ax.set_title(f"{title}")
        w, a = get_peaks(data, signals = signals, split = split)
        hist_2d = fhist.make_2d_hist_plot(a, w, ax_ = ax)

        n_binned = np.sum(hist_2d[0], dtype = int)

        fhist.addlabel(ax, f"total datapoints: {len(w)}")
        fhist.addlabel(ax, f"datapoints in bins: {n_binned}")


        ax.legend(loc = "upper right")

    
    plt.subplots_adjust(
        left = .05,
        right = .95,
        bottom=  0.05,
        top = .925,
        wspace = 0.5,
        hspace = 0.25,
    )
    
    if fname is False:
        plt.show()
    else:
        plt.savefig(f"{fname}.png")
        plt.close()