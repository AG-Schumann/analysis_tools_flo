import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy.optimize
from scipy.special import erf
from matplotlib.patches import Rectangle
from datetime import datetime
import sys
import inspect
from threading import Thread, Event

try:
    from IPython.display import clear_output
    def clear():
        clear_output(True)
except:
    def clear():
        pass

# Info on Krypton:
# from wikipedia: https://en.wikipedia.org/wiki/Isotopes_of_krypton

# halflife 154.4(11) ns
tau_kr_lit = 154.4/np.log(2)
stau_kr_lit = 1.1/np.log(2)

str_tau_kr_lit = f"τ = ({tau_kr_lit:.1f} ± {stau_kr_lit:.1f}) ns"
label_tau_kr_lit = f"$\\tau = ({tau_kr_lit:.1f} \\pm {stau_kr_lit:.1f})$ ns"

def make_dict(**kwargs):
    '''
    a function for lazy people to turn the arguemnts of a fucntion call into a dictionary (via copy paste)
    '''    
    return(
        {n:v for n, v in kwargs.items()}
    )

def make_fig(nrows=1, ncols=1, w=6, h=4, reshape_ax = True, *args, **kwargs):
    '''
    creates a figure with nrows by ncols plots
    set its size to w*ncols and  h*nrows
    returns fig and reshapen ax (as 1d list) elements
    '''
    
    fig, axs = plt.subplots(nrows, ncols, *args, **kwargs)
    fig.set_size_inches(ncols*w, nrows*h)
    
    if reshape_ax is True:
        if isinstance(axs, plt.Axes):
            axs = np.array([axs])
        else:
            axs = axs.reshape(-1)
            
    return(fig, axs)



def addlabel(ax, label, color = "black", linestyle = "", marker = "", *args, **kwargs):
    ax.plot([], [], label = label, color = color, linestyle=linestyle, marker=marker, *args, **kwargs)



def errorbar(
    ax, x, y, sy,
    color = None, capsize = 5,
    linestyle = "", label = "",
    marker = "", plot = False,
    *args, **kwargs):
    
    if plot is True:
        if marker is not "":
            marker_plot = marker
        else:
            marker_plot = "."
        color = ax.plot(x, y, marker = marker_plot, color = color, linestyle = linestyle, label = label,  *args, **kwargs)[0].get_color()
        label = None
    
    ax.errorbar(x, y, sy, label = label, color = color, capsize=capsize, linestyle=linestyle, marker=marker, *args, **kwargs)

    






def add_fit_parameter(ax, l, p, sp=np.inf, u="", fmt =".1f"):
    '''
    adds nicely formated fit results to legend
    
    l: label
    p: parameter
    sp: uncertainty
    u: unit
    
    '''
    
    brackets = False
    if np.isinf(sp) or (np.abs(sp) > 10*np.abs(p)):
        str_ = f"{p:.1f}"
        
    else:
        str_ = f"{p:{fmt}} \\pm {sp:{fmt}}"
        brackets = True
    
    

    if brackets and (u != ""):
        str_ = f"({str_})"
    
        
    
    str_ = f"${l} = {str_}$"
    if (u != ""):
        str_ = f"{str_} {u}"
    
    addlabel(ax, str_)
    
    return(None)





def median(x, percentile = 68.2, clean = True):
    x_ = np.array(x)*1
    if clean is True:
        x_ = x_[np.isfinite(x_)]

    med = np.median(x_)
    mad = np.percentile(np.abs(x_ - med), percentile)
    unc_med = mad/len(x_)**.5
    
    return(med, mad, unc_med)

def mean(x, percentile = 68.2, clean = True):
    x_ = np.array(x)*1
    if clean is True:
        x_ = x_[np.isfinite(x_)]

    med = np.mean(x)
    mad = np.std(x, ddof = 1)
    unc_med = mad/len(x)**.5
    
    return(med, mad, unc_med)


def median_w(
    x,
    sx,
    percentile = 68.2,
    percentile_med = 50
):
    '''
    returns the weighted median for x with uncertainty sx

    further parameters:
    * percentile (def.: 68.2):
        percentile for calculation of uncertainty
    * percentile_med (def.: 50.0) :
        percentil for median
        
    returns:
        median, mad and uncertainty

    '''

    x, sx = sort(x, sx)
    w = 1/sx

    wx = w * x
    s_wx = np.sum(wx)
    c_wx = np.cumsum(wx)


    thr = s_wx * percentile_med/100

    id_l = max(np.nonzero(c_wx <= thr)[0])
    id_r = min(np.nonzero(c_wx >= thr)[0])
    ids = np.array([id_l, id_r])

    med = np.interp(thr, c_wx[ids], x[ids])


    mad = np.percentile(np.abs(x - med), percentile)
    unc_med = mad/len(x)**.5

    return(med, mad, unc_med)



def bins(x0, x1, bw):
    '''    
    returns bins such that the first bin center is x0
    (x0 is NOT the left side of the first bin!!!)
    '''
    return(np.arange(x0-bw/2, x1+bw, bw))



def count(x):
    '''
    counts each unique value in x
    returns a dict
    '''
    return(
       {
           entrie: np.count_nonzero(np.array(x) == entrie)
           for entrie in np.unique(x)
           
       }
    )
    
    

def chi_sqr(f, x, y, s_y, *pars, ndf = False):
    '''
    returns a tuple with chi^2, ndf and reduced chi^2
    '''
    if ndf is False:
        ndf = len(x) - len(pars)
    
    x = np.array(x)
    y = np.array(y)
    s_y = np.array(s_y)
    
    y_f = f(x, *pars)
    chi = np.sum(((y - y_f)/s_y)**2)
    
    return(
        (chi, ndf, chi/ndf, f"{chi:.1f}/{ndf:.0f} = {chi/ndf:.1f}")
    )


def binning(x, y, bins):
    '''
    returns y sorted into x-bins
    x and y need to have the same form
    '''

    bin_centers = get_bin_centers(bins)
    bin_ids = np.digitize(x, bins)


    bin_contents = {bin_center:[] for bin_center in bin_centers}

    _ = [bin_contents[bin_centers[i-1]].append(y_) for i, y_ in zip(bin_ids, y) if (i > 0) & (i <= len(bin_centers))]
    
    return(bin_contents)




def str_range(x, op = False, debug = False):
    '''
    returns the range of x ("min to max") as a string
    some operations (op) can be applied:
    - "log10" takes the log10 of the absolute values 
      and adds the values sign
    
    
    '''
    try:
        
        dp(f"len(x) at start: {len(x)}")
        x = x[~np.isnan(x)]
        dp(f"len(x) after removing nans: {len(x)}")

        if op == False:
            min_ = min(x)
            max_ = max(x)
            appendix = ""
        elif op == "log10":
            x = x[np.nonzero(x)]
            dp(f"len(x) after removing zeros: {len(x)}")
            min_ = min(x)
            max_ = max(x)
            min_ = np.sign(min_) * np.log10(abs(min_))
            max_ = np.sign(max_) * np.log10(abs(max_))
            appendix = " (log10)"
        elif op == "abs":
            abs_x = abs(x)
            min_ = min(abs_x)
            max_ = max(abs_x)
            appendix = " (abs)"

        digits = max(1, int(np.ceil(np.log10(abs(min_)))+2), int(np.ceil(np.log10(abs(max_)))+2),  )

        return(f"{min_:.{digits}f} to {max_:.{digits}f} {appendix}")
    except Exception as e:
        return(f"error: {e}")
    


def get_parameters(func, start = 0):
    '''
    returns all parameters of a function as strings
    setting start 
    '''
    
    return(
        str(inspect.signature(func))[1:-1].replace(" ", "").split(",")[start:]
    )


defaults = {
    "2d_hist_bins_area": np.logspace(0,5,100),
    "2d_hist_bins_width": np.logspace(1,4,100),

    "2d_hist_label_area": 'Area [pe]',
    "2d_hist_label_width": 'Width [ns]',
    
    "lifetime_hist_bins": np.linspace(-25, 2525, 50),
}

descriptions = {
   "sp_krypton_id_labels": ["1st S1", "2nd S1", "S2", "", ""]
}

def draw_peak(ax_, peak, t0 = 0, x_scaling = 1, y_scaling = 1,  **kwargs):
    if t0 == "auto":
        t0 = peak["time"]
    y = np.trim_zeros(peak["data"])
    x = (peak["time"] - t0 + peak["dt"] * range(len(y))) * x_scaling
    ax_.plot(x, y*y_scaling, **kwargs)


def draw_event(ax_, event, peaks, labels = descriptions["sp_krypton_id_labels"], x_scaling = 1,color_all_peaks = False, **kwarg):
    '''
    old event drawer for the old plugin
    '''
    
    
    t0 = event["s1_1_time"]
    
    
    peaks_this = get_peaks_from_time_tuple(peaks, event["first_and_last_peak_time"])
    
    if color_all_peaks:
        default_color = {"linestyle": "dashed"}
    else:
        default_color = {"color": "grey"}
    
    for label, peak_this in zip(labels, peaks_this[event["peaks_ids"][0:3]]):
        draw_peak(ax_, peak_this, t0 = t0, label = label, x_scaling = x_scaling, **kwarg)
        
    for peak_i, peak_this in enumerate(peaks_this):
        if peak_i not in event["peaks_ids"][0:3]:
            draw_peak(ax_, peak_this, t0 = t0, alpha = 0.75, x_scaling = x_scaling, **default_color, **kwarg)
    
        
    ax_.set_xlabel("time / ns")
    ax_.set_ylabel("signal / PE/sample")
    






def get_ETA(t_start, i, N):
    if i > 0:
        return((datetime.now() - t_start) / i * (N-i))
    else:
        return(float("inf"))


def get_peaks_from_time_tuple(peaks, time_tuple):
    try: 
        id_start = np.nonzero(peaks["time"] == time_tuple[0])[0][0]
        id_end = np.nonzero(peaks["time"] == time_tuple[1])[0][0]
        peaks_return = peaks[range(id_start, id_end+1)]
        
        return(peaks_return)

    except IndexError:
        return([])
    
    
    


def get_bin_centers(bins):
    return(np.array([ np.mean([x1, x2]) for x1, x2 in zip(bins[:-1], bins[1:])]))







def get_hist_data(data, s_offset = 0, **kwargs):
    ''''
    Computes a np.histogram based on 'data', should support all of
    np.histograms parameters
    calculates uncertainties, density and uncertainty of density based on sqrt(N) per bin
    
    Parametrers:
    s_offset: used to calculate s_counts: sqrt(n + s_offset) (default = 0)
    '''
    
    hist_data = np.histogram(data, **kwargs)
        
    bins_centers = [ np.mean([x1, x2]) for x1, x2 in zip(hist_data[1][:-1], hist_data[1][1:])]
    
    
    bins = hist_data[1]
    counts = hist_data[0]
    s_counts = np.sqrt(hist_data[0]+s_offset)
    
    counts_sum = sum(counts)
    s_counts_sum = sum(s_counts**2)**.5
    
    
    density = counts/counts_sum
    s_density = (
          (s_counts/counts_sum)**2
        + (counts/counts_sum**2 * s_counts_sum)**2
    )**.5
    
    
    
    return({
        "bin_centers": bins_centers,
        "bins": bins,
        "counts": counts,
        "s_counts": s_counts,
        "density": density,
        "s_density": s_density,
    })

def draw_2d_hist_plot(ax_, counts, bins_x = False, bins_y = False, aowp = True, colorbar_label = "Counts/bin"):
    if bins_x is False:
        bins_x = defaults["2d_hist_bins_area"]
    if bins_y is False:
        bins_y = defaults["2d_hist_bins_width"]
    
    im = ax_.pcolormesh(bins_x, bins_y, counts.T, norm=LogNorm())
    if colorbar_label:
        cb = plt.colorbar(im, ax=ax_, label=colorbar_label)
    
    if aowp:
        ax_.set_xscale('log')
        ax_.set_yscale('log')

        ax_.set_xlabel(defaults["2d_hist_label_area"])
        ax_.set_ylabel(defaults["2d_hist_label_width"])
    

def dp(message = ""):
    '''
    debug print
    '''
    frame = sys._getframe().f_back
    if "debug" in frame.f_locals and frame.f_locals["debug"]:
        print(f"  \33[34m{frame.f_code.co_name}:\33[0m {message}")



def make_2d_hist_plot(
        x_data,
        y_data,
        ax_ = False,
        bins_x = None, # np.logspace(0,5,100)
        bins_y = None, # np.logspace(1,4,100)
        aowp = True,
        colorbar_label = "Counts/bin",
        debug = False):
    '''
    creates a 2d histogram (eg. area over width) into ax_ (if false, plots directly)
    'aowp' (area over width plot): sets scales to log and adds labels to axis
    'colorbar_label': if not empty string, adds a colorbar with that label
    'bins_x/y' uses default values if not specified: np.logspace(0,5,100) and np.logspace(1,4,100)
    
    '''
    try:
        len(x_data)
        len(y_data)
    except TypeError:
        raise TypeError("ax_ must is the third parameter!")
    
    
    if bins_x is None:
        bins_x = defaults["2d_hist_bins_area"]
        dp("setting bins_y to default")
    if bins_y is None:
        bins_y = defaults["2d_hist_bins_width"]
        dp("setting bins_y to default")
    
    try:
        dp(f'len(bins_x): {len(bins_x)}')
        dp(f'len(bins_y): {len(bins_y)}')
        dp(f'len(x_data): {len(x_data)}')
        dp(f'len(y_data): {len(y_data)}')
    except TypeError:
        dp(f'(bins_x): {bins_x}')
        dp(f'(bins_y): {bins_y}')
        dp(f'(x_data): {x_data}')
        dp(f'(y_data): {y_data}')
    
    counts, bins_x, bins_y = np.histogram2d(
        x_data,
        y_data,
        bins = (bins_x, bins_y),
    )
    
    bin_centers_x = get_bin_centers(bins_x)
    bin_centers_y = get_bin_centers(bins_y)
    
    
    
    
    im = ax_.pcolormesh(bins_x, bins_y, counts.T, norm=LogNorm())
    if colorbar_label:
        cb = plt.colorbar(im, ax=ax_, label=colorbar_label)
    
    if aowp:
        ax_.set_xscale('log')
        ax_.set_yscale('log')

        ax_.set_xlabel(defaults["2d_hist_label_area"])
        ax_.set_ylabel(defaults["2d_hist_label_width"])
    
    if not ax_:
        fig.show()
    
    return((counts, bin_centers_x, bin_centers_y))

def exp_decay(t, A, kappa, C):
    return(A*np.exp(-t*kappa)+C)

def gaus(x, mu = 0, sigma = 1, A = 1, C = 0):
    return(
        A *np.exp(-.5*((x-mu)/sigma)**2) + C
    )

def fit_exp(x, y, offset = -1, meta = False, C0 = True, params = None):
    '''
    fit_exp(x, y, offset = -1)
    '''
    if offset >= 0:
        idx_max_fit = np.argmax(y)+int(offset)
        x = x[idx_max_fit:]
        y = y[idx_max_fit:]
    if params is None:
        params = {}
    if C0:
        params = {"bounds": ((-np.inf, -np.inf, -1),(np.inf, np.inf, 0))}
    fit, cov = scipy.optimize.curve_fit(
                exp_decay,
                x,
                y,
                p0 = [max(y), 1/150, 0],
                absolute_sigma=True,
                **params
        )

    A = fit[0]
    kappa = fit[1]
    C = fit[2]

    s_A = cov[0, 0]**.5
    s_kappa = cov[1, 1]**.5
    s_C = cov[2, 2]**.5

    tau = 1/kappa
    s_tau = s_kappa / kappa**2
    
    range_x = np.array(np.linspace(min(x), max(x), 1000))
    calc_y_exp = A * np.exp(-range_x * kappa) + C

    t_half = tau * np.log(2)
    s_t_half = s_tau * np.log(2)

    
    if meta is not False:
        meta = {
            "fit_start": idx_max_fit,
            "data": (x, y),
            "calc": (range_x, calc_y_exp),
            
        }
    
    
    return({
        "parameters": {
            "A": A,
            "C": C,
            "kappa": kappa,
        },
        "uncertainties": {
            "A": A,
            "C": C,
            "kappa": kappa,
        },
        "halflife": {
            "t_half": t_half,
            "s_t_half": s_t_half,
            "tau": tau,
            "s_tau": s_tau,
        },
        "meta": meta
        
    })





def sort(x, *args):
    '''
    sorts all inputs by the order of x
    '''
    idx = np.argsort(x)
    out  = [x[idx]]
    
    for arg in args:
        out.append(arg[idx])
        
    return(out)




def remove_zero(x, *args):
    '''
    takes arbitraly many arrays and keeps only entries where x is not zero
    returns the filtered values each as np.array
    '''
    idx_keep = np.nonzero(x)
    x = np.array(x)[idx_keep]
    out = []
    for xi in args:
        if xi is not None:
            out.append(np.array(xi)[idx_keep])
        else:
            out.append(None)
    if len(args) > 0:
        return(x, *out)
    else:
        return(x)


def fit_gaus(x, y, absolute_sigma = True, sigma = None, meta = False, **kwargs):

    x = np.array(x)
    y = np.array(y)
    
    sigma_bool = False
    if sigma is not None:
        # sigma will be used for the fit parameter sigma
        sigma_bool = True
        # remove zero Sigma
        sigma, x, y = remove_zero(sigma, x, y)
        
        
    
    idx_max_fit = np.argmax(y)
    
    x_top = x[np.nonzero(y > 0.5 * max(y))[0]]
    
    
    
    
    start_params = {
        "mu": x[idx_max_fit],
        "sigma": max(x_top) - min(x_top),
        "A": max(abs(y))-min(abs(y)),
        "C": min(y),
    }
    
    
    fit, cov = scipy.optimize.curve_fit(
            gaus,
            x,
            y,
            p0 = [list(start_params.values())],
            absolute_sigma = absolute_sigma,
            sigma = sigma,
            **kwargs,
    )
    
        
    mu, sigma, A, C = fit
    s_mu, s_sigma, s_A, s_C = np.diagonal(cov)**.5
    
    
    parameters = {
        "mu": mu,
        "sigma": sigma,
        "A": A,
        "C": C,
    }
    
    if sigma_bool:
        y_fit = gaus(x, **parameters)
        chi_2 = np.sum(((y_fit - y)/sigma)**2)
        ndf = len(x) - 4
        chi2_red = chi_2/ndf
        chi_2_dict = {
            "chi2": chi_2,
            "ndf": ndf,
            "chi2_red": chi2_red,
        }
    else:
        chi_2_dict = {
            "chi2": False,
            "ndf": False,
            "chi2_red": False,
        }
    
        
    
    
    range_x = np.array(np.linspace(min(x), max(x), 1000))
    calc_y = gaus(range_x, **parameters)
    start_y = gaus(range_x, **start_params)
    
    
    if meta is not False:
        meta = {
            "data_x": x,
            "data_y": y,
            "calc_x": range_x,
            "calc_y": calc_y,
            
            "start_y": start_y,
        }
    
    
    
    return({
        "parameters": parameters,
        "chi2": chi_2_dict,
        "uncertainties": {
            "mu": s_mu,
            "sigma": s_sigma,
            "A": s_A,
            "C": s_C,
        },
        "meta": meta,
        
    })


def get_corners(filter_, debug = False, corners = False):
    '''
    calculates corners for rectangles by a list of filters.
    each filter is a tuple with three entries:
      1.: "x" or "y"
      2.: "<", "<=" or ">", ">="
      3.: the value for this filter
    '''
    
    corner_labels = ["left", "right", "bottom", "top"]
    if not corners:
        corners = [
            min(defaults["2d_hist_bins_area"])/100,  # left
            max(defaults["2d_hist_bins_area"])*100,  # right
            min(defaults["2d_hist_bins_width"])/100, # bottom
            max(defaults["2d_hist_bins_width"])*100, # top
        ]
    
    
    
    for axis, comp, value in filter_:
        corner = 2*(axis=="y") + ("<" in comp[0])
        if debug: print(f'  <get_corners> axis:{axis}, comp: {comp}, value: {value}, corner: {corner} ({corner_labels[corner]})')
        
        corners[corner] = value
    return(corners)


def draw_rect(ax_, corners, edgecolor = "red", linewidth = 2, facecolor='none', debug = False, **kwarg):
    '''
    draws rectangles into ax ax_ with corners given by corners,
    modify style via **kwargs
    '''
    xl = max(ax_.get_xlim()[0], corners[0])
    xr = min(ax_.get_xlim()[1], corners[1])
    yb = max(ax_.get_ylim()[0], corners[2])
    yt = min(ax_.get_ylim()[1], corners[3])

    
    
    
    
    dp(f"x: {xl, xr}")
    dp(f"y: {yb, yt}")
    
    ax_.add_patch(
        Rectangle(
            (
                xl,
                yb
            ),
            xr-xl,
            yt-yb,
            facecolor=facecolor,
            edgecolor = edgecolor,
            linewidth = linewidth,
            **kwarg
        )
    )
    
    
    
def iterator_cell(ax_, min_i=0, max_i=float("inf")):
    n_rows, n_cols, *_ = [*ax_.shape, 1,1,1]
    i = min_i
    while ((i < n_cols*n_rows) & (i < max_i)):
        if n_cols > 1 and n_rows > 1:
            row, col = int(i/n_cols), i%n_cols
            yield(i, (row, col))
        else:
            yield(i, (i))
        i += 1

def scale(x, s=1, l=0, u=1):
    x = x-min(x)
    x = x / max(x) * s
    return(x)



def project_counts(ax_, hist_2d,
    width_threshold = False, alpha = .25,
    f_x = .2, f_y = .5,
    scale_x = "linear", scale_y = "linear",
    color_x = "blue", color_y ="orange",
    *args, **kwargs
    ):
    '''
    add projections into ax_. hist_2d is the full output (3-tuple) of make_2d_hist_plot.
    use scale_x/y to scale values in x/y up/down or turn them off
    f_x/y: scaling factors
    scale_x/y: type of x/y scale (linear, log, ....)
    '''
    counts, bin_center_area, bin_center_width = hist_2d

    projection_area = np.sum(counts, axis = 1)
    projection_width = np.sum(counts, axis = 0)
    
    if f_x:
        ax_x = ax_.twinx()
        ax_x.set_yscale(scale_x)
        ax_x.set_ylim(0, 1)
        proj_x = scale(projection_area, s = f_x)
        ax_x.plot(bin_center_area, proj_x)
        ax_x.axes.yaxis.set_visible(False)

    
    if f_y:
        ax_y = ax_.twiny()
        ax_y.set_xscale(scale_y)
        ax_y.set_xlim(0, 1)
        proj_y = scale(projection_width, s = f_y)
        ax_y.plot(proj_y, bin_center_width, alpha = alpha, color = color_y, *args,**kwargs)
        ax_y.axes.xaxis.set_visible(False)

    if width_threshold:
        id_width_thr = min(np.nonzero(bin_center_width >= width_threshold)[0])
        
        projection_width_lower = projection_width[:id_width_thr]
        projection_width_higher = projection_width[id_width_thr:]
        
        
        n_peak_width_lower = int(np.sum(projection_width_lower))
        n_peak_width_higher = int(np.sum(projection_width_higher))

        ax_.plot([], [], " ", label = f"N_peaks (width <= {width_threshold}): {n_peak_width_lower}")
        ax_.plot([], [], " ", label = f"N_peaks (width > {width_threshold}): {n_peak_width_higher}")
    
    return(projection_area, projection_width)





def get_data_quick(run_str, context, dtype = "peaks", max_time = 10, config = False, *args, **kwargs):
    '''
    function that obtains strabra run data but stops after a set timeout
    
    retuns a 2-tuple with:
        1: bool, wheter the data got obtained or not
        2: the dataset, the errormessage or False if it timed out
        
    parameters:
    run_str: the runstring of the desired run (sring, five digits, leading zeros)
    context: the context whih whioch to laod the data
    dtype: datatsype ('peaks' is default, events, .... can be chosen)
    max_time: the maximum amount of seconds the system tries to obtain data
    
    '''
    if config is False:
        config = {}
    
    result_mutlithreading = (False, False)
    
    def check_data(run_str, dtype = "peaks", context = context):
        
        # access closest result_mutlithreading variable
        nonlocal result_mutlithreading, config, args, kwargs
        try:
            result_mutlithreading = (True, context.get_array(run_str, dtype, config = config, *args, **kwargs))
        except Exception as e:
            result_mutlithreading = (False, e)

    
    
    action_thread = Thread(target=check_data, kwargs={"run_str": run_str, "dtype": dtype, "context": context})

    action_thread.start()
    action_thread.join(timeout = max_time)
    
    
    return(result_mutlithreading)








def multi_gaus_wrapper(x, *args):
    '''
    fitting wrapper for the multi_gaus function
    order of *args: A1, mu1, sigma1, A2, ....
    (used for curve_fit)
    '''
    return(
        multi_gaus(x, *multi_gaus_argument_compactor(*args))
    )


def multi_gaus_argument_compactor(*args):
    '''
    Turnn aribtrary long linst of arguments into three tuples used
    in multi_gaus
    '''    

    if(len(args) %3 != 0):
        raise ValueError(f"number of arguments must be multiple of 3 (is {len(args)})")
    
    
    return(args[0::3], args[1::3], args[2::3])




def multi_gaus(x, A, mu, sigma):
    '''
    Calculates y for the sum of arbitrarily many gaus functions
    A, mu and sigma must be tuples of same length.
    
    use multi_gaus_wrapper for fitting (arguments there: A1, mu1, sigma1, A2, ....)
    '''

    if not (len(A) == len(mu) == len(sigma)):
        raise ValueError(f"lengths are different: A: {len(A)}, mu: {len(mu)}, sigma: {len(sigma)}")

        
    y = np.sum(
        [
            Ai * np.exp(-(x-mui)**2 / (2*sigmai)**2)
            for Ai, mui, sigmai in zip(A, mu, sigma)
        ],
        axis = 0
    )
    return(
        y
    )