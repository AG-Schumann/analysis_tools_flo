import sys
import os
from datetime import datetime
from flo_analysis import *
import flo_fancy

import flo_decorators
from threading import Thread, Event
import matplotlib.pyplot as plt
import matplotlib as mpl

def now():
    return(datetime.now())

# import 
sys.path.insert(0,"/data/workspace/Flo/straxbra_flo/strax")
import strax
print("\33[34mStrax:\33[0m")
print(f"Strax version: {strax.__version__}")
print(f"Strax file:    {strax.__file__}")

sys.path.insert(0,"/data/workspace/Flo/straxbra_flo/straxbra")
import straxbra
print("\33[34mStraxbra:\33[0m")
print(f"straxbra file:           {straxbra.__file__}")
# print(f"straxbra version:        {straxbra.__version__}")
# print(f"SpKrypton version:       {straxbra.plugins.SpKrypton.__version__}")
# print(f"GaussfitPeaks version:   {straxbra.plugins.GaussfitPeaks.__version__}")
# print(f"SPKryptonS2Fits version: {straxbra.plugins.SPKryptonS2Fits.__version__}")
print(f"EventFits version:       {straxbra.plugins.EventFits.__version__}")



context_sp = straxbra.SinglePhaseContext()
context_dp = straxbra.XebraContext()
db = straxbra.utils.db


labels = {
    "event_fits": np.array(["first S1", "second S1", "first S2", "second S2"]),
    "event_fits_summary": np.array(["first S1", "second S1", "first S2", "second S2", "unsplit S1", "unsplit S2", "total S1", "total S2", "afterpulse"]),
    "sp_krypton_summary": np.array(['first S1', 'second S1', 'first S2', 'second S2', 'unsplit S1', 'unsplit S2', 'total S1', 'total S2']),
    
}

def draw_woa_multi(
    ds, ax = False, draw = "0123", plugin = "event_fits_summary", add_grid = True, show_only_max_count = True,
    vmin = 1, global_scale = False, show_counts = True, title = "", show_precut = False,
    labels_ = False, field_a  = "areas", field_w  = "widths", cmaps = False, cmap_cb = False):
    '''
    draws multiple signals in one width over area plot
    each signal get a different color
    
    params:
    ds: the dataset that contains the fields "areas" and "widths"
    draw: the field indize to draw. must be iteratble (string  or list)
        default: "0123"
    plugin: used to get labels via labels[plugin]
        default: event_fits_summary
    add_grid: adds a grid to the plot if set to True
        default: True
    show_only_max_count: If True draws only the max count per bin.
        prevents lower values overwriting higher values
        default True
    vmin: bins with less counts are set to 0
        default 2
    global_scale: wheter the colorbars should have the same vmax or not
        default True
    show_counts: wheter the legend shoudl also contain the number of binned datapoints
        default True
    labels_: alternative to plugin, must be a list
        default: False
    field_a, field_w: fields that contain area/width
        default: "areas", "widths"
    ax: where to put the plot into
        default: False, creates a new one
    cmaps: the colormaps to use, if cmaps run out fall back to cmap_cb
        default False uses ['Purples', 'Blues', 'Greens', 'Reds', 'Oranges']
    cmap_cb: the colormap for the colorbar
        default False uses "Greys"
    
    
    '''
    
    if (vmin is False) or (vmin < 1):
        vmin = 1
    
    if labels_ is False:
        labels_ = labels[plugin]
    
    draw = [int(i) for i in draw]
    n_draw = len(draw)

    counts = [False]*n_draw
    labs = [""]*n_draw
    for i, j in enumerate(draw):
        try:
            
            count, bca, bcw = fhist.make_2d_hist_plot(ds[field_a][:, j], ds[field_w][:, j])
            
            n_precut = np.nansum(count, dtype = int)            
            
            count[count < vmin] = 0
            n_postcut = np.nansum(count, dtype = int)
            counts[i] = count
            
            labs[i] = labels_[j]
            if show_counts is True:
                if show_precut is True:
                    str_appendix = f" (N: {n_postcut}/{n_precut})"
                else:
                    str_appendix = f" (N: {n_precut})"
                labs[i] = f"{labs[i]}{str_appendix}"
            
        except Exception as e:
            print(f"failed at {i} (label: {labs[i]}): {e}")
    
    
    if ax is False:
        ax = fhist.ax()
    
    if title != "":
        ax.set_title(title)
    vmax = np.nanmax(counts)
    
    if vmax == 0:
        raise ValueError("maxmimum of counts is zero!")
    

    if cmaps is False:
        cmaps = [
            'Purples',
            'Blues',
            'Greens',
            'Reds',
            'Oranges'
        ]
    if cmap_cb is False:
        cmap_cb = "Greys"
    
    if global_scale is True:
        im = ax.pcolormesh(
            bca, bcw, np.zeros_like(count.T),
            norm=fhist.LogNorm(),
            cmap = cmap_cb, vmin=vmin, vmax=vmax
        )
        cb = plt.colorbar(im, ax=ax, label="Counts/bin",)
        lims = dict(
            vmin = vmin,
            vmax = vmax,
        )
    else:
        lims = dict(
            vmin = vmin,
        )
    
    max_values = np.nanmax(np.array(counts), axis=0)
    
    for count, label in zip(counts, labs):
        
        if (count is not False) and (label != ""):
            if len(cmaps) > 0:
                cmap = cmaps.pop(0)
            else:
                cmap = cmap_cb
            ax.plot(
                [],
                "o",
                color = mpl.cm.get_cmap(cmap)(.75),
                label = label
            )
            
            if show_only_max_count is True:
                count_ = count * (count >= max_values)
            else:
                count_ = count
            
            
            im_ = ax.pcolormesh(
                bca, bcw,
                count_.T,
                norm=fhist.LogNorm(),
                cmap = cmap,
                **lims
            )




    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(fhist.defaults["2d_hist_label_area"])
    ax.set_ylabel(fhist.defaults["2d_hist_label_width"])

    ax.legend(loc = "upper right")
    if add_grid is True:
        ax.grid()
    return(counts, bca, bcw)





def draw_woa(ds, plugin = "event_fits_summary", draw = "all", title = "", ret = False, loc = "upper right", **kwargs):
    '''
        draws each individual signals if available
    '''
    
    titles = labels[plugin]
    
    if draw == "all":
        draw = np.array(list(range(len(titles))))
    else:
        draw = np.array([int(i) for i in draw])
    fig, axs = fhist.make_fig(n_tot = -len(draw), axis_off = True)
    
    if title != "":
        fig.suptitle(title)
    
    for i, ax, title in zip(draw, axs, titles[draw]):
        ax.set_title(title)
        ax.set_axis_on()
        try:
            fhist.make_2d_hist_plot(ds["areas"][:,i],ds["widths"][:,i], ax, loc = loc, **kwargs)
            ax.grid()
        except Exception:
            pass
    plt.subplots_adjust(
        hspace = .25
    )
    plt.show()
    if ret is True:
        return(fig, axs)




def rs(runs):
    '''
    turns runs into a list of run strings
    '''
    
    if isinstance(runs, int):
        runs = [runs]
    elif isinstance(runs, str):
        runs = [runs]

    
    rstrs = [f"{r:0>5}" for r in runs]
    
    return(rstrs)


def check(runs, target, context = False, config = False, cast_int = True, v = True):
    
    if context is False:
        context = context_sp
    if config is False:
        if target[:10] == "sp_krypton":
            config = default_config
        else:
            config = {}
        
    runs = rs(runs)
    
    t0 = now()
    if v is True: print(f"checking \33[1m\33[34m{target}\33[0m for {len(runs)} runs...")
    runs = [r for r in runs if context.is_stored(r, target, config = config)]
    t1 = now()
    if cast_int is True:
        runs = list(map(int, runs))
    if v is True: print(f"    found data for {len(runs)} runs in {t1-t0}")
    return(runs)



def load(runs, target, context = False, config = False, check_load = True, v = True, **kwargs):
    '''
    loads data if availabe
    
    parameters:
        runs: list of runs to load
            (either list or one value; either string or int; doestn't matter)
        target: the plugin to load
        context: which context to use (mystrax.context_sp or mystrax.context_dp)
            default False falls back to mystrax.context_sp
        config: custom config to be loaded
            default False
            if False uses default configs:
                - mystrax.default_config for runs that start with sp_krypton
                - {} for everythin else
        check load: check if the data is available before attmepting to load it
            use a string of a different target to check the existance of that data to load this data
            set to false if you want to skip all checks
            default True
        v: verbose toggle, prints messages
            default True
            
    check_load
    
    '''
    
    if context is False:
        context = context_sp
    if config is False:
        if target[:10] == "sp_krypton":
            config = default_config
        else:
            config = {}
    
    runs = rs(runs)
    if check_load is not False:
        if isinstance(check_load, str):
            target_ = check_load
        else:
            target_ = target        
        runs = check(runs = runs, target = target_, context = context, config = config, v = v)
        
        
    if len(runs) == 0:
        if v is True: print("no runs to load")
        return None
        
    t0 = now()
    if v is True: print(f"loading \33[1m\33[34m{target}\33[0m of {len(runs)} runs...")
    
    runs = rs(runs)
    data =  context.get_array(runs, target, config = config, **kwargs)
    t1 = now()
    if v is True: print(f"    data loaded: {len(runs)} entries in {t1-t0}")
    return(data)

c = {
    "s":context_sp,
    "d":context_dp,
}
experiments = {
    "s":"xebra_singlephase",
    "d": "xebra",
}


default_config = {'split_min_ratio': 1.5, 'split_min_height': 0, 'split_n_smoothing': 4}


energies_signals = {
    ("combined", "", 41.6),
    ("first", "1", 9.4),
    ("second", "2", 32.2),
}

def append_unique(df, fields, sep="__", uname = "unique"):
    '''
    modifies df directly!!
    '''
    out = False
    for f in fields:
        if out is False:
            out = df[f].astype(str)
        else:
            out = out + sep + df[f].astype(str)
    df[uname] = out



def make_unique_runs_dict(runs_all, fields):
    '''
    flattens the given array and merges multiple fields properties
    returns a dict where each unique fields combination contains all runs 
        with this combination
    '''
    df = make_dV_df(runs_all)
    append_unique(df, fields)
    
    uniques = {u:df.loc[df["unique"] == u]["run"].values for u in np.unique(df["unique"])}
    
    return(uniques)



def make_dV_dict(runs_all):
    _Vs = np.unique([x["fields"]["dV_Anode"] for x in runs_all.values()])
    runs_all_dVs = {dV:[r for r, x in runs_all.items() if x["fields"]["dV_Anode"] == dV] for dV in _Vs}
    return(runs_all_dVs)


def make_dV_df(runs_all):
    df = pd.DataFrame()
    for r, d in runs_all.items():
        d = {"run": r, **flatten_dict(d)}
        df = df.append(d, ignore_index=True)
        
    df = df.astype({
        'run': 'int32',
        'N': 'int32',
        'fields.adc': 'int32',
    })
    return(df)


def get_all_runs(query = False):
    '''
    runs_all, runs_all_dVs = mystrax.get_all_runs()
    
    parameter:
        - query: database query
    
    '''
    if query is False:
        query = {}
    db = list(mycol.find(query))
    
    runs_all = {db_i["run"]:db_i for db_i in db}


    runs_all_dVs  = make_dV_dict(runs_all)
    
    return(runs_all, runs_all_dVs)
    
    
    
    
# description for sp_krypton exits
DEVELOPER_fails={
    0: "no peaks in event",
    1: "less than 2 large peaks",
    2: "less than 3 large peaks",
    3: "first S1 area too large",
    4: "second S1 area too large",
    5: "decay time limit exceeded",
    6: "drift time limit exceeded",
    7: "first S2 smaller than first S1 (area)",
    8: "first S1 smaller than second S1 (area)",
}



def get_unit_sp(x):
    if x[:11] == "time_drift":
        return("µs")
    elif x[:11] == "time_decay":
        return("ns")
    elif x[:5] == "width":
        return("ns")
    elif x[:4] == "area":
        return("PE")
    elif x[:4] == "energy":
        return("keV")
    elif x[:2] == "cS":
        return("PE")
    else:
        return("")
 




print("\33[34mPython:\33[0m")
print(sys.executable)
print(sys.version)
print(sys.version_info)

folder_cache = "/data/storage/strax/cached/singlephase"
print(f"\33[32mImport done at {datetime.now()}\33[0m")



mycol = db["calibration_info"]

def db_dict():
    return(
        {f["run"]:f for f in mycol.find({})}
    )


def filter_peaks(peaks, sp_krypton_s1_area_min = 25, sp_krypton_max_drifttime_ns=500e3 ):
    ps = peaks[peaks["area"] >= sp_krypton_s1_area_min]
    idx = np.nonzero(np.diff(ps["time"]) <= sp_krypton_max_drifttime_ns)[0]
    idx = np.unique(np.append(idx, idx+1))
    return(ps[idx])


def draw_peaks(ax, ps, t0 = False, y_offset = 0, *args, **kwargs):
    if t0 is False:
        t0 = ps[0]["time"]
    elif t0 == "start":
        t0 = False
    for pi, p in enumerate(ps):
        draw_peak(ax, p, t0 = t0, y_offset = y_offset*pi, *args, **kwargs)

def draw_peak(ax, p, t0 = False, label="", show_area = False, label_peaktime = False, show_peaktime = False, 
    PE_ns = True, y_offset = 0, **kwargs):
    if t0 is False:
        t0 = p["time"]
    t_offs = p["time"]-t0
    y = np.trim_zeros(p["data"], trim = "b")
    if PE_ns is True:
        y = y / p["dt"]
    y = y + y_offset
    
    x = t_offs+np.arange(0, len(y))*p["dt"]
    props = []
    if show_area is True:
        props.append(f"{p['area']:.1f} PE")
    if label_peaktime is True:
        props.append(f"{(p['time']-t0)+p['time_to_midpoint']:.0f} ns")
    
    if len(props) > 0:
        props = f" ({', '.join(props)})"
    else:
        props = ""
    plt_i = ax.plot(x, y, label = f"{label}{props}", **kwargs)[0]
    if show_peaktime is True:
        ax.axvline((p['time']-t0)+p['time_to_midpoint'], color = plt_i.get_color())

def draw_gauss_peak(ax, p, t0 = False, show_pars = True, **kwargs):
    if show_pars is False:
        show_pars = []
    elif show_pars is True:
        show_pars = [*straxbra.plugins.GaussfitPeaks.props_fits]
    
    
    if t0 is False:
        t0 = p["time"]
    draw_peak(ax=ax, p=p, t0=t0, **kwargs)
    
    if "OK_fits" not in p.dtype.names:
        print("\33[31muse peaks of type 'gaussfit_peaks' for this function\33[0m")
        return(0)
    
    t_offs = p["time"]-t0
    y = np.trim_zeros(p["data"], trim = "b")
    x = t_offs+np.arange(0, len(y))*p["dt"]
    xp = np.linspace(0, max(x), 1000)
    
    
    for g, (label, f, fp0, fb, pars, units) in straxbra.plugins.GaussfitPeaks.props_fits.items():
        if p[f"OK_fit_{g}"] is np.True_:
            yf = f(xp, *p[f"fit_{g}"])
            res = p[f'sum_resid_sqr_fit_{g}']
            
            ax.plot(xp, yf, label = f"{label} gauss (res/ndf: {res:.1f})")
            if g in show_pars:
                for par, v, sv, u in zip(pars, p[f"fit_{g}"], p[f"sfit_{g}"], units):
                    fhist.add_fit_parameter(ax, par, v, sv, u)
    
    


def draw_kr_event(
    ax, event,
    show_peaks = "0123", label_area = True,
    show_peaktime = False, label_peaktime = False,
    PE_ns = True,
    leg_loc = False, t_ref = 0, t0 = False,
    **kwargs):
    '''
    plots S1s and S2s into ax
    
    parameters:
    - show peaks: string with numbers from 0 to 4 of whuich peaks to draw
      (default = "0123") ==> all 4 peaks are shown
    - show_peaktime:
      show midpoint ime of peak
    - PE_ns (True): convert peaks from PE/sample to PE/ns
    - label_peaktime:
        add midpoint time of peak to legend
    - leg_loc: set this to a valid legend_loc value to draw the legend,
      leave blank to not draw legend
    - t0: which time to use as reference. default False.
      if not given fallback to t_ref
    - t_ref: which peak to use for time reference
      (default: 0, so its the first S1)
    - **kwargs: used to format the plots
    
    '''
    if t0 is False:
        if t_ref in [0, 1, 2, 3]:
            t0 = event["time_signals"][t_ref]
        else:
            t0 = 0
    
    # extra offset in case t0 is not first peaks time
    t_offset_abs = event["time_signals"][0]-t0
    
    
    if event["s2_split"]:
        labels = ["first S1", "second S1", "first S2", "second S2"]
    else:
        labels = ["first S1", "second S1", "S2"]
    
    for peak_i, (p_data, t_peak, t_peak_in_event_time, label) in enumerate(zip(event["data_peaks"], event["time_signals"], event["time_peaks"], labels)):
        if str(peak_i) in show_peaks:
            t_offs = t_peak-t0
            y = np.trim_zeros(p_data, trim = "b")
            if PE_ns is True:
                y = y/event["dt"][peak_i]
            x = t_offs+np.arange(0, len(y))*event["dt"][peak_i]
            area = event[f"area_s{(peak_i>1)+1}{peak_i%2+1}"]
            
            props = []
            if label_area is True:
                props.append(f"{area:.1f} PE")
            if label_peaktime is True:
                props.append(f'{event["time_peaks"][peak_i]:.0f} ns')
            if len(props) > 0:
                props = f" ({', '.join(props)})"
            else:
                props = ""
            plt_i = ax.plot(x, y, label = f"{label}{props}", **kwargs)[0]
            
            if show_peaktime is True:
                ax.axvline(event["time_peaks"][peak_i] + t_offset_abs, color = plt_i.get_color())

    if PE_ns is True:
        ax.set_ylabel("signal / PE/ns")
    else:
        ax.set_ylabel("signal / PE/sample")
    ax.set_xlabel("time / ns")

    if leg_loc is not False:
        try:
            ax.legend(loc = leg_loc)
        except BaseException as e:
            print(e)






def draw_event(ax, event, peaks = False, show_peaks = "0123", show_peaktime = False, leg_loc = False, t_ref = 0, **kwargs):
    '''
    plots S1s and S2s into ax
    old version that requires dedicated peaks
    
    parameters:
    - show peaks: string with numbers from 0 to 4 of whuich peaks to draw
      (default = "0123") ==> all 4 peaks are shown
    - leg_loc: set this to a valid legend_loc value to draw the legend,
      leave blank to not draw legend
    - t_ref: which peak to use for time reference
      (default: 0, so its the first S1)
    
    - **kwargs: used to format the plots
    
    '''
    
    if peaks is False:
        raise ValueError("\33[31mare you sure you want to use 'draw_event' instead of 'draw_kr_event'\33[0m?")
    
    
    
    if t_ref in [0, 1, 2, 3]:
        t0 = event["time_signals"][t_ref]
    else:
        t0 = 0
    
    peaks_event = get_peaks_by_timestamp(peaks, event["time_signals"])
    
    
    
    if event["s2_split"]:
        labels = ["first S1", "second S1", "first S2", "second S2"]
    else:
        labels = ["first S1", "second S1", "S2"]
    
    for peak_i, (peak , label) in enumerate(zip(peaks_event, labels)):
        if str(peak_i) in show_peaks:
            y = np.trim_zeros(peak["data"])
            x = peak["time"]-t0+np.arange(0, peak["dt"]*len(y), peak["dt"])
            plt_i = ax.plot(x, y, label = f"{label} ({peak['area']:.1f} PE)", **kwargs)[0]
            if show_peaktime is True:
                ax.axvline(event["time_peaks"][peak_i]-t0, color = plt_i.get_color())

    ax.set_ylabel("signal / PE/sample")
    ax.set_xlabel("time / ns")

    if leg_loc is not False:
        try:
            ax.legend(loc = leg_loc)
        except BaseException as e:
            print(e)



def draw_peak_fits(p, fname = False, highlight_best_peak = True, best_fit_i = "auto", title = ""):
    '''
        plots all fits for given gaussfitpeak
    '''
    props_fits = straxbra.plugins.GaussfitPeaks.props_fits

    fig, axs = fhist.make_fig(n_tot = len(props_fits))
    
    if title != "":
        fig.suptitle(title)
    
    xp = np.linspace(0, p["dt"]*p["length"], 1000)
    for i, (ax, (l, (title, f, p0_f, b_f, pars, units))) in enumerate(zip(axs, props_fits.items())):
        if best_fit_i == "auto":
            best_fit_i = p["best_fit"]
        if (i+1 == best_fit_i) and (highlight_best_peak is True):
            ax.set_title(f"{title} (best fit)")
        else:
            ax.set_title(title)
        ax.set_xlabel("time / ns")
        ax.set_ylabel("signal / PE")

        color = ax.plot([])[0].get_color()
        draw_peak(ax, p, show_area = True, label = "data", color = color)
        tmid = p["time_to_midpoint"]
        ax.axvline(tmid, label = f"$t_\\mathrm{{midpoint}}$: {tmid:.1f} ns", color = color, linestyle = "dashed")


        fit = p[f"fit_{l}"]
        sfit = p[f"sfit_{l}"]
        res = p[f"sum_resid_sqr_fit_{l}"]

        yf = f(xp, *fit)

        ax.plot(xp, yf, label = f"fit ($\\mathrm{{res}}^2_\\mathrm{{red}}$.:{res:.3g})")


        for par, v, sv, u in zip(pars, fit, sfit, units):
            fhist.add_fit_parameter(ax, par, v, sv, u)
        ax.legend(loc = "upper right")

    plt.subplots_adjust(
        hspace = .3,
        wspace = .2,
        left = .05,
        right = .98,
        top = .95,
        bottom = .075,
    )
    if fname is False:
        plt.show()
    else:
        plt.savefig(fname)
        plt.close()
    





def get_calibration_data(run):
    run = int(run)
    
    myquery = { "run":  run}
    mydoc = list(mycol.find(myquery))
    return(mydoc)


def get_peaks_by_timestamp(peaks, timestamps):
    return(peaks[np.in1d(peaks["time"], timestamps)])

def get_peaks_by_event(peaks, event):
    return(peaks[
          (peaks["time"] >= event["time"])
        & (peaks["time"] <= event["endtime"])
        
    ])

def get_krskru(kr, bools = True):
    '''
    returns split and unsplit events for easy access
    
    use:
    krs, kru = get_krskru(kr)
    
    '''
    if bools is True:
        krs = kr[kr["s2_split"]]
        kru = kr[~kr["s2_split"]]
    else:
        krs = kr[bools]
        kru = kr[~bools]

    return(krs, kru)


def get_min_peaks(kr, merge = True):
    '''
    returns events that have the minimum number of large peaks
    (split = 4, unsplit = 3)
    
    set parameter to False if krs and kru should be returend seperately
    
    '''
    
    
    krs, kru = get_krskru(kr)
    krs = krs[krs["n_peaks_large"] == 4]
    kru = kru[kru["n_peaks_large"] == 3]

    if merge is True:
        return(
            np.append(krs, kru)
        )
    return(krs, kru)
    

@flo_decorators.silencer
def load_run_kr(runs, config = False, mconfig = None, gs = False, W = 13.5, return_peaks = False, context = context_sp, return_db = False, correct = True, filter_events = True, *args, **kwargs):
    '''
    returns sp_krypton and calibration data of runs
    
    parameters:
    config (False): if given, this config is used instead of the default custom config
    mconfig (False): modifications to the default config when loading data, overrules default parameters
        use this if you just want to add/modify a few paramters and do not want to take care of default config
    
    gs (False): tuple of g1 and g2, if given, this is used to calculate the energy of the peak
    W (13.5): required to calculate the Energy with g1 and g2
    return_peaks (False): whether or not to also return the peaks of the runs
        (WARNING: this might take some time as all peaks have to be loaded first and are later filtered before returning to save RAM.
        the event peak data is also stored in sp_krypton though, so this should not be needed anymore)
    context (context_sp): which context to laod data from
    correct (True):
        Whether or not to apply the S1 and S2 corrections (if) found in the database
    filter_events (True):
        whether or not to remove events that have "is_event" set to False
        
    '''
    
    
    if "calibrate" in kwargs:
        print("\33[41mlegacy parameter 'calibrate' was used. Use 'correct' instead\33[0m")
        correct = calibrate
    if "peaks" in kwargs:
        print("\33[41mlegacy parameter 'peaks' was used. Use 'return_peaks' instead\33[0m")
        return_peaks = peaks
    
    
    if mconfig is None:
        mconfig = {}
        
    
    
    
    
    t_start = datetime.now()
    if config is False:
        config = default_config
    
    config = {**config, **mconfig}
    
    
    
    if isinstance(runs, int):
        runs = [runs]
    runs_str = [f"{r:0>5}" for r in runs]

    print(runs_str)

    print("start loading data")    
    
    
    print("  \33[34mconfig:\33[0m")
    for key, value in config.items():
        print(f"    \33[35m{key}:\33[0m {value}")
    
    
    sp = context.get_array(runs_str, "sp_krypton", config = config)
    print("loading done")
    if filter_events:
        sp = sp[sp["is_event"]]
    
    calibration = False
        
    
    
    db = list(mycol.find({"run":{"$in": runs}}))
    db_dict = {db_i["run"]:db_i for db_i in db}

        
    
    if correct is True:
        calibration = db_dict
        if "run_id" not in sp.dtype.names:
            # correct only one
            calibration = db[0]
            print("correcting single run: drifttime", end = "")
            correct_drifttime(sp, tpc_corrections = calibration["tpc_corrections"])
            print(", S1", end = "")
            correct_s1(sp, tpc_corrections = calibration["tpc_corrections"], corr_pars = calibration["s1_corr_pars"])
            print(", S2", end = "")
            correct_s2(sp, lifetime = calibration["elifetime"])
            
        else:
            runs_found = np.unique(sp["run_id"])
            # correct multiple
            print("correcting multple runs", end = ": ")

            out = None
            for run in runs_found:
                print(f"\n  - {run}:", end = "")
                try:
                    sp_ = sp[sp["run_id"] == run]
                    calibration = db_dict[run]
                    print(" dt", end = "")
                    correct_drifttime(sp_, tpc_corrections = calibration["tpc_corrections"])
                    print(", S1", end = "")
                    correct_s1(sp_, tpc_corrections=calibration["tpc_corrections"], corr_pars=calibration["s1_corr_pars"])
                    print(", S2", end = "")
                    correct_s2(sp_, lifetime = calibration["elifetime"])
                    if out is None:
                        out = sp_
                    else:
                        out = np.append(out, sp_)
                except Exception as e:
                    print(f" \33[31mERROR: {e}\33[0m")
            sp = out
        if gs is not False:
            print("calculating energy")
            sp["energy_s1"] = sp["cS1"] / gs[0] * W
            sp["energy_s2"] = sp["cS2"] / gs[1] * W
            sp["energy_total"] = sp["energy_s1"] + sp["energy_s2"]
        
    print("\ncorrecting done")
    
    
    out = []
    if return_db is True:
        out.append(db_dict)
    
    
    if return_peaks is True:
        print("start loading peaks")
        peaks = context.get_array(runs_str, "peaks", config = config)
        print("loading done, filtering")
        peaks = peaks[np.in1d(peaks["time"], sp["time_large_peaks"].reshape(-1))]
        print("filtering done")
        out.append(peaks)
    
    t_end = datetime.now()
    print(f"all done in {t_end - t_start}")
        
    if len(out) > 0:
        return(sp, *out)
    else:
        return(sp)





def load_run_kr_quick(*args, max_time = 10, **kwargs):
    ret = False
    
    def wrapper(*args, **kwargs):
        nonlocal ret
        try:
            ret = load_run_kr(*args, **kwargs)
        except Exception as e:
            ret = e
        
    
    
    action_thread = Thread(target=wrapper, args = args, kwargs=kwargs)

    action_thread.start()
    action_thread.join(timeout = max_time)
    
    
    return(ret)



def make_todo(kr):
    '''
        returns kru, krs, todo list
    '''
    
    krs = kr[kr["s2_split"]]
    kru = kr[~kr["s2_split"]]


    todo = (
        ("s1", "combined S1", kr),
        ("s11", "first S1", kr),
        ("s12", "second S1", kr),
        
        ("s2", "unsplit S2", kru),
        ("s21", "first S2", krs),
        ("s22", "second S2", krs),
    
    )
    

    return(kru, krs, todo)