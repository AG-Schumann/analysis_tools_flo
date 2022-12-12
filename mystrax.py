import sys
import os
from datetime import datetime
from flo_analysis import *
import flo_decorators
from threading import Thread, Event

# import 
sys.path.insert(0,"/data/workspace/Flo/straxbra_flo/strax")
import strax
print("\33[34mStrax:\33[0m")
print(f"Strax version: {strax.__version__}")
print(f"Strax file:    {strax.__file__}")

sys.path.insert(0,"/data/workspace/Flo/straxbra_flo/straxbra")
import straxbra
print("\33[34mStraxbra:\33[0m")
print(f"straxbra file:    {straxbra.__file__}")
print(f"straxbra version: {straxbra.__version__}")
print(f"SpKrypton version:    {straxbra.plugins.SpKrypton.__version__}")

context_sp = straxbra.SinglePhaseContext()
context_dp = straxbra.XebraContext()
db = straxbra.utils.db


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


def draw_peak(ax, p, t0 = False, label="", show_area = False, label_peaktime = False, show_peaktime = False, **kwargs):
    if t0 is False:
        t0 = p["time"]
    t_offs = p["time"]-t0
    y = np.trim_zeros(p["data"], trim = "b")
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

def draw_gauss_peak(ax, p, t0 = False, **kwargs):
    draw_peak(ax=ax, p=p, t0=t0, **kwargs)
    if "OK_fit_s" not in p.dtype.names:
        print("\33[31muse peaks of type 'gaussfit_peaks' for this function\33[0m")
        return(0)
    
    if t0 is False:
        t0 = p["time"]
    t_offs = p["time"]-t0
    y = np.trim_zeros(p["data"], trim = "b")
    x = t_offs+np.arange(0, len(y))*p["dt"]
    xp = np.linspace(0, max(x), 1000)
    
    
    for g, label, f in [
            ("s", "single", straxbra.plugins.GaussfitPeaks.sg),
            ("d", "double", straxbra.plugins.GaussfitPeaks.dg)
    ]:
        if p[f"OK_fit_{g}"] is np.True_:
            pars, units = straxbra.plugins.GaussfitPeaks.props_fits[g]
            
            yf = f(0, xp, *p[f"fit_{g}"])
            res = p[f'sum_resid_sqr_fit_{g}']
            
            ax.plot(xp, yf, label = f"{label} gauss (res/ndf: {res:.1f})")
            for par, v, sv, u in zip(pars, p[f"fit_{g}"], p[f"sfit_{g}"], units):
                fhist.add_fit_parameter(ax, par, v, sv, u)
    
    


def draw_kr_event(ax, event, show_peaks = "0123", label_area = True, show_peaktime = False, label_peaktime = False, leg_loc = False, t_ref = 0, t0 = False, **kwargs):
    '''
    plots S1s and S2s into ax
    
    parameters:
    - show peaks: string with numbers from 0 to 4 of whuich peaks to draw
      (default = "0123") ==> all 4 peaks are shown
    - show_peaktime:
      show midpoint ime of peak
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




    





def get_calibration_data(run):
    run = int(run)
    
    myquery = { "run":  run}
    mydoc = list(mycol.find(myquery))
    return(mydoc)


def get_peaks_by_timestamp(peaks, timestamps):
    return(peaks[np.in1d(peaks["time"], timestamps)])

def get_krskru(kr):
    '''
    returns split and unsplit events for easy access
    
    use:
    krs, kru = get_krskru(kr)
    
    '''
    krs = kr[kr["s2_split"]]
    kru = kr[~kr["s2_split"]]

    return(kr[kr["s2_split"]], kr[~kr["s2_split"]])


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
                    print("\33[31mERROR: {e}\33[0m")
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