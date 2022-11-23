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





def draw_kr_event(ax, event, show_peaks = "0123", show_peaktime = False, leg_loc = False, t_ref = 0, **kwargs):
    '''
    plots S1s and S2s into ax
    
    parameters:
    - show peaks: string with numbers from 0 to 4 of whuich peaks to draw
      (default = "0123") ==> all 4 peaks are shown
    - leg_loc: set this to a valid legend_loc value to draw the legend,
      leave blank to not draw legend
    - t_ref: which peak to use for time reference
      (default: 0, so its the first S1)
    
    - **kwargs: used to format the plots
    
    '''
    
    if t_ref in [0, 1, 2, 3]:
        t0 = event["time_signals"][t_ref]
    else:
        t0 = 0
    
    
    if event["s2_split"]:
        labels = ["first S1", "second S1", "first S2", "second S2"]
    else:
        labels = ["first S1", "second S1", "S2"]
    
    for peak_i, (p_data, t_peak, label) in enumerate(zip(event["data_peaks"], event["time_signals"], labels)):
        if str(peak_i) in show_peaks:
            t_offs = t_peak-t0
            y = np.trim_zeros(p_data, trim = "b")
            x = t_offs+np.arange(0, len(y))*event["dt"][peak_i]
            area = event[f"area_s{(peak_i>1)+1}{peak_i%2+1}"]
            plt_i = ax.plot(x, y, label = f"{label} ({area:.1f} PE)", **kwargs)[0]
            if show_peaktime is True:
                ax.axvline(event["time_peaks"][peak_i]+t_offs, color = plt_i.get_color())

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



@flo_decorators.silencer
def load_run_kr(runs, config = False, gs = False, W = 13.5, peaks = False, context = context_sp, return_db = False, correct = True, *args, **kwargs):
    '''
    returns sp_krypton and calibration data of runs
    
    parameters:
    config (False): if given, this config is used instead of the default custom config
    gs (False): tuple of g1 and g2, if given, this is used to calculate the energy of the peak
    W (13.5): required to calculate the Energy with g1 and g2
    peaks (False): whether or not to also return the peaks of the runs
        (WARNING: this might take some time as all peaks have to be loaded first and are later filtered before returning to save RAM.
        the event peak data is also stored in sp_krypton though, so this should not be needed anymore)
    context (context_sp): which context to laod data from
    correct (True):
        Whether or not to apply the S1 and S2 corrections (if) found in the database
    '''
    
    if "calibrate" in kwargs:
        print("\33[41mlegacy parameter 'calibrate' was used\33[0m")
        correct = calibrate
    
    
    
    
    
    
    t_start = datetime.now()
    if config is False:
        config = default_config
    
    
    if isinstance(runs, int):
        runs = [runs]
    runs_str = [f"{r:0>5}" for r in runs]

    print(runs_str)

    print("start loading data")    
    sp = context.get_array(runs_str, "sp_krypton", config = config)
    print("loading done")
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
            print("correcting multple runs", end = ", ")

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
    
    
    if peaks is True:
        print("start loading peaks")
        peaks = context.get_array(runs_str, "peaks", config = config)
        print("loading done, filtering")
        peaks = peaks[np.in1d(peaks["time"], sp["time_signals"].reshape(-1))]
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