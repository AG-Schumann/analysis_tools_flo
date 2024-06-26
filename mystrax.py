import sys
import os
from datetime import datetime
from flo_analysis import *
import flo_fancy

import flo_decorators
from threading import Thread, Event
import matplotlib.pyplot as plt
import matplotlib as mpl
import mystrax_corrections as msc
import get_corrections as gc

def now():
    return(datetime.now())

mystrax_limporttime = now()

def age():
    print(f"strax loaded from \33[1m{straxpath}\33[0m at {mystrax_limporttime} (age: {now() - mystrax_limporttime})")



tcol = "\33[36m"

if "straxpath" in os.environ:
    straxpath = os.environ["straxpath"]
    print(f"found straxpath envornmental variable: {straxpath}")
else:
    straxpath = "/data/workspace/Flo/straxbra_flo"


# import 
sys.path.insert(0,f"{straxpath}/strax")
import strax

sys.path.insert(0,f"{straxpath}/straxbra")
import straxbra


eff = straxbra.plugins.eff
context_sp = straxbra.SinglePhaseContext()
context_dp = straxbra.XebraContext()
db = straxbra.utils.db

# database collections
mycol = db["calibration_info"]
corr_coll = db["correction_info"]



labels = gc.labels

fields_df = gc.fields_df



def time_from__id(_id):
    return(datetime.fromtimestamp(int(str(_id)[:8], 16)))




def get_field(ds, field, prio_identical = True):
    '''
    returns the field of ds that start with field
    '''
    lf = len(field)
    
    if (field in ds.dtype.names) and (prio_identical is True):
        return(field)
    
    fields = [n for n in ds.dtype.names if n[:lf]==field]
    if len(fields) == 1:
        return(fields[0])
    else:
        return(fields)


def s1_ratios(ds):
    field = get_field(ds, "areas")
    ratios = ds[field][:,0]/ds[field][:,1]
    return(ratios)

def draw_multi_woa_multi(ds, axs, draw_list=False, plugin = "event_fits_summary", titles = False, **kwargs):
    '''
    wrapper fucntion to draw all three default woa multis of ds into axs
    '''
    if draw_list is False:
        draw_list = ["0123", "45", "67"]
        if titles is False:
            titles = ["", "", ""]
        
    for ax, draw, title in zip(axs, draw_list, titles):
        draw_woa_multi(ds, ax = ax, draw = draw, plugin = plugin, title = title, **kwargs)

def draw_woa_single(
    a, w, ax = False,
    label = "Signal",
    cmap = "Purples",
    show_counts = True,
    setup = True,
):
    '''
    
    cmap:
        'Purples', 'Reds', 'Blues','Greens','Oranges'
    '''

    if not isinstance(ax, plt.Axes):
        raise TypeError("ax must be plt.Axes")
        
    count, bca, bcw = fhist.make_2d_hist_plot(a, w)
    n_count = np.nansum(count, dtype = int)
    count_ = count / n_count
    
    
    if show_counts is True:
        label = f"{label} (N: {n_count})"
    im_ = ax.pcolormesh(
        bca, bcw,
        count_.T,
        norm=fhist.LogNorm(),
        cmap = cmap,
    )

    if setup is True:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(fhist.defaults["2d_hist_label_area"])
        ax.set_ylabel(fhist.defaults["2d_hist_label_width"])


    color = mpl.cm.get_cmap(cmap)(.75)
    if label != "":
        ax.plot(
            [],
            "o",
            color = color,
            label = label
        )
        
        ax.legend(loc = "upper right")
    return(color, n_count)


def draw_woa_scatter(
    ds,
    ax = False,
    draw = "0123",
    plugin = "event_fits_summary",
    labels_ = False,
    field_a = "areas_corrected",
    field_w = "widths",
    colors = False, alpha = .01,
    marker = ".",
    rasterized = True,
):

    if ax is False:
        ax = fhist.ax()

    if colors is False:
        colors = fhist.default_colors
    
    if labels_ is False:
        labels_ = labels[plugin]
        
        
    for i_draw, field_id_str, color in enumezip(draw, colors):
        field_id = int(field_id_str)
        label = labels_[field_id]
        w = ds[field_w][:, field_id]
        a = ds[field_a][:, field_id]
        addlabel(ax, f"{label}", marker = "o", color = color )
        ax.plot(
            a, w,
            linestyle = "",
            marker = marker, markeredgewidth = 0,
            alpha = alpha, color = color,
            rasterized = rasterized
        )
    
    ax.set_xlim(10,1e5)
    ax.set_ylim(10,1e4)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("area / PE")
    ax.set_ylabel("width / ns")

    ax.legend(loc = "upper right")
    ax.grid()


def draw_woa_multi(
    ds, ax = False, draw = "0123", plugin = "event_fits_summary", add_grid = True, show_only_max_count = True,
    vmin = 1, global_scale = False, show_counts = True, title = "", show_precut = False,
    labels_ = False, field_a  = "auto", field_w  = "auto", cmaps = False, cmap_cb = False):
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
    
    if field_a  == "auto":
        field_a = get_field(ds, "areas")
        
    if field_w  == "auto":
        field_w  = get_field(ds, "widths")
    
    
    if (vmin is False) or (vmin < 1):
        vmin = 1
    
    if labels_ is False:
        labels_ = labels[plugin]
    
    draw = [int(i) for i in draw if int(i) < len(ds[field_a][0])]
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
        ax.set_axis_off()
        return(None)
        # raise ValueError("maxmimum of counts is zero!")
    

    if cmaps is False:
        cmaps = [
            'Purples',
            'Reds',
            'Blues',
            'Greens',
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


def jr(runs, sep = ", "):
    '''
    joins list of anything (maily ints of runnumbers) to list
    '''
    return(sep.join(map(str, runs)))


def rs(runs):
    '''
    turns runs into a list of run strings (zero padded to five digits) to be used when loading data
    '''
    
    if isinstance(runs, int):
        runs = [runs]
    elif isinstance(runs, str):
        runs = [runs]

    
    rstrs = [f"{r:0>5}" for r in runs]
    
    return(rstrs)


def find_config(target = ""):
    if target[:10] == "sp_krypton":
        config = default_config
    else:
        config = {}
    return(config)





def get_correction_for(typ = False, start = False, *_, plugin = False, run = False, limit = 1, v = False):
    '''
    returns correcions for types various types:
        get the types by calling this funciotn witout an argument
        
    parameters: 
        * typ: type of correction see above
        * start (now()): time of run, correction must be older than this
        * limit (1): how many entries to return
        * v (False): 'verbose', prints query parameters
        
    '''
    valid_types = list(msc.corrections)
    
    if typ is True:
        print(f"valid types: {valid_types}")
    if typ is True or typ is False:
        return(valid_types)
    
    if typ not in valid_types:
        raise TypeError(f"typ must be in ({valid_types})")

    if start is False:
        start = datetime.now()
    
    
    
    filters = [{"type": typ}]
    
    
    if run is not False:
        filters.append({
            "$or": [
                {"run":run},
                {"$and":[
                    {"run": {"$exists": False}},
                    {"date": {"$lte": start}},
                ]}
            ]
        })
    else:
        filters.append({"date": {"$lte": start}})

    if isinstance(plugin, str):
        filters.append({
            "$or": [
                {"plugin": {"$exists": False}},
                {"plugin": plugin},
            ]
        })
    
    
    filters = {"$and": filters}
    
    
    
    correction = list(corr_coll.find(
        filters,
        sort = [('_id', -1)],
        limit = limit,#
    ))
    
    
    if (limit == 1) and len(correction) == 1:
        correction = correction[0]
    return(correction)









def fiduzalize_z(ds, tpc_geometry, id_bool = False, sigmas = 1):
    
    if not isinstance(id_bool, np.ndarray):
        id_bool = np.array([True] * len(ds))
    
    if "info" in tpc_geometry:
        tpc_geometry = tpc_geometry["info"]
    
    fiducal_dft = (
        tpc_geometry["dft_gate"] + sigmas * tpc_geometry["sigma_dft_gate"],
        tpc_geometry["dft_cath"] - sigmas * tpc_geometry["sigma_dft_cath"],
    )
    
    
    outside_fid_volume_bool = (
          (ds["drifttime"] < fiducal_dft[0])
        | (ds["drifttime"] > fiducal_dft[1])
    )
    set_zero_bool = id_bool * outside_fid_volume_bool
    ds["z"][set_zero_bool] = 0
    
    ds = ds[~set_zero_bool]
    return(ds)



def check(runs, target, context = False, config = False, cast_int = True, v = True):
    
    if context is False:
        context = context_sp
    if config is False:
        config = find_config(target)
        
    runs = rs(runs)
    
    t0 = now()
    if v is True: print(f"checking \33[1m{tcol}{target}\33[0m for {len(runs)} runs...")
    runs = [r for r in runs if context.is_stored(r, target, config = config)]
    t1 = now()
    if cast_int is True:
        runs = list(map(int, runs))
    if v is True: print(f"    found data for {len(runs)} runs in {t1-t0}")
    return(runs)






def load(
    runs, target,
    context = False,
    config = False,
    mconfig = None,
    check_load = True,
    v = True,
    correct = True,
    filters = True,
    fidu_z = False,
    **kwargs
):
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
        mconfig (None): modifiying automatic config
            (overwrites default configs settings if names match)
        check load: check if the data is available before attmepting to load it
            use a string of a different target to check the existance of that data to load this data
            set to false if you want to skip all checks
            default True
        v: verbose toggle, prints messages
            default True
        filters (True): if True: looks for fields like OK and clean and returns only fields where these fields are True
            if list of strings: uses list entires and checks them
        fidu_z (0): if this is a number a fidicilisatzion if z will be done (only if corrections are applied)
            the number is the measure for how many sigmas we move inwards to the main volume
            (based on drift time position of gate and cathode)
    check_load
    
    '''
    
    if context is False:
        context = context_sp
    if config is False:
        config = find_config(target)
    
    if isinstance(mconfig, dict):
        config = {**config, **mconfig}
    
    if v is True:
        print("config:")
        print(config)
    
    
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
    if v is True: print(f"loading \33[1m{tcol}{target}\33[0m of {len(runs)} runs...")
    
    runs = rs(runs)
    data =  context.get_array(runs, target, config = config, **kwargs)
    t1 = now()
    if v is True: print(f"    data loaded: {len(data)} entries for {len(runs)} runs in {t1-t0}")
    
    
    
    if filters is True:
        filters = ["OK", "clean"]

        
    dtype_names = data.dtype.names
    if isinstance(filters, list):
        filters = [f for f in filters if (isinstance(f, str) and f in dtype_names)]
        if v is True: qp(f"filtering: ")
        for filter_i in filters:
            if v is True: qp(f"{len(data)} -(\33[1m{tcol}{filter_i}\33[0m)-> ")
            data = data[data[filter_i] == True]
        if v is True: print(f"{len(data)}")
        
    
    run_ids = list(map(int, runs))
    if correct is True:
        correct = get_correction_for()
    if (target[-8:] == "_summary") and isinstance(correct, list):
        print("correcting")
        runs_info = list(db.runs.find({
            "experiment": context.config["experiment"],
            "run_id": {"$in": run_ids}
        }))
        runs_info = {x["run_id"]:x for x in runs_info}

        for run_id in run_ids:
            if len(run_ids) == 1:
                bool_run = np.array([True]*len(data))
            else:
                bool_run = data["run_id"] == run_id

            start_run = runs_info[run_id]["start"]
            qp(f"  * {run_id} ({start_run.strftime('%H:%M:%S %d.%m.%Y')}):")
            
            tpc_info = get_correction_for("gate_cathode", start_run)["info"]
            pre_string = ""
            for corection_type in correct:
                qp(f"{pre_string} {corection_type}")
                corr = get_correction_for(corection_type, start_run, plugin = target, run = run_id)
                if "run" in corr:
                    qp(f" ({corr['run']})")
                info = corr["info"]
                msc.corrections[corection_type](data, info, tpc_info, bool_run)
                pre_string = ","
        
                
        
                # why is bool a subclass of int.......
            if not isinstance(fidu_z, bool) and isinstance(fidu_z, (int, float)):
                data = fiduzalize_z(data, tpc_info, id_bool = bool_run, sigmas = fidu_z)
                qp(f", fidu_z ({len(data)})")
        
            print(", done")
    
        print("correcting done")
    
    print(f"\n\33[1m{tcol}data is ready\33[0m")
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


def make_df(runs_all, dict_types = True, *args, **kwargs):
    df = pd.DataFrame()
    t0 = min([x["start"] for r, x in runs_all.items()])
    
    for r, d in runs_all.items():
        d = {
            "run": r,
            "t_rel": (d["start"] - t0).total_seconds(),
            **flatten_dict(d, *args, **kwargs),
            
        }
        df = df.append(d, ignore_index=True)
    
    
    
    if dict_types is True:
        # add here freely, only existing columns will be renamed
        dict_types = {
            'run': 'int32',
            'N': 'int32',
            'fields.adc': 'int32',
        }
    
    
        
    if isinstance(dict_types, dict):
        columns_names = df.columns.values
        
        dict_types = {
            n:v
            for n, v in dict_types.items()
            if n in columns_names
        }
        
        df = df.astype(dict_types)
    
    return(df)


def get_all_runs(query = False, *args, **kwargs):
    '''
    runs_all, runs_all_dVs = mystrax.get_all_runs()
    
    parameter:
        - query: database query
    
    '''
    if query is False:
        query = {}
    db = list(mycol.find(query))
    
    runs_all = {db_i["run"]:db_i for db_i in db}


    runs_all_df  = make_df(runs_all, *args, **kwargs)
    
    return(runs_all, runs_all_df)
    
    
    
    
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
 





folder_cache = "/data/storage/strax/cached/singlephase"







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
    no_labels = False, show_peaks = "0123", label_area = True,
    show_peaktime = False, label_peaktime = False,
    PE_ns = True,
    leg_loc = False, t_ref = 0, t0 = False,
    yoffset = 0,
    **kwargs):
    '''
    plots S1s and S2s into ax
    
    parameters:
    - no_labels: prevents the creaation of any labels
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
            if no_labels is True:
                label_ = ""
            else:
                label_ = f"{label}{props}"
            plt_i = ax.plot(x, y+yoffset, label = label_, **kwargs)[0]
            
            if show_peaktime is True:
                ax.axvline(event["time_peaks"][peak_i] + t_offset_abs, color = plt_i.get_color())

    if PE_ns is True:
        ax.set_ylabel("Signal [PE/ns]")
    else:
        ax.set_ylabel("Signal [PE/sample]")
    ax.set_xlabel("Time [ns]")

    if leg_loc is not False:
        try:
            ax.legend(loc = leg_loc)
        except BaseException as e:
            print(e)





def draw_kr_event_split(
    ev, title = "", evs = False, same_xscale = True, show_areas = True, show_decaytime = False,
    style_peak_fit = dict(),
    style_wave_fit = dict(),
):

    fig, axs = plt.subplots(1, 2, sharey = True)
    ax1, ax2 = axs
    fig.suptitle(f"{title}")



    axl = ax1.twiny()
    
    
    
    if show_decaytime is True:
        addlabel(axl, f"decay: {ev['time_decay_s1']:.0f} ns")
    
    axl.set_axis_off()

    if ev["s2_split"]:
        labels = ["first S1", "second S1", "first S2", "second S2"]
    else:
        labels = ["first S1", "second S1", "S2"]

        
    time_S21 = ev["time_signals"][2] - ev["time_signals"][0]
    

    draw_kr_event(ax1, ev, show_peaks = "0", color = fhist.default_colors[0], linewidth = 2)
    draw_kr_event(ax1, ev, show_peaks = "1", color = fhist.default_colors[1], linewidth = 2)

    draw_kr_event(ax2, ev, show_peaks = "2", color = fhist.default_colors[2], linewidth = 2)
    
    if ev["s2_split"]:
        draw_kr_event(ax2, ev, show_peaks = "3", color = fhist.default_colors[3], linewidth = 2)
    ax2.get_yaxis().set_visible(False)

    
    if isinstance(evs, np.void) and ("fit" in evs.dtype.names):
        
        time_S21 = ev["time_signals"][2] - ev["time_signals"][0]
        
        
        w_s21 = np.sum(ev["data_peaks"][2]!=0) * ev["dt"][2]
        
        t_end_s2 = time_S21 + w_s21
        
        if ev["s2_split"]:
            w_s22 = np.sum(ev["data_peaks"][3]!=0) * ev["dt"][3]
            time_S22 = ev["time_signals"][3] - ev["time_signals"][0]
            t_end_s2  = time_S22 + w_s22

        xp_s2 = np.linspace(time_S21, t_end_s2, 2000)


        fit = evs["fit"]*1
        
        fit[0] = fit[0]+time_S21
        
        fit_S21 = fit*1
        fit_S22 = fit*1
        fit_S21[4] = 0
        fit_S22[3] = 0
        
        y_s21 = eff.sum_gauss(xp_s2, *fit_S21)
        y_s22 = eff.sum_gauss(xp_s2, *fit_S22)
        y_s2 = eff.sum_gauss(xp_s2, *fit)

        ax2.plot(xp_s2, y_s21, color = fhist.default_colors[0], **style_peak_fit)
        ax2.plot(xp_s2, y_s22, color = fhist.default_colors[1], **style_peak_fit)
        ax2.plot(xp_s2, y_s2, **dict(color = "black", **style_wave_fit))
        a21 = evs["areas"][2] * np.exp(evs["drifttime"] / 55)
        a22 = evs["areas"][3] * np.exp(evs["drifttime"] / 55)
        

        addlabel(axl, f"1 st. S2: {a21:.0f} PE")
        addlabel(axl, f"2 nd. S2: {a22:.0f} PE")
    
    
    
    for peak_i, label, color in enumezip(labels, fhist.default_colors):
        if show_areas is True:
            area = ev[f"area_s{(peak_i>1)+1}{peak_i%2+1}"]
            label = f"{label} ({area:.1f} PE)"
        addlabel(axl, label, color = color, linestyle = "-")


    ax2.set_xlabel("")
    ax1.xaxis.set_label_coords(1,-0.1)
    
    xlim_1 = ax1.get_xlim()
    xlim_2 = ax2.get_xlim()

    w1 = (xlim_1[1]-xlim_1[0])/2
    w2 = (xlim_2[1]-xlim_2[0])/2

    m1 = xlim_1[0] + w1
    m2 = xlim_2[0] + w2

    w = max(w1, w2)
    if same_xscale is True:
        ax1.set_xlim(m1-w, m1+w)
        ax2.set_xlim(m2-w, m2+w)
    
    
    t1 = ax1.get_xticks()
    t2 = ax2.get_xticks()
    tl1 = ax1.get_xticklabels()
    tl2 = ax2.get_xticklabels()

    for ti in np.nonzero(t1 > (m1 + w*.8))[0]:
        plt.setp(tl1[ti], visible=False)
    for ti in np.nonzero(t2 < (m2 - w*.95))[0]:
        plt.setp(tl2[ti], visible=False)

    
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_color("grey")
    
#     axl.legend(loc = "upper left")
    plt.subplots_adjust(wspace = 0, right=0.99, top = 0.99)
    return(fig, axs)



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
        print("\33[31mlegacy parameter 'calibrate' was used. Use 'correct' instead\33[0m")
        correct = calibrate
    if "peaks" in kwargs:
        print("\33[31mlegacy parameter 'peaks' was used. Use 'return_peaks' instead\33[0m")
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
    
    
    print(f"  {tcol}config:\33[0m")
    for key, value in config.items():
        print(f"    {tcol}{key}:\33[0m {value}")
    
    
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
    
    
    
    
    
    
    
    
# processing data per run one plugin at a time is much faster and
#   more reiliable than loading all at once

def get_linage(target, context = context_sp):
    return(
        context.lineage(
            run_id = 0,
            data_type = target
        )
    )

def get_load_order(target, context = context_sp):
    linage = get_linage(target, context = context)
    linage2 = {tar:len(get_linage(tar))-1 for tar in linage}
    
    target_order = [[]] * (max(linage2.values())+1)
    for n,c in linage2.items():
        target_order[c] = (*target_order[c], n)

    target_order.reverse()
    target_order = [t for to in target_order for t in to]

    return(target_order)
    
def get_linage_todo(
    run,
    target,
    config = False,
    mconfig = None,
    context = False,
    verbose = True,
):
    if context is False:
        context = context_sp
        if verbose: print(f"{tcol}context\33[0m: default to context_sp")
    if config is False:
        config = find_config(target)

    if isinstance(mconfig, dict):
        config = {**config, **mconfig}
    
    if verbose: print(f"{tcol}config\33[0m: {config}")
    load_order = get_load_order(target, context = context)
    
    todo = []
    load_order = get_load_order(target, context = context)
    
    for i_load, load in enumerate(load_order):
        check = context.is_stored(
            run_id = f"{run:0>5}",
            target = load,
            config = config,
        )
        if verbose: print(f"* {tcol}{load}\33[0m: {check}")
        if check is True:
            break
        todo.insert(0, load)
    return(todo)
    
    
def process_linage(
    run,
    target,
    context = False,
    config = False,
    mconfig = None,
    todo = False,
    process = True,
    verbose = True,
    title = False,
    f_clear = False
):
    if not isinstance(run, (str, int, np.number)):
        for i_r, r in enumerate(run):
            if callable(f_clear):
                f_clear()
            print(f"{i_r+1}/{len(run)}: {r}")
            process_linage(
                r, target = target,
                context = context, config = config, mconfig = mconfig,
                todo = todo, process = process,
                verbose = verbose, title = title,
            )
    else:
        if context is False:
            context = context_sp
            if verbose: print(f"{tcol}context\33[0m: default to context_sp")
        if config is False:
            config = find_config(target)
            if verbose: print(f"{tcol}config\33[0m: {config}")
            
        if todo is False:
            todo = get_linage_todo(
                run = run,
                target = target,
                config = config,
                mconfig = mconfig,
                context = context,
                verbose = verbose,
            )
                
        for target_todo in todo:
            if title is not False:
                print(f"\n\033]0;{title}: {target_todo}\a", flush = True)
            if verbose: print(f"{tcol}{target_todo}\33[0m is being loaded")
            if target_todo == "raw_records":
                print(f"  \33[31m!!!raw records do not exist for run {run}, skipping!!!\33[0m")
                break
            elif process is True:
                _ = load(
                    run, target_todo,
                    config = config,
                    mconfig = mconfig,
                    context = context,
                    v = False, 
                    check_load = False,
                    correct = False
                )
            else:
                if verbose: print(f"{tcol}{target_todo}\33[0m not being loaded")
    return(None)






# eff companion functions
fit_par_bins = {
    "t_S11": fhist.make_bins(0, 750, 10),
    "t_decay": fhist.make_bins(0, 1500, 20),
    "t_drift": fhist.make_bins(0, 60_000, 500),
    "tau": fhist.make_bins(0, 100, 2),
    "a": fhist.make_bins(0, 15, .25),
    "sigma": fhist.make_bins(0, 750, 10),
    "A1": fhist.make_bins(0, 10, .25),
    "A2": fhist.make_bins(0, 10, .25),
    "A3": fhist.make_bins(0, 15, .1),
    "A4": fhist.make_bins(0, 1, .025),
    "dct_offset": 25,
}
def hist_indiv(ax, data, bins, label = "", **kwargs):
    c, b = np.histogram(data, bins = bins)
    bc = fhist.get_bin_centers(b)
    
    ax.plot(
        bc, c,
        drawstyle = "steps-mid",
        label = f"{label} ({np.sum(c):,.0f}/{len(data):,})",
        **kwargs
    )


    
    
def draw_param(ax, dss, fef_i, title = ""):
    
    fef_par_label = eff.f_event_txt[fef_i]
    
    

    
    bins = fit_par_bins[fef_par_label]
    n_drawn = 0
    
    for ds_label, ds in dss.items():
        fit_field, fit_labels, fit_units = eff.get_fit_infos(ds)
        if fef_par_label in fit_labels:
            fit_i = fit_labels.index(fef_par_label)
            data = ds[fit_field][:, fit_i]
            style = dict(alpha = .75)
            n_drawn += 1
        else:
            data = []
            style = dict(alpha = .25, linestyle = "dashed")
        hist_indiv(ax, data = data, bins = bins, label = ds_label, **style)
    
    
    if n_drawn > 0:
        ax.set_title(f"{title} ({fef_i}: {fef_par_label})")
        ax.set_xlabel(f"{fef_par_label} / {eff.f_units[fef_par_label]}")
        ax.set_ylabel("counts")
        ax.set_yscale("log")
        ax.legend(loc = "upper right")
    
    return(n_drawn)

def draw_all_kr_lifetimes(dss, fo, prefix = "", suffix = "", title = ""):
    qp(f"\n(12) \33[34mKr lifetime\33[0m")
    ax = fhist.ax(w = 10)
    ax.set_title(title)
    
    plt.subplots_adjust(left = .10, right = .7)
    for ds_label, ds in dss.items():
        _ = get_kr_lifetime_from_run(
            kr = ds,
            ax = ax,
            draw_info = False,
            draw_fit = False,
            t_lims = (150, 1000),
            label = ds_label
        )
        fhist.add_fit_parameter(ax, "\\tau", *_[0], "ns")
        fhist.addlabel(ax, " ")
    ax.set_yscale("log")
    ax.legend(loc = (1.01, -.10))
    qp(", saving")
    plt.savefig(f"{fo}/{prefix}12_Kr_lifetime{suffix}.png")
    qp(", closing")
    plt.close()
    qp(", done")


def draw_all_param(dss, fo, prefix = "", suffix = "", title = ""):
    for i_fit_label, label in enumerate(eff.f_event_txt):
        qp(f"\n({i_fit_label:>2}) \33[34m{label}\33[0m")

        ax = fhist.ax()
        n_drawn = draw_param(ax, dss, i_fit_label, title = title)

        if n_drawn > 0:
            qp(", saving")
            plt.savefig(f"{fo}/{prefix}{i_fit_label:0>2}_{label}{suffix}.png")
        else:
            qp(", \33[31mnothing drawn\33[0m")
        qp(", closing")
        plt.close()
        qp(", done")
    draw_all_kr_lifetimes(dss = dss, fo=fo, prefix = prefix, suffix = suffix, title = title)
    print("\nALL  Done")
    





labels_ops = {
    "lt":  "<", "le": "<=",
    "gt":  ">", "ge": ">=",
    "eq": "==", "ne": "!=",
}  

    
def check_x(x, op, y, fallback = True, v = False):
    '''
    wrapper function that performs check 'op' on x against y
    
    x: the values to check
    op: the operation that is uesed to check
        (see 'labels_ops' to see some available parameters)
    y: the value to check against
    
    
    fallback (True): an array like x full of 'fallback' will be returned if:
        - op or __op__ does not exist in x
        - calling op or __op__ causes an error
    
    '''
    
    x_dir = x.__dir__()
    dunder_op = f"__{op}__"
    op_call = False
    if op in x_dir:
        op_call = f"{op}"
    elif dunder_op in x_dir:
        op_call = f"{dunder_op}"

    try:
        ylen = len(y)
    except TypeError:
        ylen = 1
        
    xshape = x.shape
    if len(xshape) > 1:
        if xshape[1] > ylen:
            if v: print("shortened x to match y")
            x = x[:, :ylen]
        else:
            if v: print("shortened y to match x")
            y = y[:xshape[1]]
    elif ylen > 1:
        if v: print("shortened y to match x")
        y = y[0]
        
    
    str_op = f"{op}"
    if op in labels_ops:
        str_op = f"{labels_ops[op]}"
    else:
        str_op = f"{op}"
    
    if op_call is False:
        if v: print(f"\33[31mcan't find '{op}' in x\33[0m")
        return(fallback * np.ones_like(x))
        
        
    if v: print(f"{op_call} ({str_op}) in x")
    
    call = x.__getattribute__(op_call)
    try:
        try:
            result = call(y)
        except TypeError:
            result = call()
        return(result)
    except Exception as e:
        print(f"failed appliying {op_call} to x: \33[31m{e}\33[0m")
        return(fallback * np.ones_like(x))




def filter_ds(ds, filters):
    '''
        allows the quick filtering of datasets
    '''
    try:
        fit_field, fit_labels, fit_units = eff.get_fit_infos(ds)
    except IndexError:
        fit_field, fit_labels, fit_units = [], [], []
    
    dsf = ds
    for par, *filt in filters:
        # part where values are obtained
        if callable(par):
            vs = par(dsf)
            print(f"  \33[36m{par.__name__}:\33[0m ", end = "")
        else:
            pars_derived = get_field(dsf, par)
            if isinstance(pars_derived, str):
                par = pars_derived
            print(f"  \33[34m{par}:\33[0m ", end = "")
            if par in ds.dtype.names:
                vs = dsf[par]
            elif par in fit_labels:
                id_field = fit_labels.index(par)
                vs = dsf[fit_field][:, id_field]
            else:
                print(f"\33[31m(parameter not found)\33[0m")
                continue

        # part where all checks are performed
        for op, lim in filt:
            print(f" ({op}: {lim})", end = "")
            chk = check_x(vs, op, lim)

            if len(vs.shape) != 1:
                chk = np.all(chk, axis = 1)
            dsf = dsf[chk]
            vs = vs[chk]
        print()
    print(f"    N: {len(ds)} --> {len(dsf)}")
    return(dsf)
    
    
    
    
def label_filters(ax, filters, loc = "upper right"):

    for par, *filt in filters:
        if callable(par):
            par_str = f"{par.__name__}()"
        else:
            par_str = f"{par}"
        for op, lim in filt:
            if op in labels_ops:
                op_str = labels_ops[op]
            else:
                op_str = f"{op}"
                
            fhist.addlabel(ax, f" {par_str} {op_str} {lim}")
    if loc is not False:
        try:
            ax.legend(loc = loc)
        except Exception:
            pass
            
            
            





print(f"{tcol}Python:\33[0m")
print(sys.executable)
print(sys.version)
print(sys.version_info)
print(f"{tcol}Import done at {datetime.now()}\33[0m")
