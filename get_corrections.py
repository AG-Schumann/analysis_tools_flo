import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import flo_histograms as fhist
import flo_functions as ff
import matplotlib.pyplot as plt
from default_bins import *
from flo_fancy import *

import inspect
from datetime import datetime











# NEW METHOD FOR GETTING Gate and Cathode
gc_info = {
   "gate": (ff.erf_lin, "lower right", "S1 area", "PE", "gate"),
   "cath": (ff.erf, "upper right", "counts", "", "cathode"),
   "gate_ratio": (ff.erf_lin, "upper right", "S2/S1 area", "", "gate"),
}
gc_units_all = {
    "mu": "µs",
    "sigma": "µs",
    "y0": True,
    "y1": True,
    "a": "",
}



labels = {
    "event_fits": np.array(["first S1", "second S1", "first S2", "second S2"]),
    "event_fits_summary": np.array(["first S1", "second S1", "first S2", "second S2", 'unsplit S1', 'unsplit S2', "total S1", "total S2"]),
    "sp_krypton_summary": np.array(['first S1', 'second S1', 'first S2', 'second S2', 'unsplit S1', 'unsplit S2', 'total S1', 'total S2']),
}




def gc_get_xydata(ds, what, bins):
    if what == "gate_ratio":
        dat = fhist.binning(ds["drifttime"], ds["areas"][:, 7]/ds["areas"][:, 6], bins = bins)
    else:
        dat = fhist.binning(ds["drifttime"], ds["areas"][:, 6], bins = bins)
    
    medians = pd.DataFrame(columns=['bc', 'mu', 'smu', "spr"])
    if what[:4] == "gate":
        for bc, data in dat.items():
            if len(data) > 0:
                try:
                    mu, spr, smu = fhist.median_gauss(data, bins = 25)
                    medians = medians.append({
                        "bc": bc,
                        "mu": mu,
                        "smu": smu,
                        "spr":spr,            
                    }, ignore_index = True)
                except Exception:
                    pass
            
    if what == "cath":
        for bc, data in dat.items():
            if len(data) > 0:
                try:
                    mu = len(data)
                    spr = len(data)**.5
                    medians = medians.append({
                        "bc": bc,
                        "mu": mu,
                        "spr":spr,            
                    }, ignore_index = True)
                except Exception:
                    pass
        medians["smu"] = fhist.s_pois(medians["mu"])[1]
        

    return(medians)

def gc_get_unit(p, unit_dict, unit = ""):
    if p in unit_dict:
        _ = unit_dict[p]
        if _ is True:
            return(unit)
        return(_)
    return("")




def find_cathode_and_gate(
    ds,
    N_bin_shifts = 4,
    figs_path = False,
    title = "",
    append_dict = None,
    what_todo = True,
):
    if what_todo is True:
        what_todo = ["gate", "cath"]
    
    if isinstance(what_todo, str):
        what_todo = [what_todo]
    
    if not isinstance(what_todo, (list, np.ndarray)):
        raise ValueError("what_todo must be list, str or numpy.array")
    
    if not isinstance(append_dict, dict):
        append_dict = dict()

    fmt_i = len(str(N_bin_shifts))
    df_indiv = pd.DataFrame()
    df_summary = pd.DataFrame()


    for i_what, what in enumerate(what_todo):
        df_what = pd.DataFrame()
        print(what)
        f, loc, label, unit, what_title = gc_info[what]
        units = [gc_get_unit(p, gc_units_all, unit) for p in f.parameters]
        bins = default_bins[f"find_{what[:4]}"]
        bw = np.diff(bins)[0]
        bin_offset_each = bw / N_bin_shifts



        for i_bin_shift in range(N_bin_shifts):
            bin_offset = bin_offset_each*i_bin_shift
            iteration = i_bin_shift+1
            bins_iteration = bins + bin_offset
            medians = gc_get_xydata(ds, what, bins_iteration)
            qp(f" ({iteration:>{fmt_i}}/{N_bin_shifts}) ")

            x = medians["bc"].values
            y = medians["mu"].values
            sy = medians["smu"].values
            spr = medians["spr"].values


            try:
                p0 = f.p0(x, y)
                fit, cov = curve_fit(
                    f, 
                    x, y,
                    sigma = sy,
                    absolute_sigma = True,
                    p0 = p0
                )
                sfit = np.diag(cov)**.5
                chi2 = chi_sqr(f, x, y, sy, *fit)
                out = {
                    "what": what,
                    "iteration": iteration,
                    "chi2": chi2[2],
                    "parameters": f.parameters,
                    
                }
                mu, smu, sigma, ssigma = [False, False, False, False]
                for par, v, sv in zip(f.parameters, fit, sfit):
                    out[f"result_{par}"] = v
                    out[f"result_s{par}"] = sv
                    if par == "mu":
                        mu = v
                        smu = sv
                    elif par == "sigma":
                        sigma = v
                        ssigma = sv
                out = {**out, **append_dict}
                df_what = df_what.append(out, ignore_index = True)


                strs_results = [
                    f"µ:({v_str(mu)} +- {v_str(smu)}) µs",
                    f"σ:({v_str(sigma)} +- {v_str(ssigma)}) µs",
                    f"(χ² = {v_str(chi2[2])})",
                ]

                print(", ".join(strs_results))


                if (figs_path is not False) and isinstance(figs_path, str):


                    figs_path_iter = figs_path.replace("%WHAT%", what).replace("%ITER%", f"{iteration:0>{fmt_i}}")

                    ax2 = fhist.ax()
                    ax = ax2.twinx()
                    ax2.set_title(f"{what_title} {title} (i = {iteration})")
                    color = ax2.plot(x, y, ".", label = "data")[0].get_color()
                    
                    
                    fhist.errorbar(ax2, x, y, sy, color = color, ax2 = ax, slimit = True)
                    fhist.errorbar(ax2, x, y, spr,color = color, ax2 = ax, slimit = True, alpha = .2, )
                    xp = np.linspace(min(bins), max(bins), 1000)

                #     y0 = f(xp, *p0)
                #     ax.plot(xp, y0, label = "p0")

                    yf = f(xp, *fit)
                    ax2.plot(xp, yf, label = f"fit {chi2[4]}")
                    add_fit_parameters(ax2, f, fit, sfit, units)
                    ax2.set_xlabel(f"drift time / µs")
                    ax2.set_ylabel(string_join(f"{label}", unit, sep = " / "))
                    ax2.legend(loc = loc)
                    
                    ax.set_axis_off()
                    ax.set_ylim(ax2.get_ylim())
                    plt.savefig(figs_path_iter)
                    plt.close()
                    
            except Exception as e:
                print(f"\33[31mfailed indiv.\33[0m: {e}")
                pass
        df_indiv = df_indiv.append(df_what)


        # summary bloc kstarts here
        out_summary = {
            "what": what,
            "parameters": f.parameters,
        }
        
        
        strs_results = ""
        try:
            x = df_what["chi2"].values
            

            for par, par_tex, unit in zip(f.parameters, f.parameters_tex, units):
                y = df_what[f"result_{par}"].values
                sy = df_what[f"result_s{par}"].values
                v, sprv, sv = fhist.mean_w(y, sy)

                out_summary[f"result_{par}"] = v
                out_summary[f"result_s{par}"] = sv 


                if (figs_path is not False) and isinstance(figs_path, str):
                    ax2 = fhist.ax()
                    ax = ax.twinx()
                    ax.set_axis_off()
                    figs_path_param = figs_path.replace("%WHAT%", what).replace("%ITER%", f"summmary_{par}")
                    ax2.set_title(f"{title}{what} ${par_tex}$")
                    ax2.set_xlabel("$\\chi^2_\\mathrm{{red}}$")

                    ax2.set_ylabel(string_join(f"${par_tex}$", unit, sep = " / "))
                    fhist.errorbar(ax2, x, y, sy, plot = True, label = f"${par_tex}$", slimit=sv*10, ax2 = ax)

                    fmt = get_nice_format(v, sv)
                    ax2.axhline(v, label = f"mean: ({v:{fmt}}±{sv:{fmt}}) {unit}")
                    ax2.axhspan(v-sv, v+sv, alpha = .25)

                    ax2.legend(loc = "upper right", fontsize = 8)
                    
                    ax.set_ylim(ax2.get_ylim())
                    plt.savefig(figs_path_param)
                    plt.close()
                    
            mu, smu = out_summary[f"result_mu"], out_summary[f"result_smu"]
            sigma, ssigma = out_summary[f"result_sigma"], out_summary[f"result_ssigma"]
            
            strs_results  = f"µ:({v_str(mu)} +- {v_str(smu)}) µs, "
            strs_results += f"σ:({v_str(sigma)} +- {v_str(ssigma)}) µs"
            strs_results += "\33[0m"
                    
        except Exception as e:
            print(f"\33[31mfailed summary\33[0m: {e}")
            pass
            
        
        
        
        qp(f" \33[1m\33[35m{'MEAN':>{2*fmt_i+3}} ")
        print(strs_results)

        
        out_summary = {**out_summary, **append_dict}
        df_summary = df_summary.append(out_summary, ignore_index = True)

    df_indiv = df_indiv.astype({
        'iteration': 'int32',
    })

    return(df_summary, df_indiv)


# end of gate / cathode 



# start of S2 correction



def get_electron_lifetime(
    ds,
    tpc_geometry = False,
    fig_path = False,
    ax = False,
    title = "",
    f = ff.exp_decay,
    bins = True,
    index_tau = True,
    verbose = False,
    field_id = 2,
    plugin = "event_fits_summary",
    show_corrected = False,
    calc_linearity = False,
    return_all = False,
    path_medians = False,
):
    '''
    calculates the electron lifetime based on drifttime and total S2 area
    either '\33[1mtpc_geometry\33[0m' or '\33[1mbins\33[0m' need to be given to specify the time range/drift time bins
    
    returns tau, sigma_tau, chi^2_reduced
    
    parameters:
    * ds:
        the dataset (a [...]_summary datakind) to apply the fit
    * tpc_geometry (False):
        a gate_cathode correction dictionary (either the full thing or only the info part)
        mystrax.get_correction_for("gate_cathode")
    * bins (True):
        bins to be used to bin S2 areas by drifttime into
        if True: calculate bins based on tpc_geometry
        if list/array: use those bins
        anything else:
            raises error
    * fig_path (False):
        if this is a string the script creats a figure and saves it onto the given filename
    * title (""):
        the title for the plot ("electron lifetime {title}")
    * f (ff.exp_decay):
        the function that is fitted to the data
        (it should be of type fit_function)
    * index_tau (True):
        which of fs parameters is tau
        if set to True the script searches the f.parameters for 'tau'
        raises an error very early if failing
        
    '''
    qp(f"starting >{title}<", verbose = verbose)
    
    
    if len(ds) <= 10:
        print(f" \33[0mgot only {len(ds)} datapoints, quitting\33[0m")
        return(0,0,0)
    
    out = {}
    if (index_tau is True) and ("tau" in f.parameters):
            index_tau = f.parameters.index("tau")
            qp(f", tau= {index_tau}", verbose = verbose)
    elif isinstance(index_tau, bool) or not isinstance(index_tau, int):
        # bool is a subclass of int.....
        raise ValueError(f"can not find tau in functions parameters ({f.parameters}), please specify '\33[1mindex_tau\33[0m'")
    
    if isinstance(fig_path, str):
        qp(", \33[32max created\33[0m", verbose = verbose)
        ax = fhist.ax()
    
        
    if isinstance(ax, plt.Axes):
        qp(", \33[32max given\33[0m", verbose = verbose)
        data_label = labels[plugin][field_id]
        ax2 = ax
        ax = ax2.twinx()
        ax.set_axis_off()
        ax2.set_title(f"electron lifetime {title}")
        ax2.set_xlabel(f"drift time / µs")
        ax2.set_ylabel(f"mean {data_label} area / PE")
    else:
        ax = False
        qp(", \33[31mno ax\33[0m", verbose = verbose)
    
    
    if bins is True:
        qp(", auto bins", verbose = verbose)
        if tpc_geometry is False:
            raise ValueError("either '\33[1mbins\33[0m' or '\33[1mtpc_geometry\33[0m' need to be given")
        if "info" in tpc_geometry:
            tpc_geometry = tpc_geometry["info"]
    
        bins = np.arange(tpc_geometry["dft_gate"], tpc_geometry["dft_cath"], 2)
    
    if not isinstance(bins, (list, np.ndarray)):
        raise ValueError("either '\33[1mbins\33[0m' or '\33[1mtpc_geometry\33[0m' need to be given")
        
    qp(", binning", verbose = verbose)
    
    if np.all(ds["drifttime_corrected"] == 0):
        data_x = ds["drifttime"]
    else:
        data_x = ds["drifttime_corrected"]
        bins = bins + ds["drifttime_corrected"][0] - ds["drifttime"][0] 
    
    data_y = ds["areas"][:,field_id]
    
    out["data"] = [data_x, data_y]
    
    df_median = fhist.get_binned_median(
        data_x, data_y,
        bins = bins,
        n_counts_min=5,
        path_medians = path_medians, xlabel= f"{data_label} area / PE", title = f"{title} %BC% µs"
    )
    
    x = df_median["bc"]
    y = df_median["median"]
    sy = df_median["s_median"]
    spr = df_median["spread"]
    out["data_binned"] = [x, y, sy, spr]
    
    
    if isinstance(ax, plt.Axes):
        
        slimit = np.median(sy)*10
        color = fhist.errorbar(ax, x, y, sy, plot = True, slimit = slimit, label = f"data: {data_label} area", ax2 = ax2)
        _ = fhist.errorbar(ax, x, y, spr, color = color, alpha = .5, slimit = slimit, ax2 = ax2)
        
        
        
    
    qp(", fitting", verbose = verbose)
    fit_res = ff.fit(f, x, y, sy, ax = ax, color = color, units = ["PE", "µs"])
    fit, sfit, chi2 = fit_res
    
    out["fit"] = fit_res
    out["f_fit"] = f
    
    
    if isinstance(ax, plt.Axes):
        if (show_corrected is True) or (calc_linearity is True):
            data_y_c = data_y * np.exp(data_x/fit[index_tau])
            
            out["data_c"] = [data_x, data_y_c]
            df_median_c = fhist.get_binned_median(data_x, data_y_c, bins = bins, n_counts_min=5)
            
            x_c = df_median_c["bc"]
            y_c = df_median_c["median"]
            sy_c = df_median_c["s_median"]
            spr_c = df_median_c["spread"]
            out["data_binned"] = [x_c, y_c, sy_c, spr_c]
            
            
            color_corrected = fhist.errorbar(ax, x_c, y_c, sy_c, plot = True, slimit = slimit, label = "corrected data", marker = "x", ax2 = ax2)
            _ = fhist.errorbar(ax, x_c, y_c, spr_c, color = color_corrected, alpha = .5, slimit = slimit, ax2 = ax2)
        
            if calc_linearity is True:
                f_fit_poly_0 = ff.poly_0
                f_fit_poly_1 = ff.poly_1
                fit_poly_0 = ff.fit(f_fit_poly_0, x_c, y_c, sy_c, ax = ax, color = color_corrected, units = ["PE"], label = "constant fit")
                fit_poly_1 = ff.fit(f_fit_poly_1, x_c, y_c, sy_c, ax = ax, color = color_corrected, units = ["PE/µs", "PE"], label = "linear fit", kwargs_plot=dict(linestyle = "dashed"))
                
                out["fit_poly_0"] = fit_poly_0
                out["fit_poly_1"] = fit_poly_1
                out["f_fit_poly_0"] = f_fit_poly_0
                out["f_fit_poly_1"] = f_fit_poly_1
            
    
    
        ax.legend(loc = (1.01, 0.2), fontsize = 8)
        ax2.set_ylim(ax.get_ylim())
        
        if isinstance(fig_path, str):
            qp(", saving", verbose = verbose)
            plt.savefig(fig_path)
            qp(", closing", verbose = verbose)
            plt.close()
        qp(", done", end = "\n", verbose = verbose)
        
    if return_all is not True:
        return(fit[index_tau], sfit[index_tau], chi2[2])
    else:
        return(fit[index_tau], sfit[index_tau], out)


# S1 correction

def get_S1_correction_paramters(
    ds,
    tpc_geometry = False,
    fig_path = False,
    title = "",
    f = ff.poly_1,
    bins = True,
    verbose = False,
    
):
    '''
    calculates the electron lifetime based on drifttime and total S2 area
    either '\33[1mtpc_geometry\33[0m' or '\33[1mbins\33[0m' need to be given to specify the time range/drift time bins
    
    returns tau, sigma_tau, chi^2_reduced
    
    parameters:
    * ds:
        the dataset (a [...]_summary datakind) to apply the fit
    * tpc_geometry (False):
        a gate_cathode correction dictionary (either the full thing or only the info part)
        mystrax.get_correction_for("gate_cathode")
    * bins (True):
        bins to be used to bin S2 areas by drifttime into
        if True: calculate bins based on tpc_geometry
        if list/array: use those bins
        anything else:
            raises error
    * fig_path (False):
        if this is a string the script creats a figure and saves it onto the given filename
    * title (""):
        the title for the plot ("electron lifetime {title}")
    * f (ff.exp_decay):
        the function that is fitted to the data
        (it should be of type fit_function)
    * index_tau (True):
        which of fs parameters is tau
        if set to True the script searches the f.parameters for 'tau'
        raises an error very early if failing
        
    '''
    qp(f"starting >{title}<", verbose = verbose)
    
    if isinstance(fig_path, str):
        qp(", \33[32max\33[0m", verbose = verbose)
        ax = fhist.ax()
        ax.set_title(f"S1 correction {title}")
        ax.set_xlabel(f"corrected drift time / µs")
        ax.set_ylabel(f"mean S2 area / PE")
    else:
        ax = False
        qp(", \33[31mno ax\33[0m", verbose = verbose)
    
    
    if bins is True:
        qp(", auto bins", verbose = verbose)
        if tpc_geometry is False:
            raise ValueError("either '\33[1mbins\33[0m' or '\33[1mtpc_geometry\33[0m' need to be given")
        if "info" in tpc_geometry:
            tpc_geometry = tpc_geometry["info"]
    
        bins = np.arange(0, tpc_geometry["dft_cath"], 2)
    
    if not isinstance(bins, (list, np.ndarray)):
        raise ValueError("either '\33[1mbins\33[0m' or '\33[1mtpc_geometry\33[0m' need to be given")
        
    qp(", binning", verbose = verbose)
    df_median = fhist.get_binned_median(ds["drifttime_corrected"], ds["areas"][:,6], bins = bins, n_counts_min=5)
    
    x = df_median["bc"][1:-1]
    y = df_median["median"][1:-1]
    sy = df_median["s_median"][1:-1]
    spr = df_median["spread"][1:-1]
    
    if isinstance(ax, plt.Axes):
        color = fhist.errorbar(ax, x, y, sy, plot = True, slimit = True)
        _ = fhist.errorbar(ax, x, y, spr, color = color, alpha = .5, slimit = True)
    
    qp(", fitting", verbose = verbose)
    fit_res = ff.fit(
        f,
        x, y, sy,
        ax = ax,
        units = ["PE/µs", "PE"],
        show_f = True)
    fit, sfit, chi2 = fit_res
    
    if isinstance(ax, plt.Axes):
        ax.legend(loc = "upper right")
        
        if isinstance(fig_path, str):
            qp(", saving", verbose = verbose)
            plt.savefig(fig_path)
            qp(", closing", verbose = verbose)
            plt.close()
        qp(", done", end = "\n", verbose = verbose)
        

    
    return(fit_res)

