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








# NEW METHOD FOR GETTINGF Gate and Cathode
gc_info = {
   "gate": (ff.erf_lin, "lower right", "S1 area", "PE", "gate"),
   "cath": (ff.erf, "upper right", "counts", "", "cathode"),
}
gc_units_all = {
    "mu": "µs",
    "sigma": "µs",
    "y0": True,
    "y1": True,
    "a": "",
}



def gc_get_xydata(ds, what, bins):
    dat = fhist.binning(ds["drifttime"], ds["areas"][:, 6], bins = bins)
    
    medians = pd.DataFrame(columns=['bc', 'mu', 'smu', "spr"])
    if what == "gate":
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
    append_dict = None
):

    if not isinstance(append_dict, dict):
        append_dict = dict()

    fmt_i = len(str(N_bin_shifts))
    df_indiv = pd.DataFrame()
    df_summary = pd.DataFrame()


    for i_what, what in enumerate(["gate", "cath"]):
        df_what = pd.DataFrame()
        print(what)
        f, loc, label, unit, what_title = gc_info[what]
        units = [gc_get_unit(p, gc_units_all, unit) for p in f.parameters]
        bins = default_bins[f"find_{what}"]
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

                    ax = fhist.ax()
  
                    ax.set_title(f"{what_title} {title} (i = {iteration})")
                    color = ax.plot(x, y, ".", label = "data")[0].get_color()
                    fhist.errorbar(ax, x, y, sy, color = color)
                    fhist.errorbar(ax, x, y, spr, color = color, alpha = .2)
                    xp = np.linspace(min(bins), max(bins), 1000)

                #     y0 = f(xp, *p0)
                #     ax.plot(xp, y0, label = "p0")

                    yf = f(xp, *fit)
                    ax.plot(xp, yf, label = f"fit {chi2[4]}")
                    add_fit_parameters(ax, f, fit, sfit, units)
                    ax.set_xlabel(f"drift time / µs")
                    ax.set_ylabel(string_join(f"{label}", unit, sep = " / "))
                    ax.legend(loc = loc)
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
        except Exception as e:
            print(f"\33[31mfailed summary\33[0m: {e}")
            pass
        out_summary = {**out_summary, **append_dict}
        df_summary = df_summary.append(out_summary, ignore_index = True)

    df_indiv = df_indiv.astype({
        'iteration': 'int32',
    })

    return(df_summary, df_indiv)
