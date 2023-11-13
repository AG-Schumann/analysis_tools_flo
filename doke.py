import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import flo_fancy
from flo_fancy import qp
import flo_functions as ff
import flo_histograms as fhist






def get_df_from_ds(
    ds,
    Energys = True,
    W = 13.7,
    field_ids = True,
    field = "areas_corrected",
    label_prefix = "$^{{83}}$Kr: ",
    label_suffix = "",
    doke_df = False,
    axs = False,
    title = "",
    verbose = False,
    
):
    if Energys is True:
        Energys = [9.4053, 32.1516, 41.5569] # keV
    if field_ids is True:
        field_ids = [(1,3), (0,2)]#, (6,7)]
    if axs is False:
        axs = [False, False] * len(Energys)
    elif axs is True:
        fig, axs = fhist.make_fig(min(len(Energys), len(field_ids)), 2, rax = False)
        fig.suptitle(title)
        
    if verbose is True:
        print(f"Energys: {Energys}")
        print(f"field_ids: {field_ids}")
        
        

        
    if not isinstance(doke_df, pd.DataFrame):
        doke_df = pd.DataFrame()
    
    for i_E, E, fids in flo_fancy.enumezip(Energys, field_ids):
        signals = ds[field][:, fids]
        s1, s2 = signals[:,0], signals[:,1]
        
        _, s1, s2 = flo_fancy.remove_zero((s1 > 0) & (s2 > 0), s1, s2)
        
        ax1 = ax2 = False
        if axs is not False:
            if verbose is True:
                print(f"plotting {E}")
            try:
                if len(axs.shape) == 2:
                    ax1 = axs[0, i_E]
                    ax2 = axs[1, i_E]
                else:
                    ax1 = axs[0]
                    ax2 = axs[1]
            
                ax1.set_title(f"S1: {E:.1f} keV")
                ax2.set_title(f"S2: {E:.1f} keV")
                ax1.set_xlabel(f"corrected S1 area / PE")
                ax2.set_xlabel(f"corrected S2 area / PE")
                
            except Exception as e:
                    ax1 = ax2 = False
                    if verbose is True:
                        print(e)
            
        S1, spr_S1, sS1 = fhist.median_gauss(s1, strict_positive=True, ax = ax1)
        S2, spr_S2, sS2 = fhist.median_gauss(s2, strict_positive=True, ax = ax2)


        out = {
            "S1": S1,
            "S2": S2,
            "sS1": sS1,
            "sS2": sS2,
            "E": E,
            "label": f"{label_prefix}{E:.1f} keV{label_suffix}",
        }
        for s_, l in [("S1","x"), ("S2", "y"), ("sS1", "sx"), ("sS2", "sy")]:
            soe = out[f"{s_}"] / E
            out[f"{s_}oE"] = soe
            out[f"{l}"] = soe / 1000*W
            
        doke_df = doke_df.append(
            out,
            ignore_index = True
        )
        
        
    return(doke_df)





def df_plot(
    ax,
    doke_df,
    per_quanta = True,
    show_zero = True,
    add_labels = True,
    add_text_inplot = False,
    ax2 = None
):
    
    '''
    
    plots the doke_df
    
    
    parameters:
    - ax:
        where to plot into (needs to be an plt.Axes element)
    - doke_df:
        the doke_df obtained from doke.get_df_from_ds()
    - per_quanta (True):
        wheter to show the per quanta values (True) or per energy (False)
    - show_zero (True):
        shows (0,0)
    '''
    
    if not (ax, plt.Axes):
        raise TypeError("ax must be an axis element")
    
    if not (doke_df, pd.core.frame.DataFrame):
        raise TypeError("doke_df must be a DataFrame")

    if show_zero is True:
        ax.plot(0, 0, ".", color = "black", alpha = .0)
    
    
    if ("label" not in doke_df) and (add_labels is True):
        doke_df["label"] = ""
    else:
        label = ""

    for i_row, row in doke_df.iterrows():

        if per_quanta is True:
            x, y, sy, sx = row["x"], row["y"], row["sy"], row["sx"]
        else:
            x, y, sy, sx = row["S1oE"], row["S2oE"], row["sS2oE"],row["sS1oE"]
        
        if add_labels:
            label = row["label"]
        
        color = fhist.errorbar(
            ax,
            x, y, sy,
            sx = sx, plot = True,
            label = label,
            marker = ".",
            capsize = 0,
            ax2 = ax2,
        )
        
        if add_text_inplot is True:
            ax.text(x, y, f"  {label}", color = color, verticalalignment = "baseline")
        
        
    if per_quanta is True:
        ax.set_xlabel("SY W / PE/γ")
        ax.set_ylabel("CY W / PE/e")
    else:
        ax.set_xlabel("SY / PE/keV")
        ax.set_ylabel("CY / PE/keV")


def g1g2_uncertainty_from_doke_fit(
    fit,
    cov,
    ax = False,
    color = None,
    fixed_values = None,
    show_value_unc = False,
    verbose = False,
):
    
    f = ff.doke
    g1, g2 = fit
    
    qp(f"\33[35mg1g2_uncertainty_from_doke_fit\33[0m", verbose = verbose, end = "\n")
    qp(f"- g1: {g1}", verbose = verbose, end = "\n")
    qp(f"- g2: {g2}", verbose = verbose, end = "\n")
    
    str_g1 = f"$g_1: {g1:.2f}\, \\mathrm{{PE/γ}}$"
    str_g2 = f"$g_2: {flo_fancy.tex_value(g2, unit = 'PE/e', digits = 3, lim = (-3,0))}$"
    color_g1, color_g2 = "black", "black"
    
    
    if (g1 < 0) or (g2 < 0):
        return((np.inf, np.inf), np.inf)
    
    if fixed_values is None:
        fixed_values = dict()
    
    
    # does not make sense to calculate a value if both are fixed
    # (just a fallback in case)
    if ("g1" in fixed_values) and ("g2" in fixed_values):
        flo_fancy.addlabel(ax, str_g1, color = color_g1, marker = "x")
        flo_fancy.addlabel(ax, str_g2, color = color_g2, marker = "*")
        
        return((0, 0), 0)
    
    
    xp = np.logspace(-2,6, 10_000)
    yp = f(xp, *fit)
    sp = f.sf(xp, *fit, cov = cov)

    xl = np.interp(0, -(yp - sp), xp, right = np.inf, left=np.inf)
    xr = np.interp(0, -(yp + sp), xp, right = np.inf, left=np.inf)
    xp = np.linspace(0, xr, 10_000)

    yp = f(xp, *fit)
    sp = f.sf(xp, *fit, cov = cov)

    syl = yp-sp
    syh = yp+sp
    syl[syl < 0] = 0


    sg1r = xr-g1
    sg1l = g1-xl

    sg2 = f.sf(0, *fit, cov = cov)
    
    ypg0 = yp >= 0
    

    if isinstance(ax, plt.Axes):
        
        color = ax.plot(xp[ypg0], yp[ypg0], color = color)[0].get_color()
        ax.fill_between(xp, syl, syh, alpha = .2, color = color, linewidth = 0)
        
        
        if "g1" not in fixed_values:
            str_g1 = f"$g_1: \\left({g1:.2f}^{{+{sg1r:.2f}}}_{{-{sg1l:.2f}}}\\right)\, \\mathrm{{PE/γ}}$"
            
        if "g2" not in fixed_values:
            str_g2 = f"$g_2: {flo_fancy.tex_value(g2, sg2, 'PE/e', digits = 3, lim = (-3,0))}$"
        
    
        ax.axvline(0, color = "grey", alpha = .2)
        ax.axhline(0, color = "grey", alpha = .2)
        
        
        
        style_1 = dict(color = color_g1)
        style_2 = dict(color = color_g2)
        
           
        if "g1" in fixed_values:
            style_1["capsize"] = 0
        if "g2" in fixed_values:
            style_2["capsize"] = 0
        
        if show_value_unc is True:
            fhist.errorbar(ax, g1, 0, sy = False, sx = [[sg1l], [sg1r]], plot = True, marker = "x", **style_1)
            fhist.errorbar(ax, 0, g2, sg2, plot = True, marker = "*", **style_2)
        
        
        flo_fancy.addlabel(ax, str_g1, color = color_g1, marker = "x")
        flo_fancy.addlabel(ax, str_g2, color = color_g2, marker = "*")
        
        ax.set_xlim(0)
        ax.set_ylim(0)
        
    return((sg1l, sg1r), sg2)



def fit_df(
    doke_df,
    ax = False,
    label = "",
    fixed_values=None,
    kwargs_fit=None,
    sg_via_width = True,
    show_fit_result = False,
    color = "black",
    
    verbose = False,
    
):
    
    units = ["PE/γ", "PE/e"]
    
    if kwargs_fit is None:
        kwargs_fit = dict()
    fit_res = ff.fit(
        ff.doke,
        doke_df["x"], doke_df["y"], doke_df["sy"], doke_df["sx"],
        ax = ax,
        color = color,
        return_cov = True,
        ODR = True,
        xp = False,
        fixed_values = fixed_values,
        show_fit_result = show_fit_result,
        verbose = verbose,
        **kwargs_fit,
    )
    
    fit, cov, chi2 = fit_res
    sfit = np.diag(cov)**.5
    fit_result = fit, sfit, chi2, cov
    

    sg1, sg2 = g1g2_uncertainty_from_doke_fit(fit, cov, ax = ax, fixed_values = fixed_values, color = color, verbose = verbose)
    
    
    
    return(fit_result, (sg1, sg2))



def fit_df_lin(df, ax = False):
#     x, y, sx, sy = ss.S1oE, ss.S2oE, ss.sS1oE, ss.sS2oE
    x, y, sx, sy = df.x, df.y, df.sx, df.sy
    
    fit_dict = ff.fit(
        ff.poly_1,
        x, y,
        sigma = sy,
        sx = sx,
        ODR = True,
        ax = ax,
        show_fit_uncertainty=True,
        return_as_dict=True,
        xp = np.linspace(0, .17, 100),
        show_fit_result=False,
        label = "",
        color = "black"
    )
    fit = fit_dict["fit"]
    s_fit = np.diag(fit_dict["cov"])**.5
    m, c = fit
    s_m, s_c = s_fit
    cov_cm = fit_dict["cov"][0,1]
    
    g1 = (-c/m)
    s_g1 = (
          (s_c/m)**2
        + (c/m**2 * s_m)**2
        + (-c/m**3*cov_cm)
        
    )**.5
    
    g2 = c
    s_g2 = s_c
    
    str_g1 = f"$g_1: ({g1:.3f} \\pm {s_g1:.3f})$ PE/γ"
    str_g2 = f"$g_2: ({g2:.3f} \\pm {s_g2:.3f}) PE/e$"
    
    if isinstance(ax, plt.Axes):
        ax.set_xlim(0)
        ax.set_ylim(0)
        fhist.addlabel(ax, str_g1)
        fhist.addlabel(ax, str_g2)
    
    
    fit_dict["results"] = {
        "g1": g1,
        "s_g1": s_g1,
        "g2": g2,
        "s_g2": s_g2,
    }
    
    return(fit_dict)
    
