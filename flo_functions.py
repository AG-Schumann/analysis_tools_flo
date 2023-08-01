import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import inspect
from flo_fancy import *
from scipy.optimize import curve_fit
from scipy import odr
import scipy


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

class fit_function:
    def __init__(self, f, f_p0,
        description = "",
        sf = False,
        short_description = "",
        units = None,
        parameters = None, parameters_tex = None,
        formula = "", formula_tex = None,
        docstring = ""
    ):
        self.f = f
        self.p0 = f_p0
        self.sf = sf
        
        self.description = description
        self.short_description = short_description
        self.s = short_description
        self.__name__ = f.__name__
        self.__doc__ = docstring
        
        
          
        
        if parameters is None:
            raise ValueError("parmaters for function not given! use ")
        self.parameters = parameters
        self.plen = max([len(p) for p in parameters])
        
        if parameters_tex is None:
            self.parameters_tex = parameters
        else:
            self.parameters_tex = parameters_tex
        if units is None:
            self.units = [""] * len(self.parameters)
        else:
            self.units = units
        self.formula = formula
        if formula_tex is None:
            self.formula_tex = formula
        else:
            self.formula_tex = formula_tex
        
    
    def __call__(self, x, *args, **kwargs):
        return(self.f(x, *args, **kwargs))
    def __repr__(self):
        return(self.formula)
    def __str__(self):
        return(self.formula_tex)
    def __iter__(self):
        for pt in self.parameters_tex:
            yield(pt)
    def __len__(self):
        return(len(self.parameters))



def fit(
    f, x, y, sigma = None,
    sx = None, 
    p0 = True,
    ax = False,
    clean_input = True,
    units = None,
    color = None,
    label = "fit: ",
    show_fit_result = True,
    show_gof = True,
    show_f = False,
    show_p0 = False,
    verbose = False,
    kwargs_curvefit = None,
    kwargs_plot = None,
    return_cov = False,
    return_output = False,
    return_as_dict = False,
    scale_plot = 1,
    absolute_sigma = True,
    xp = True,
    show_fit_uncertainty = False,
    ODR = False,
    fixed_values = None,
):
    output = False
    
    if not isinstance(f, fit_function):
        qp("f is not a fit function", verbose = verbose, end = "\n")
        return(False)
    qp(f"\33[35musing {f} for fit\33[0m", verbose = verbose, end = "\n")
    
    if fixed_values is None:
        fixed_values = dict()
        qp("- no fixed values given", verbose = verbose, end = "\n")
    
    
    try:
        if not isinstance(kwargs_curvefit, dict):
            qp("- kwargs_curvefit not provided", verbose = verbose, end = "\n")
            kwargs_curvefit = {}
        if not isinstance(kwargs_plot, dict):
            qp("- kwargs_plot not provided", verbose = verbose, end = "\n")
            kwargs_plot = {}
        
        qp("- setting up parameters", verbose = verbose, end = "\n")
        
        
        fit_out = np.zeros(len(f))
        sfit_out = np.zeros(len(f))
        cov_out = 0*np.eye(len(f))
        ids_free = []
        pars_names = []
        
        for par_i, par in enumerate(f.parameters):
            if par in fixed_values:
                fit_out[par_i] = fixed_values[par]
            else:
                ids_free.append(par_i)
                pars_names.append(par)

        qp(f"- ids_free: {ids_free}", verbose = verbose, end = "\n")
        qp(f"- pars_names: {pars_names}", verbose = verbose, end = "\n")
        
        
        if p0 is True:
            qp("- obtaining p0", verbose = verbose, end = "\n")
        
            try:
                p0 = np.array(f.p0(x, y, **fixed_values))
                qp("- default worked", verbose = verbose, end = "\n")
                
            except TypeError:
                qp("- failed p0 with fixed_values, trying without", verbose = verbose, end = "\n")
                p0 = np.array(f.p0(x, y))
            
            if not isinstance(p0, np.ndarray):
                p0 = np.array([p0])
            p0_all = p0
            try:
                qp(f"- p0 all: {p0}", verbose = verbose, end = "\n")
                if (p0 is not False) and (len(ids_free) != len(f)):
                
                    qp("- trying to cut p0", verbose = verbose, end = "\n")
                    p0 = p0[np.array([*ids_free])]
            except Exception as e:
                qp(f"- p0 cutting error: {e}", verbose = verbose, end = "\n")

            
            
            qp(f"- p0 final: {p0}", verbose = verbose, end = "\n")
            qp(f"- len(p0): {len(p0)}", verbose = verbose, end = "\n")
            
        qp("- setting up f", verbose = verbose, end = "\n")
        f_fit = lambda x, *args: f.f(x, **{n:v for n,v in zip(pars_names, args)}, **fixed_values)
        
        
        
        if sigma is None:
            qp(f"- no sy given", verbose = verbose, end = "\n")
            absolute_sigma = False
        elif len(sigma.shape) > 1:
            qp(f"- sigma has more than one dimension, using upper values", verbose = verbose, end = "\n")
            sigma = sigma[1]
        if sx is not None and len(sx.shape) > 1:
            qp(f"- sx has more than one dimension, using upper values", verbose = verbose, end = "\n")
            sx = sx[1]

        if clean_input is True:
            qp(f"- cleaning data", verbose = verbose, end = "\n")
            x, y, sigma, sx = clean(x, y, sigma, sx)
        
        
        
        
        # the actual fits 
        if (ODR is True):
            qp(f"- ODR: testing", verbose = verbose, end = "\n")
            if (sx is None):
                raise ValueError("sx is not given for ODR")
            if (sigma is None):
                raise ValueError("sigma is not given for ODR")
            qp(f"- ODR: tests passed", verbose = verbose, end = "\n")
            
            qp(f"- ODR: rebuilding f", verbose = verbose, end = "\n")
            lf = lambda beta, x: f_fit(x, *beta)
            qp(f"- ODR: creating model", verbose = verbose, end = "\n")
            model = odr.Model(lf)
            qp(f"- ODR: creating data", verbose = verbose, end = "\n")
            data = odr.RealData(x, y, sx = sx, sy = sigma)
            qp(f"- ODR: creating ODR", verbose = verbose, end = "\n")
            myodr = odr.ODR(data, model, beta0 = p0)
            qp(f"- ODR: fitting", verbose = verbose, end = "\n")
            output = myodr.run()
            qp(f"- ODR: fitting done", verbose = verbose, end = "\n")
            
            fit = output.beta
            cov = output.cov_beta
            sfit = np.diag(cov)**.5
        else:
            for i in range(2):
                qp(f"- fitting {i}", verbose = verbose, end = "\n")
                fit, cov, *output = curve_fit(
                    f_fit, 
                    x, y,
                    sigma=sigma,
                    absolute_sigma=absolute_sigma,
                    p0 = p0,
                    full_output = True,
                    **kwargs_curvefit
                )
                # sometimes good starting parameters yield infinite covariance matrix..... urgh
                sfit = np.diag(cov)**.5
                if verbose is True:
                    nev = output[0]["nfev"]
                    print(f"fit {i+1} ({nev} iterations)")
                    for p, v, sv in zip(pars_names, fit, sfit):
                        print(f"{p:>{f.plen}}: {v:12.4g} ±{sv:12.4g}")
                
                if np.inf not in cov:
                    break
                elif i == 0:
                    if verbose is True: print("\33[31mWarning: found infinite covariance matrix, fitting again with p0 = fit + .1\33[0m")
                    p0 = fit + .1
        
        
        if len(fixed_values) > 0:
            for i_fit, id_par in enumerate(ids_free):
                fit_out[id_par] = fit[i_fit]
                for j_fit, jd_par in enumerate(ids_free):
                    cov_out[id_par, jd_par] = cov[i_fit, j_fit]
            sfit_out = np.diag(cov_out)**.5
        else:
            fit_out = fit
            sfit_out = sfit
            cov_out = cov
            
        qp("- fit results:", verbose = verbose, end = "\n")
        qp(fit, verbose = verbose, end = "\n")
        
        try:
            if sigma is not None:
                qp(f"- calculating chi^2", verbose = verbose, end = "\n")
                chi2 = chi_sqr(f_fit, x, y, sigma, *fit)
                chi_str = f" {chi2[4]}"
            else:
                qp(f"- calculating residuals^2", verbose = verbose, end = "\n")
                chi2 = res_sqr(f_fit, x, y, *fit)
                chi_str = f" {chi2[4]}"
        except ZeroDivisionError:
            qp(f"- \33[31mwarning: division by zero, chi2/residual dont make sense\33[0m", verbose = verbose, end = "\n")
            
            chi2 = [0,0,0,"", ""]
            show_gof = False
            chi_str = ""
        
        
        
        
        if show_gof is False:
            chi_str = ""
        if isinstance(ax, plt.Axes):
            # figure out whether the data is lin or log
            
            
            if xp is True:
                xp = lin_or_logspace(x, 1000)
                
            if xp is not False:
                qp(f"- calculating fit curve", verbose = verbose, end = "\n")
                yf = scale_plot*f(xp, *fit_out)
                color = ax.plot(xp, yf, label = f"{label}{chi_str}", color = color, **kwargs_plot)[0].get_color()
            
            if (show_fit_uncertainty is True) and callable(f.sf):
                try:
                    qp(f"- showing fit curve uncertainty", verbose = verbose, end = "\n")
                    qp(f"- len(fit_out): {len(fit_out)}", verbose = verbose, end = "\n")
                    qp(f"- cov_out.shape: {cov_out.shape}", verbose = verbose, end = "\n")
                    
                    s_yf = f.sf(xp, *fit_out, cov = cov_out)
                    qp(f"- type syf: {type(s_yf)}", verbose = verbose, end = "\n")
                    ax.fill_between(xp, yf-s_yf, yf+s_yf, color = color, alpha = .2, linewidth = 0)
                    qp(f"- curve drawn", verbose = verbose, end = "\n")
                except Exception as e:
                    qp(f"  - \33[31m{e}\33[0m", verbose = verbose, end = "\n")
            
            if show_f is True:
                qp(f"- adding description of f", verbose = verbose, end = "\n")
                addlabel(ax, f)
            
            if show_p0 is True:
                y0 = f(xp, *p0)
                ax.plot(xp, y0, color = color, linestyle = "dashed")
            
            if units is None:
                
                qp(f"- defaulting units", verbose = verbose, end = "\n")
                units = f.units
                
            if show_fit_result is True:
                qp(f"- adding fit results to plot", verbose = verbose, end = "\n")
                add_fit_parameters(ax, f, fit_out, sfit_out, units)
        
        qp(f"- preparing output list", verbose = verbose, end = "\n")
        out = [fit_out, sfit_out, chi2, output]


        if return_cov is True:
            qp(f"- replacing sfit with cov", verbose = verbose, end = "\n")
            out[1] = cov_out
        
        if return_output is not True:
            qp(f"- removing full output", verbose = verbose, end = "\n")
            out = out[:3]

        qp(f"- returning all", verbose = verbose, end = "\n")
        
        if return_as_dict is True:
            out = dict(
                fit = fit_out,
                cov = cov_out,
                chi2 = chi2,
                output = output,
                p0 = p0_all,
            )
        
        return(out)
        
        
            
    except Exception as e:
        print(e)
        return(False)
    





def add_fits_result_to_df(df, f, fit_result, dict_append = None):
    if not isinstance(df, pd.core.frame.DataFrame):
        raise TypeError("df must be a pandas Dataframe")
    if not isinstance(f, fit_function):
        raise TypeError("f must be a fit_function object")
    if not isinstance(dict_append, dict):
        dict_append = {}
    
    out = {**dict_append}
    sfit = [False] * len(f)
    if isinstance(fit_result, dict):
        fit = fit_result["fit"]
        
        if "chi2" in fit_result:            
            out["chi2"] = fit_result["chi2"][2]
        
        if "cov" in fit_result:
            out["cov"] = fit_result["cov"]
            if "sfit" not in fit_result:
                sfit = np.diag(out["cov"])**.5
        
        if "sfit" in fit_result:
            sfit = fit_result["sfit"]
        
    
    elif len(fit_result) == 2:
        fit, sfit = fit_result
    elif len(fit_result) == 3:
        fit, sfit, chi2 = fit_result
        out["chi2"] = chi2[2]
    elif len(fit_result) == 4:
        fit, sfit, chi2, cov = fit_result
        out["chi2"] = chi2[2]
        out["cov"] = cov
    else:
        raise TypeError
    out["fit"] = fit
    
    
    
    for p, v, sv in zip(f.parameters, fit, sfit):
        out[f"result_{p}"] = v
        out[f"result_s{p}"] = sv
    df = df.append(out, ignore_index = True)
    return(df)



# constant function
def f_poly_0(x, c):
    return(0*x+c)
    
def f_p0_poly_0(x, y):
    x_, y_ = clean(x, y)
    return([np.mean(y)])

def f_spoly_0(x, c, sfit = False, cov = False):
    if (sfit is False) and (cov is False):
        print("\33[31mno uncertaintys given for f_spoly_1 (set either sfit or cov)\33[0m")
        return(np.zeros_like(x))
    if (sfit is False):
        sc = cov[0,0]**.5
    else:
        sc = sfit
    return(sc)

    
poly_0 = fit_function(
    f = f_poly_0,
    sf = f_spoly_0,
    f_p0 = f_p0_poly_0,
    description = "constant function",
    short_description = "c",
    parameters = ["c"],
    parameters_tex = ["c"],
    formula = "c",
    formula_tex = "$c$",
    docstring = ''''''
) 




# first order polynomial
def f_poly_1(x, m, c):
    return(m*x+c)
    
def f_p0_poly_1(x, y):
    x_, y_ = clean(x, y)
    return(np.polyfit(x_, y_, deg = 1))
    
def f_spoly_1(x, m, c, sfit = False, cov = False):
    if (sfit is False) and (cov is False):
        print("\33[31mno uncertaintys given for f_spoly_1 (set either sfit or cov)\33[0m")
        return(np.zeros_like(x))
    if (cov is False):
        cov_ = 0
    else: 
        cov_ = cov[0,1]
    if (sfit is False):
        sm, sc = np.diag(cov)**.5
    else:
        sm, sc = sfit
    return((sm**2 * x**2 + sc**2 + 2*cov_*x)**.5)

poly_1 = fit_function(
    f = f_poly_1,
    sf = f_spoly_1,
    f_p0 = f_p0_poly_1,
    description = "first order polynomial",
    short_description = "lin + c",
    parameters = ["m", "c"],
    parameters_tex = ["m", "c"],
    formula = "mx + c",
    formula_tex = "$m x + c$",
    docstring = ''''''
) 



# second order polynomial
def f_poly_2(x, a2, a1, a0):
    return( a2*x**2 + a1*x + a0 )
    
def f_p0_poly_2(x, y):
    x_, y_ = clean(x, y)
    return(np.polyfit(x_, y_, deg = 2))



def f_spoly_2(x, a2, a1, a0, sfit = False, cov = False):
    if (sfit is False) and (cov is False):
        print("\33[31mno uncertaintys given for f_spoly_1 (set either sfit or cov)\33[0m")
        return(np.zeros_like(x))
    if (cov is False):
        cov_01 = cov_02 = cov_12 = 0
    else: 
        cov_01, cov_02, cov_12 = cov[0,1],cov[0,2],cov[1,2]
        
    if (sfit is False):
        sa2, sa1, sa0 = np.diag(cov)**.5
    else:
        sa2, sa1, sa0 = sfit
    
    d2 = x**2
    d1 = x
    d0 = 1
     
    return((
          (d2 * sa2)**2 + (d1 * sa1)**2 + (d0 * sa0)**2
        + cov_01*d0*d1 + cov_02*d0*d2 + cov_12*d1*d2
        
    )**.5)

    
poly_2 = fit_function(
    f = f_poly_2,
    sf = f_spoly_2,
    f_p0 = f_p0_poly_2,
    description = "second order polynomial",
    short_description = "second order polynomial",
    parameters = ["a2", "a1", "a0"],
    parameters_tex = ["a_2", "a_1", "a_0"],
    formula = "a2 x² + a1 x + a0",
    formula_tex = "$a_2 x^2 + a_1 x + a_0$",
    docstring = ''''''
) 



def f_parabola(x, mu, a, c):
    return(a*(x-mu)**2+c)

def f_p0_parabola(x, y):
    a2, a1, a0 = np.polyfit(x, y, 2)
    a = a2
    mu = -a1/(2*a2)
    c = a0 - a2 * mu**2    
    return(mu, a, c)
    
    
parabola = fit_function(
    f = f_parabola,
    f_p0 = f_p0_parabola,
    description = "barbolic fit with mu as high/low point instead of default polynomial",
    short_description = "parabolic fit",
    parameters = ["mu, a", "c"],
    parameters_tex = ["\\mu", "a", "c"],
    formula = "a*(x-mu)**2+c",
    formula_tex = "$a(x-\\mu)^2+c$",
    docstring = ''''''
) 





# exponential decay with constant
def f_exp_decayC(t, A, tau, C):
    return(A*np.exp(-t/tau)+C)

def f_p0_exp_decayC(x, y, tau_lit = 1):
    return([y[0]/f_exp_decayC(x[0], 1, tau_lit,0), tau_lit, 0])




exp_decayC = fit_function(
    f = f_exp_decayC,

    f_p0 = f_p0_exp_decayC,
    description = "exponential decay with constant term",
    short_description = "decay+C",
    parameters = ["A", "tau", "C"],
    parameters_tex = ["A", "\\tau", "C"],
    formula = "A exp(t / tau) + C",
    formula_tex = "$A \\cdot \\exp{{( t / \\tau)}} + C$",
    docstring = '''
An exponential decay with a constant offset

usage: y = exp_decayC(x; A, tau, C)
'''
)


# exponential decay withOUT constant
def f_exp_decay(t, A, tau):
    return(A*np.exp(-t/tau))

def f_p0_exp_decay(x, y, tau_lit = 1):
    return([y[0]/f_exp_decayC(x[0], 1, tau_lit, 0), tau_lit])



def f_sexp_decay(t, A, tau, sfit = False, cov = False):
    if (sfit is False) and (cov is False):
        print("\33[31mno uncertaintys given for f_sexp_decay (set either sfit or cov)\33[0m")
        return(np.zeros_like(x))
    if (cov is False):
        cov_ = 0
    else: 
        cov_ = cov[0,1]
    if (sfit is False):
        sA, stau = np.diag(cov)**.5
    else:
        sA, stau = sfit
        
    
    return(
        (
              (sA   * np.exp(-t/tau))**2
            + (stau * A * t / tau**2 * np.exp(-t/tau))**2
            + (2 * np.exp(-t/tau) * A * t / tau**2 * np.exp(-t/tau) * cov_)
        )**.5
    )

exp_decay = fit_function(
    f = f_exp_decay,
    sf = f_sexp_decay,
    f_p0 = f_p0_exp_decay,
    description = "exponential decay",
    short_description = "decay",
    parameters = ["A", "tau"],
    parameters_tex = ["A", "\\tau"],
    formula = "A exp(-t / tau)",
    formula_tex = "$A \\cdot \\exp{{(-t / \\tau)}}$",
    docstring = '''
An exponential decay that coverges to zero

usage: y = exp_decayC(x; A, tau)
'''
)


# errofunctions
def f_erf(x, mu, sigma, y0, y1):
    return(
        y0 +((scipy.special.erf((x-mu)/sigma/(2**.5))+1)/2) * (y1-y0)
    )
def f_p0_erf(x, y):
    x_, y_ = clean(x, y)
    x_, y_ = sort(x_, y_)
    n = len(y)
    y0 = np.mean(y_[:int(n/5)])
    y1 = np.mean(y_[int(n/5*4):])
    dy = y1-y0
    y2_ = (y_ - y0)/dy

    mu, s0, s1 = np.interp([0.5, .16, .84], y2_, x_)
    sigma = (s1-s0)*(np.pi*2)**-.5

    p0 = np.array([mu, sigma, y0, y1])

    return(p0)


erf = fit_function(
    f = f_erf,
    f_p0 = f_p0_erf,
    description = "error function",
    parameters = ["mu", "sigma", "y0", "y1"],
    parameters_tex = ["\\mu", "\\sigma", "y_{{0}}", "y_{{1}}"],
    formula = "erf(x; mu, sigma, y0, y1)",
    formula_tex = "$\\erf(x; \\mu, \\sigma, y_{{0}}, y_{{1}})$",
    docstring = '''
    an error function implementation
'''
)
def f_erf_lin(x, mu, sigma, y0, y1, a):
    return(
        (y0 +((scipy.special.erf((x-mu)/sigma/2)+1)/2) * (y1-y0)) * (a*x + 1)
    )
def f_p0_erf_lin(x, y):
    x_, y_ = clean(x, y)
    x_, y_ = sort(x_, y_)
    n = len(y)
    y0 = np.mean(y_[:int(n/5)])
    y1 = np.mean(y_[int(n/5*4):])
    dy = y1-y0
    y2_ = (y_ - y0)/dy

    mu, s0, s1 = np.interp([0.5, .16, .84], y2_, x_)
    sigma = (s1-s0)*(np.pi*2)**-.5

    p0 = np.array([mu, sigma, y0, y1, 0])

    return(p0)


erf_lin = fit_function(
    f = f_erf_lin,
    f_p0 = f_p0_erf_lin,
    description = "error function multiplied by (a*x + 1)",
    parameters = ["mu", "sigma", "y0", "y1", "a"],
    parameters_tex = ["\\mu", "\\sigma", "y_{{0}}", "y_{{1}}", "a"],
    formula = "erf(x; mu, sigma, y0, y1,)*(a*x + 1)",
    formula_tex = "$\\erf(x; \\mu, \\sigma, y_{{0}}, y_{{1}}) (a x + 1)$",
    docstring = '''
    an error function
'''
)



# sigmoid function
# legacy things
meta = {
    "sigmoid": {
        "params": ["mu", "sigma", "y0", "y1"],
        "params_tex": ["\\mu", "\\sigma", "y_{{0}}", "y_{{1}}"],
        "params_unit": ["µs", "µs", "", ""],
        "formula": "$\\frac{{(y_1 - y_0)}}{{1+\\frac{{exp((\\mu-x)}}{{sigma}}}} + y_0$",
    },
    
}

def f_sigmoid(x, mu=0, sigma=1, y0=0, y1=1):
    '''
    a simple sigmoid function based on 1/(1+exp(x))
    
    centre of slpoe: mu
    width of slope: sigma (~central 46%)
    f(mu-sigma) = 0.269
    y0: start value of funciton
    y1: end value of function
    
    '''
    return(
        (y1-y0)/(1+(np.exp((mu-np.array(x))/sigma)))+y0
    )

def f_ssigmoid(x, mu, sigma, y0, y1, sfit = False, cov = False):
    if (sfit is False) and (cov is False):
        print("\33[31mno uncertaintys given for f_ssigmoid (set either sfit or cov)\33[0m")
        return(np.zeros_like(x))
    if (cov is False):
        cov_musigma = cov_muy0 = cov_muy1 = cov_sigmay0 = cov_sigmay1 = cov_y0y1 = 0
    else: 
        cov_musigma, cov_muy0, cov_muy1, cov_sigmay0, cov_sigmay1, cov_y0y1 = \
        cov[0,1], cov[0,2], cov[0,3], cov[1,2], cov[1,3], cov[2,3]
    if (sfit is False):
        smu, ssigma, sy0, sy1 = np.diag(cov)**.5
    else:
        smu, ssigma, sy0, sy1 = sfit
    
    
    d_mu = sigma * (y0-y1)*np.exp(mu+x) / (np.exp(mu) + sigma * np.exp(x))**2
    d_sigma = (y1-y0)*np.exp(mu + x) / (np.exp(mu + sigma * np.exp(x)))**2
    d_y0 = np.exp(mu) / (np.exp(mu) + sigma * np.exp(x))
    d_y1 = sigma * np.exp(x) / (np.exp(mu) + sigma * np.exp(x))
    
    d_mu[~np.isfinite(d_mu)] = 0
    d_sigma[~np.isfinite(d_sigma)] = 0
    d_y0[~np.isfinite(d_y0)] = 0
    d_y1[~np.isfinite(d_y1)] = 0
    
    return(
        (
              (d_mu * smu)**2
            + (d_sigma * ssigma)**2
            + (d_y0 * sy0)**2
            + (d_y1 * sy1)**2
            + (d_mu * d_sigma * cov_musigma)
            + (d_mu * d_y0 * cov_muy0)
            + (d_mu * d_y1 * cov_muy1)
            + (d_sigma * d_y0 * cov_sigmay0)
            + (d_sigma * d_y1 * cov_sigmay1)
            + (d_y0 * d_y1 * cov_y0y1)
        )**.5
    )


def f_p0_sigmoid(x, y):
    return([
        x[np.argmax(np.abs(np.diff(y))/np.diff(x))], # mu
        (x[-1]-x[0])/20, # sigma
        np.mean(y[:3]),  # y0
        np.mean(y[-3:])] # y1
    )

sigmoid = fit_function(
    f = f_sigmoid,
    sf = f_ssigmoid,
    f_p0 = f_p0_sigmoid,
    description = "sigmoid function",
    parameters = ["mu", "sigma", "y0", "y1"],
    parameters_tex = ["\\mu", "\\sigma", "y_{{0}}", "y_{{1}}"],
    formula = "(y1-y0)/(1+exp(mu-sigma)/sigma) + y0",
    formula_tex = "$\\frac{{(y_1 - y_0)}}{{1+\\frac{{exp((\\mu-x)}}{{\\sigma}}}} + y_0$",
    docstring = '''
A sigoid function that goes from y0 to y1 via a spread of "width" sigma around mu

usage: y = sigmoid(x; mu, sigma, y1, y0)
'''
)

def f_sigmoid_lin(x, mu=0, sigma=1, y0=0, y1=1, a=0):
    '''
    modified sigmoid function that is multiplied by a first order polynomial
    
    centre of slope: mu
    width of slope: sigma (~central 46%)
    f(mu-sigma) = 0.269
    y0: start value of funciton
    y1: end value of function
    a: linear term of polynomial
    '''
    return(
        ((y1-y0)/(1+(np.exp((mu-np.array(x))/sigma)))+y0) * (a*x + 1)
    )



def f_p0_sigmoid_lin(x, y):
    return([
        3.5, # mu
        0.3, # sigma
        np.mean(y[:3]),  # y0
        np.mean(y[-3:]), # y1
        0.015,           # a  
    ])

sigmoid_lin = fit_function(
    f = f_sigmoid_lin,
    f_p0 = f_p0_sigmoid_lin,
    description = "sigmoid function multipliey by polynomial",
    parameters = ["mu", "sigma", "y0", "y1", "a"],
    parameters_tex = ["\\mu", "\\sigma", "y_{{0}}", "y_{{1}}", "a"],
    formula = "((y1-y0)/(1+exp(mu-sigma)/sigma) + y0) * (a*t + 1)",
    formula_tex = "$(\\frac{{(y_1 - y_0)}}{{1+\\frac{{exp((\\mu-x)}}{{\\sigma}}}} + y_0) (a\\cdot x + 1)$",
    docstring = '''
A sigoid function that is multiplied by (a*t + 1)

usage: y = sigmoid(x; mu, sigma, y1, y0, a)
'''
)




# gauss function
def f_gauss(x, A=1, mu=0, sigma=1):
    return(
        A * np.exp(-(np.array(x)-mu)**2 / (2 * sigma**2))
    )
def f_p0_gauss(x, y):
    A = max(y)
    ycs = np.cumsum(y) / np.sum(y)
    bounds = np.interp([.16, .84], ycs, x)
    sigma = np.abs(np.diff(bounds)[0]/2)
    return(
        A,
        x[np.argmax(y)],
        sigma
    )
    
gauss = fit_function(
    f = f_gauss,
    f_p0 = f_p0_gauss,
    description = "gauss function without constant term",
    short_description  = "gauss function", 
    parameters = ["A", "mu", "sigma"],
    parameters_tex = ["A", "\\mu", "\\sigma"],
    formula = "A exp(-(x-mu)^2 /(2 sigma^2))",
    formula_tex = "$A \\exp{{\\frac{{-(x-\\mu)^2}}{{2 \\sigma^2}}}}$",
)

# doublegauss function
def f_gauss2(x, A1=1, mu1=0, sigma1=1, A2=1, mu2=0, sigma2=1):
    return(
          A1 * np.exp(-(np.array(x)-mu1)**2 / (2*sigma1**2))
        + A2 * np.exp(-(np.array(x)-mu2)**2 / (2*sigma2**2))
    )
def f_p0_gauss2(x, y):
    
    A1 = max(y)
    A2 = max(y)
    ycs = np.cumsum(y) / np.sum(y)
    mu1, mu2, b1l, b1u, b2l, b2u = np.interp([.25, .75, .17, .33, .67, .83], ycs, x)
    
    sigma1 = b1u-b1l
    sigma2 = b2u-b2l

    
    return(A1, mu1, sigma1, A2, mu2, sigma2)
    
    
gauss2 = fit_function(
    f = f_gauss2,
    f_p0 = f_p0_gauss2,
    description = "sum of two gaus functions without a constant term",
    short_description  = "double gauss",
    parameters = ["A1", "mu1", "sigma1", "A2", "mu2", "sigma2"],
    parameters_tex = ["A_1", "\\mu_1", "\\sigma_1", "A_2", "\\mu_2", "\\sigma_2"],
    formula = "A1 exp(-(x-mu1)^2 /(2 sigma_1^2)) + A2 exp(-(x-mu2)^2 /(2 sigma_2^2))",
    formula_tex = "$A_1 \\exp{{\\frac{{-(x-\\mu_1)^2}}{{(2 \\sigma_1^2)}}}} + A_2 \\exp{{\\frac{{-(x-\\mu_2)^2}}{{(2 \\sigma_2^22}}}}$",
)

# gauss function
def gaussC(x, A=1, mu=0, sigma=1, C = 0):
    return(
        A * np.exp(-(np.array(x)-mu)**2 / (2*sigma**2)) + C
    )
def f_p0_gaussC(x, y):
    return(max(y), x[np.argmax(y)], (x[1]-x[0])/5, 0)
    
    
gaussC = fit_function(
    f = gaussC,
    f_p0 = f_p0_gaussC,
    description = "gauss function with constant term",
    parameters = ["A", "mu", "sigma", "C"],
    parameters_tex = ["A", "\\mu", "\\sigma", "C"],
    formula = "A exp(-(x-mu)^2 /(2 sigma)^2)+C",
    formula_tex = "$A \\exp{{\\frac{{-(x-\\mu)^2}}{{(2 \\sigma^2)}}}}+C$",
)



# gauss function
def f_expgauss(x, A, tau, C=0, B=1, mu=0, sigma=1):
    return(
        A*np.exp(-x/tau) + B * np.exp(-(np.array(x)-mu)**2 / (2*sigma**2)) + C
    )

def f_p0_expgauss(x, y, tau_lit = 1, mu_lit = 1600, sigma_lit = 100):
    return([
        y[0]/f_exp_decayC(x[0], 1, tau_lit, 0), # A
        tau_lit, # tau duhh..
        0, # C
        max(y[(x > mu_lit - sigma_lit) & (x <= mu_lit + sigma_lit)]), # B
        mu_lit, 
        sigma_lit,
    ])
    
    
expgauss = fit_function(
    f = f_expgauss,
    f_p0 = f_p0_expgauss,
    description = "exponential decay plus gauss",
    short_description = "exp decay + gauss + C",
    parameters = ["A", "tau", "C", "B", "mu", "sigma"],
    parameters_tex = ["A", "\\tau", "C", "B", "\\mu", "\\sigma"],
    formula = "A exp(t / tau) + B exp(-(t-mu)^2 /(2 sigma)^2) + C",
    formula_tex = "$A \\cdot \\exp{{( t / \\tau)}} + B \\exp{{\\frac{{-(t-\\mu)^2}}{{(2 \\sigma^2)}}}} + C$",
)


# exponential decay with turn on
def f_decay_to(t, t_0, tau, a, A):
    return(A * 1/(1+np.exp((t_0-t)/a)) * np.exp((t_0-t)/tau))

def f_p0_decay_to(x, y, tau_lit = 150, a_lit = 10):
    return([
        x[np.argmax(y)],
        tau_lit,
        a_lit,
        max(y)
    ])
expdecay_to = fit_function(
    f = f_decay_to,
    f_p0 = f_p0_decay_to,
    description = "exponential decay including turn on fucntion",
    short_description = "exp decay * turnon",
    parameters = ["t0", "tau", "a", "A"],
    parameters_tex = ["t_0", "\\tau", "a", "A"],
    formula = "A/(1+ exp((t_0 - t)/a)) np.exp((t_0-t)/tau)",
    formula_tex = "$\\frac{{A exp((t_0-t)/\\tau)}}{{(1 + \\exp((t_0 - t)/a))}} $",
)

# normal distributiuon
def f_normal(x, mu = 0, sigma = 1):
    return(
        1/(sigma*(2*np.pi)**.5)*np.exp(-1/2 * ((x-mu)/(sigma))**2)
    )
def f_p0_normal(x, y):
    return([
        x[np.argmax(y)],
        (np.diff(np.interp([.16, .84], np.cumsum(y), x))/2)[0],
    ])
    
normal = fit_function(
    f = f_normal,
    f_p0 = f_p0_normal,
    description = "normal distribution (normalized gauss distribution)",
    short_description = "normal distribution",
    parameters = ["mu", "sigma"],
    parameters_tex = ["\\mu", "\\sigma"],
    formula = "1/(σ √(2 π)) gauss(x;µ,σ)",
    formula_tex = "$\\frac{1}{\\sigma \\sqrt{{2 \\pi}}} \\exp{{-\\frac{{x-\\mu}}{{2\\sigma}}^2}}$",
)
    
# exponential with turn on function....
def f_exp_turn_on(t, t_0, tau, a, A):
    return(A * 1/(1+np.exp((t_0-t)/a)) * np.exp((t_0-t)/tau))

def f_p0_exp_turn_on(t, y, t_0=True, tau=True, a=True, A=True):
    
    order = np.argsort(t)
    x_ = t[order]
    y_ = y[order]
    
    y_cs = np.cumsum(y_)
    y_cs = y_cs/y_cs[-1]

    id_max = np.argmax(y_)
    
    diffs = np.interp([.05, 0.15], y_cs, x_)
    
    
    
    if A is True:
        A = y_[id_max]
    if a is True:
        a = diffs[1] - diffs[0]
    if t_0 is True:
        t_0 = diffs[1]
    if tau is True:
        y_ref = np.abs(A*np.exp(-1) - y_[id_max:])
        tau = t[np.argmin(y_ref)+id_max] - t[id_max]
        
        
        
    return(t_0, tau, a, A)


def f_sexp_turn_on(t, t_0, tau, a, A, sfit = False, cov = False):
    if (sfit is False) and (cov is False):
        print("\33[31mno uncertaintys given for f_spoly_1 (set either sfit or cov)\33[0m")
        return(np.zeros_like(x))
    if (cov is False):
        cov_01 = cov_02 = cov_03 = cov_12 = cov_13 = cov_23 = 0
    else: 
        cov_01, cov_02, cov_03, cov_12, cov_13, cov_23 = cov[0,1], cov[0,2], cov[0,3], cov[1, 2], cov[1,3], cov[2,3]
        
    if (sfit is False):
        st_0, stau, sa, sA = np.diag(cov)**.5
    else:
        st_0, stau, sa, sA = sfit
    
    # done with wolfram alpha
    dt_0 = A * (1/tau-1/a) * np.exp(-(t-t_0)*(a-tau)/a*tau)
    dtau = A * (t-t_0) * np.exp(-(t-t_0)*(a-tau)/a * tau)/tau**2
    da =   A * (t_0-t) * np.exp((t_0-t)/tau - (t_0-t)/a)/a**2
    dA =   1/(1+np.exp((t_0-t)/a)) * np.exp((t_0-t)/tau)
    
    return((
          (dt_0**2 * st_0**2) + (dtau**2 * stau**2) + (da**2 * sa**2) + (dA**2 * sA**2)
        + 2*cov_01*dt_0*dtau + 2*cov_02*dt_0*da + 2*cov_03*dt_0*dA
        + 2*cov_12*dtau*da + 2*cov_13*dtau*dA + 2*cov_23*da*dA
        
    )**.5)

 


exp_turn_on = fit_function(
    f = f_exp_turn_on,
    sf = f_sexp_turn_on,
    f_p0 = f_p0_exp_turn_on,
    description = "exponential decay with turn on",
    short_description = "exponential decay with turn on",
    parameters = ["t_0", "tau", "a", "A"],
    parameters_tex = ["t_0", "\\tau", "a", "A"],
    formula = "A * 1/(1+np.exp((t_0-t)/a)) * np.exp((t_0-t)/tau)",
    formula_tex = "$A  1/(1+exp((t_0-t)/a)) exp((t_0-t)/\\tau)$",
    docstring = '''
'''
)




# diffusion function
def f_diffusion(t, D, sigma_0, v_d = 6.9/40_000):
    return(
        (2 * D/v_d**2 * t + sigma_0**2)**.5 
    )
def f_p0_diffusion(t, w, v_d = 6.9/40_000):
    return(np.median(w**2 / t)/2*v_d**2, np.min(w), v_d)

def f_sdiffusion(t, D, sigma_0, v_d, sfit=False, cov=False):
    if (sfit is False) and (cov is False):
        print("\33[31mno uncertaintys given for f_sdoke (set either sfit or cov)\33[0m")
        return(np.zeros_like(x))
    if (cov is False):
        cDs = cDv = csv = 0
    else: 
        cDs, cDv, csv = cov[0,1], cov[0,2], cov[1,2]
    
    if (sfit is False):
        sD, ss, sv = np.diag(cov)**.5
    else:
        sD, ss, sv = sfit
    
    
    dD = t / (v_d * ((2 * D * t / v_d) + sigma_0)**.5)
    ds = 1 / (2 * ((2 * D * t / v_d) + sigma_0)**.5)
    dv = -D*t / (v_d**2 * ((2 * D * t / v_d) + sigma_0)**.5)
    
    sy = (
          (dD * sD)**2
        + (ds * ss)**2
        + (dv * sv)**2
        + (dD * ds * cDs)
        + (dD * dv * cDv)
        + (ds * dv * csv)
    )**.5
    
    return(sy)
    
    
    
diffusion  = fit_function(
    f = f_diffusion,
    f_p0 = f_p0_diffusion,
    sf = f_sdiffusion,
    description = "Diffusion formula w = √(2Dt+w0²)",
    short_description = "",
    parameters = ["D", "sigma_0", "v_d"],
    parameters_tex = ["D", "\\sigma_0", "v_\\mathrm{{drift}}"],
    formula = "√(2D/v_d t+σ_0²)",
    formula_tex = "$\\sqrt{\\frac{{2 D\ t}}{{v_\\mathrm{{drift}}}}+\\sigma_0^2}$",
    units = ["cm²/s", "ns", "cm/ns"],
)



# doke function 
def f_doke(x, g1, g2):
    return(x * -g2/g1 + g2)

def f_p0_doke(x, y, g1 = True, g2 = True):
    x_ = np.array(x) 
    y_ = np.array(y)
    
    if not isinstance(g1, bool):
        x_ = np.concatenate((x_, [g1]))
        y_ = np.concatenate((y_, [0]))
    if not isinstance(g2, bool):
        x_ = np.concatenate((x_, [0]))
        y_ = np.concatenate((y_, [g2]))
    
    pf = np.polyfit(x_, y_, 1)
    
    m, mg2 = pf
    if g2 is True:
        g2 = mg2
    
    if g1 is True:
        g1 = -g2/m
    return(g1, g2)

def f_sdoke(x, g1, g2, sfit=False, cov=False):
    if (sfit is False) and (cov is False):
        print("\33[31mno uncertaintys given for f_sdoke (set either sfit or cov)\33[0m")
        return(np.zeros_like(x))
    if (cov is False):
        cg1g2 = 0
    else: 
        cg1g2 = cov[0,1]
    
    if (sfit is False):
        sg1, sg2 = np.diag(cov)**.5
    else:
        sg1, sg2 = sfit
    
    
    dg1 = g2/g1**2*x
    dg2 = 1-x/g1
    
    
    sy = (
          (dg1 * sg1)**2
        + (dg2 * sg2)**2
        + (dg1 * dg2 * cg1g2)
    )**.5
    
    return(sy)
    
    

doke  = fit_function(
    f = f_doke,
    sf = f_sdoke,
    f_p0 = f_p0_doke,
    description = "doke fit",
    short_description = "doke fit",
    parameters = ["g1", "g2"],
    parameters_tex = ["g_1", "g_2"],
    units = ["PE/γ", "PE/e"],
    formula = "S2 = S1 * (-g2/g1) + g2",
    formula_tex = "$S_2 = \\left(S1 \\frac{{- g_2}}{{g_1}} +g_2\\right)$",
    
)




def f_moyal(x, mu = 0, scale = 1, A = 1):
    y = (x-mu)/scale
    return(
        A/scale * 1/(2*np.pi)**.5  * np.exp(-(y + np.exp(-y))/2)
    )

    
# a landau approximation


def f_p0_moyal(x, y, mu = True, scale = True, A = True):
    
    order = np.argsort(x)
    x_ = x[order]
    y_ = y[order]
    ycs = np.cumsum(y_)
    ycs_n = ycs/ycs[-1]
    
    
    if mu is True:
        mu = x[np.argmax(y)]
    if scale is True:
        scale = np.diff(np.interp([0.25, 0.5], ycs_n, x_))[0]
    if A is True:
        A = np.max(y) * scale
    return(mu, scale, A)


moyal = fit_function(
    f = f_moyal,
    f_p0 = f_p0_moyal,
    description = "moyal distribution (landau approximation)",
    short_description = "moyal",
    parameters = ["mu", "scale", "A"],
    parameters_tex = ["\\mu", "\\mathrm{{scale}}", "A"],
    formula = "",
    formula_tex = "moyal",
    docstring = ''''''
) 



# exponential decay shifted to omit overflow errors
def f_exp_decay_shift(t, A, tau, mu = 0):
    return(A*np.exp(-(t-mu)/tau))

def f_p0_exp_decay_shift(t, y, A = True, tau = True, mu = True):
    order = np.argsort(t)
    x_ = t[order]
    y_ = y[order]
    
    y_cs = np.cumsum(y_)
    y_cs = y_cs/y_cs[-1]

    id_max = np.argmax(y_)
    
    diffs = np.interp([.05, 0.15], y_cs, x_)
    
    if A is True:
        A = y_[id_max]
    if mu is True:
        mu = diffs[1]
    if tau is True:
        y_ref = np.abs(A*np.exp(-1) - y_[id_max:])
        tau = t[np.argmin(y_ref)+id_max] - t[id_max]
    
    return(A, tau, mu)



exp_decay_shift = fit_function(
    f = f_exp_decay_shift,
    f_p0 = f_p0_exp_decay_shift,
    description = "exponential decay that is shifted (reduces overflows)",
    short_description = "exp decay shifted",
    parameters = ["A", "tau", "mu"],
    parameters_tex = ["A", "\\tau", "\\mu"],
    formula = "A exp(-(t-mu) / tau)",
    formula_tex = "$A \\cdot \\exp{{(-(t-\\mu) / \\tau)}}$",
    docstring = '''
An exponential decay that coverges to zero
usage: y = exp_decayC(x; A, tau, mu = 0)
'''
)



