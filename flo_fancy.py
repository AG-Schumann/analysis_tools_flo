import numpy as np
from threading import Thread, Event

def qp(*args, sep = ", ", end = "", flush = True, verbose = True):
    if verbose is True: print(sep.join(map(str, args)), end = end, flush = flush)

def v_str(x, digits = "8.2"):
    if x < 1000:
        return(f'{x:{digits}f}')
    else:
        return(f'{x:{digits}g}')


def string_join(*args, sep = ""):
    return(sep.join([str(arg) for arg in args if arg != ""]))


def inv_small(x, ref = 1):
    x_ = x*1
    id_small = (np.abs(x) < ref) & (x != 0)
    x_[id_small] = 1/x[id_small]
    return(x_)


def logspace_over(x, N):
    x_,*_ = clean(x)
    x_ = x_[x_ > 0]
    x0 , x1 = np.min(x_), np.max(x_)
    return(np.logspace(*np.log10([x0, x1]), N))
        
def linspace_over(x, N):
    x_, *_ = clean(x)
    x0 , x1 = np.min(x_), np.max(x_)
    return(np.linspace(x0, x1, N))




def lin_or_logspace(x, N = 1000, show_ratio = False, linlog_thr = 1, return_str = False):
    '''
    checks if data is distributed log or lin
    
    set force to "lin" or "log" to force that
    
    '''
    is_lin = True
    
    
    if (min(x) > 0):
        x_ = np.sort(x)
        linlog = (
            len(np.unique(np.round(np.log(np.diff(x_)))))/
            len(np.unique(np.round(np.log(np.diff(np.log(x_))))))
        )
        if show_ratio is True:
            print(linlog)
        if (linlog >= linlog_thr):
            is_lin = False
            
    elif show_ratio is True:
        print("negative numbers found, using linscale")
        is_lin  = True
    # fallback to linspace
    
    
    if return_str is True:
        if is_lin is True:
            return("lin")
        else:
            return("log")
    else:
        if is_lin is True:
            return(linspace_over(x, N))
        else:
            return(logspace_over(x, N))
    


def enumezip(*args):
    '''
    generator that can be used when a zip-generator should also yield the index
    '''
    for i, argsi in enumerate(zip(*args)):
        yield(i, *argsi)

def flatten_dict(d, sep = "__", path = "", *args, **kwargs):
    '''
    flattens a dictionary recursevily
    '''

    out = {}
    for n, x in d.items():
        if path == "":
            path_ = f"{n}"
        else:
            path_ = f"{path}{sep}{n}"
        if isinstance(x, dict):
            out = {**out, **flatten_dict(x, sep = sep, path = f"{path_}")}
        else:
            out[path_]  = x
    return(out)
    

def addlabel(ax, label, color = "black", linestyle = "", marker = "", *args, **kwargs):
    ax.plot([], [], label = label, color = color, linestyle=linestyle, marker=marker, *args, **kwargs)



def add_fit_parameter(ax, l, p, sp=None, unit="", unit_tex="", fmt="auto"):
    '''
    adds nicely formated fit results to legend
    
    l: label
    p: parameter
    sp: uncertainty
    u: unit
    fmt: format (auto: use get_nice_format)
    
    '''
    
    str_value = tex_value(x = p, sx = sp, unit = unit, unit_tex = unit_tex)
    str_ = f"${l}: {str_value}$"
    
    addlabel(ax, str_)
    
    return(None)
    
    


def add_fit_parameters(ax, pars, fit, sfit = False, units = False, **kwargs):
    if sfit is False:
        sfit = [None]*len(pars)
    if units is False:
        units = [""]*len(pars)
    for par, v, sv, u in zip(pars, fit, sfit, units):
        if isinstance(u, str): 
            add_fit_parameter(ax, l = par, p = v, sp = sv, unit = u, **kwargs)


    

def tex_number(x, lim = 5, digits = 2):
    exp = np.floor(np.log10(np.abs(x)))
    if (np.abs(exp) >= lim) or (exp <= -digits):
        x_ = np.round(x * 10**(-exp), digits)
        return(f"{x_} \\cdot 10^{{{int(exp)}}} ")
    
    return(f"{x:.{digits}f}")

def tex_number_exp(x, exp = 0, digits = 2, prefix = ""):
    if prefix != "":
        prefix = f" {prefix} "
    
    if x is None:
        return("")
    if np.isfinite(x):
        
        
        if np.abs(exp) < digits:
            exp = 0
            x_ = x
        else:
            x_ = x * 10**(-exp)
        str_x = f"{x_:.{digits}f}"           
        str_exp = ""
        if exp != 0:  
            str_exp = f"\\cdot 10^{{{int(exp)}}}"

        return(f"{prefix}{str_x}{str_exp}")
    elif x == np.nan:
       return(f"{prefix}\\mathrm{{NaN}}")
    elif x == np.inf:
        return(f"{prefix}\\infty")
    elif x == -np.inf:
        if prefix == " \\pm ":
            return(f" \\mp \\infty")
        return(f"{prefix}-\\infty")

    return(f"{prefix}{x}")


def get_exp(x, digits = 2):
    if x is None:
        return(np.nan)
    x_ = np.floor(np.log10(np.abs(x)))
    if np.isfinite(x_):
        exp = int(x_)
        return(exp)
    return(0)



def tex_value(x, sx = None, unit = "", max_exp_diff = 4, lim = (-1,2), digits = 1, unit_tex = "", zero_lim = -8, v = False):
    
    if sx == 0:
        sx = None
    
    bool_bracket = [False, False]
    str_x = str_sx = str_exp = str_unit = str_unit = str_brl =  str_brr = ""
    
    
    if isinstance(sx, str):
        sx = None
    
    
    exp_x = get_exp(x, digits = digits)
    exp_sx = get_exp(sx, digits = digits)
    exp = False
    
    qp(f"\n exp_x:  {exp_x}", verbose = v)
    qp(f"\n exp_sx: {exp_sx}", verbose = v)
    
    if (exp_sx > exp_x) and (exp_x < zero_lim) and (sx is not None) and np.isfinite(sx):
        exp_x = exp_sx
        qp("\n set exp_x to exp_sx", verbose = v)
        
    
    if np.abs(exp_x - exp_sx) < max_exp_diff:
        # numbers are close
        qp("\n  numbers are close: ", verbose=v)
        exp = int(np.min([exp_x, exp_sx]))
        qp(f"\n  lim[0] < exp < lim[1]: {lim[0] < exp < lim[1]}", verbose=v)
        if lim[0] <= exp <= lim[1]:
            exp = 0
        qp(f"\n  exp: {exp}", verbose=v)
        
        x = x*10**(-exp)
        str_x = tex_number_exp(x, 0, digits = digits)
        
        if sx is not None:
            sx = sx*10**(-exp)
            str_sx = tex_number_exp(sx, 0, digits = digits, prefix = "\\pm")
        
        if exp != 0:
            str_exp = f"\\cdot 10^{{{int(exp)}}}"
    else:
        qp("\n  numbers are not close: ", verbose=v)
        if lim[0] <= exp_x <= lim[1]:
            qp(f"\n exp_x within limits: {lim[0]} < {exp_x} < {lim[1]}", verbose=v)
            exp_x = 0
        
        qp(f"\ntex_number_exp({x}, exp = {exp_x}, digits = {digits})", verbose=v)
        str_x = tex_number_exp(x, exp = exp_x, digits = digits)
        if sx is not None:
            if lim[0] <= exp_sx <= lim[1]:
                exp_sx = 0
            str_sx = tex_number_exp(sx, exp_sx, digits = digits, prefix = "\\pm")
        
    if sx is None:
        str_sx = ""
    
    
    
    if str_sx != "":
        bool_bracket[0] = True
    if str_exp != "":
        bool_bracket[1] = True
    if unit != "":
        bool_bracket[1] = True
        str_unit = f"\,\\mathrm{{{unit}}}" 
    if unit_tex != "":
        bool_bracket[1] = True
        str_unit = f"\, {unit_tex}" 

    if False not in bool_bracket:
        str_brl = "\\left("
        str_brr = "\\right)"


    out = f"{str_brl}{str_x}{str_sx}{str_brr}{str_exp}{str_unit}"
    return(out)
    
    
    
    
def calc_exp_integral_normalized(tau, t0 = 0, t1 = np.inf):
    '''
    Analytical calculation of the normalized integral over exp(-t/tau) from t0 (0) to t1 (inf).
    Multiply by tau to get the actual integral.
    '''
    return(
        np.exp(-t0/tau) - np.exp(-t1/tau)
    )
    
    
    

def get_nice_format(*x):
    '''
    returns ONE fmt string for all numbers to make them look the same
    '''
    y = np.array(x).reshape(-1)
    y = np.array(np.abs(y[y != None]), dtype = float)
    y = y + 1*(y==0)
    log_y = np.log10(y)
    
    decis = np.array([np.ceil(max(log_y)), np.floor(min(log_y)-1)])
    
    
    if decis[0] - decis[1] <= 5:
        
        # ignore difference, they are in the same leauge
        if decis[1] < 0:
            return(f".{-decis[1]:.0f}f")
        if decis[1] < 3:
            # default case 
            return(f".1f")
        if decis[1] < 5:
            # default case 
            return(f".0f")
        return(f".2e")
    
    # large difference or large numbers, use dynamic mode 
    return(".3g")
    




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
        
        
        
        

def sort(x, *args):
    '''
    sorts all inputs by the order of x
    '''
    idx = np.argsort(x)
    out  = [x[idx]]
    
    for arg in args:
        out.append(arg[idx])
        
    return(out)


def clean(*args):
    _ = np.prod([np.isfinite(argi) for argi in args if argi is not None], axis = 0)
    _, *argsf = remove_zero(_, *args)
    return(argsf)


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
            if len(np.shape(xi)) == 2:
                out.append([xj[idx_keep] for xj in xi])
            else:
                out.append(np.array(xi)[idx_keep])
        else:
            out.append(None)
    if len(args) > 0:
        return(x, *out)
    else:
        return(x)




def res_sqr(f, x, y, *pars):
    res = np.sum((f(x, *pars) - y)**2)
    ndf = len(x) - len(pars)
    
    str_res = tex_number(res)
    str_ndf = f"{ndf:.0f}"
    str_res_red = tex_number(res/ndf)
    
    return(
        (res, ndf, res/ndf,
            f"reduced res.² = {res:.2g}/{ndf:.0f} = {res/ndf:.2g}",
            f"res$^2_\\mathrm{{red}} = {str_res}/{str_ndf} = {str_res_red}$"
        )
    )


def chi_sqr(f, x, y, s_y, *pars, ndf = False, ignore_zeros = False, **kwargs):
    '''
    returns a tuple with chi^2, ndf and reduced chi^2
    
    parameters:
    
    f: the fucntion that was fitted
    x, y: x and y values of the data
    s_y: the uncertainties of the data
    (can be two arrays for lower and upper encertainties)
    
    
    
    '''
    x = np.array(x)
    y = np.array(y)
    s_y = np.array(s_y)
    y_f = f(x, *pars)
    
    
    
    if len(np.shape(s_y)) == 2:
        s_y = s_y[0]*(y_f < y)+s_y[1]*(y_f >= y)
    if ignore_zeros is True:
        y, s_y, x, y_f = remove_zero(y, s_y, x, y_f)
    if ndf is False:
        ndf = len(x) - len(pars)
    
    
    chi = float(np.sum(((y - y_f)/s_y)**2))
    
    return(
        (chi, ndf, chi/ndf,
            f"reduced chi² = {chi:.1f}/{ndf:.0f} = {chi/ndf:.1f}",
            f"$\\chi^2_\\mathrm{{red}} = {chi:.1f}/{ndf:.0f} = {chi/ndf:.1f}$"
        )
    )



def call_maxtime(f, *args, max_time = 10, **kwargs):
    ret = False
    
    def wrapper(*args, **kwargs):
        nonlocal ret
        try:
            ret = f(*args, **kwargs)
        except Exception as e:
            ret = e
        
    
    
    action_thread = Thread(target=wrapper, args = args, kwargs=kwargs)

    action_thread.start()
    action_thread.join(timeout = max_time)
    
    
    return(ret)
