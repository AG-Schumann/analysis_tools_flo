import numpy as np




class fit_function:
    def __init__(self, f, f_p0,
        description = "",
        short_description = "",
        parmaters = None, parmaters_tex = None,
        formula = "", formula_tex = None,
        docstring = ""
    ):
        self.f = f
        self.p0 = f_p0
        self.description = description
        self.short_description = short_description
        self.s = short_description
        self.__doc__ = docstring
        
        if parmaters is None:
            raise ValueError("parmaters for fucntion not given! use ")
        self.parmaters = parmaters
        if parmaters_tex is None:
            self.parmaters_tex = parmaters
        else:
            self.parmaters_tex = parmaters_tex
    
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
        for pt in self.parmaters_tex:
            yield(pt)






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
    parmaters = ["A", "tau", "C"],
    parmaters_tex = ["A", "\\tau", "C"],
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

exp_decay = fit_function(
    f = f_exp_decay,
    f_p0 = f_p0_exp_decay,
    description = "exponential decay",
    short_description = "decay",
    parmaters = ["A", "tau"],
    parmaters_tex = ["A", "\\tau"],
    formula = "A exp(t / tau)",
    formula_tex = "$A \\cdot \\exp{{( t / \\tau)}}$",
    docstring = '''
An exponential decay that coverges to zero

usage: y = exp_decayC(x; A, tau)
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



def f_p0_sigmoid(x, y):
    return([np.mean(x), (x[-1]-x[0])/10, np.mean(y[10]), np.mean(y[:-10])])

sigmoid = fit_function(
    f = f_sigmoid,
    f_p0 = f_p0_sigmoid,
    description = "sigmoid function",
    parmaters = ["mu", "sigma", "y0", "y1"],
    parmaters_tex = ["\\mu", "\\sigma", "y_{{0}}", "y_{{1}}"],
    formula = "(y1-y0)/(1+exp(mu-sigma)/sigma) + y0",
    formula_tex = "$\\frac{{(y_1 - y_0)}}{{1+\\frac{{exp((\\mu-x)}}{{sigma}}}} + y_0$",
    docstring = '''
A sigoid function that goes from y0 to y1 via a splot of "width" sigma around mu

usage: y = sigmoid(x; mu, sigma, y1, y0)
'''
)


# gauss function
def gaus(x, A=1, mu=0, sigma=1):
    return(
        A * np.exp(-(np.array(x)-mu)**2 / (2*sigma)**2)
    )
def f_p0_gaus(x, y):
    return([max(y)], x[np.argmax(y)], (x[1]-x[0])/10)
    
    
gauss = fit_function(
    f = gaus,
    f_p0 = f_p0_gaus,
    description = "gauss function without constant term",
    parmaters = ["A", "mu", "sigma"],
    parmaters_tex = ["A", "\\mu", "\\sigma"],
    formula = "A exp(-(x-mu)^2 /(2 sigma)^2)",
    formula_tex = "$A \\exp{{\\frac{{-(x-\\mu)^2}}{{(2 \\sigma)^2}}}}$",
)








    
