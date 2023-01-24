import numpy as np
import inspect

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
        short_description = "",
        parameters = None, parameters_tex = None,
        formula = "", formula_tex = None,
        docstring = ""
    ):
        self.f = f
        self.p0 = f_p0
        self.description = description
        self.short_description = short_description
        self.s = short_description
        self.__name__ = f.__name__
        self.__doc__ = docstring
        
        
        if parameters is None:
            raise ValueError("parmaters for fucntion not given! use ")
        self.parameters = parameters
        if parameters_tex is None:
            self.parameters_tex = parameters
        else:
            self.parameters_tex = parameters_tex
    
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

exp_decay = fit_function(
    f = f_exp_decay,
    f_p0 = f_p0_exp_decay,
    description = "exponential decay",
    short_description = "decay",
    parameters = ["A", "tau"],
    parameters_tex = ["A", "\\tau"],
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
    return([
        x[np.argmax(np.abs(np.diff(y))/np.diff(x))], # mu
        (x[-1]-x[0])/20, # sigma
        np.mean(y[:3]),  # y0
        np.mean(y[-3:])] # y1
    )

sigmoid = fit_function(
    f = f_sigmoid,
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
        x[np.argmax(np.abs(np.diff(y))/np.diff(x))], # mu
        (x[-1]-x[0])/20, # sigma
        np.mean(y[:3]),  # y0
        np.mean(y[-3:]),  # y1
        0,               # a  
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
def gaus(x, A=1, mu=0, sigma=1):
    return(
        A * np.exp(-(np.array(x)-mu)**2 / (2*sigma)**2)
    )
def f_p0_gaus(x, y):
    return(max(y), x[np.argmax(y)], (x[1]-x[0])/5)
    
    
gauss = fit_function(
    f = gaus,
    f_p0 = f_p0_gaus,
    description = "gauss function without constant term",
    parameters = ["A", "mu", "sigma"],
    parameters_tex = ["A", "\\mu", "\\sigma"],
    formula = "A exp(-(x-mu)^2 /(2 sigma)^2)",
    formula_tex = "$A \\exp{{\\frac{{-(x-\\mu)^2}}{{(2 \\sigma)^2}}}}$",
)

# doublegauss function
def gaus2(x, A1=1, mu1=0, sigma1=1, A2=1, mu2=0, sigma2=1):
    return(
          A1 * np.exp(-(np.array(x)-mu1)**2 / (2*sigma1)**2)
        + A2 * np.exp(-(np.array(x)-mu2)**2 / (2*sigma2)**2)
    )
def f_p0_gaus2(x, y):
    A1 = max(y)
    A2 = A1 / 10
    mu = x[np.argmax(y)]
    sigma1 = (x[1]-x[0])/5
    sigma2 = sigma1*10
    
    return(A1, mu, sigma1, A2, mu, sigma2)
    
    
gauss2 = fit_function(
    f = gaus2,
    f_p0 = f_p0_gaus2,
    description = "sum of two gaus functions without a constant term",
    parameters = ["A1", "mu1", "sigma1", "A2", "mu2", "sigma2"],
    parameters_tex = ["A_1", "\\mu_1", "\\sigma_1", "A_2", "\\mu_2", "\\sigma_2"],
    formula = "A1 exp(-(x-mu1)^2 /(2 sigma1)^2) + A2 exp(-(x-mu2)^2 /(2 sigma2)^2)",
    formula_tex = "$A_1 \\exp{{\\frac{{-(x-\\mu_1)^2}}{{(2 \\sigma_1)^2}}}} + A_2 \\exp{{\\frac{{-(x-\\mu_2)^2}}{{(2 \\sigma_2)^2}}}}$",
)

# gauss function
def gaussC(x, A=1, mu=0, sigma=1, C = 0):
    return(
        A * np.exp(-(np.array(x)-mu)**2 / (2*sigma)**2) + C
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
    formula_tex = "$A \\exp{{\\frac{{-(x-\\mu)^2}}{{(2 \\sigma)^2}}}}+C$",
)



# gauss function
def f_expgauss(x, A, tau, C=0, B=1, mu=0, sigma=1):
    return(
        A*np.exp(-x/tau) + B * np.exp(-(np.array(x)-mu)**2 / (2*sigma)**2) + C
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
    formula_tex = "$A \\cdot \\exp{{( t / \\tau)}} + B \\exp{{\\frac{{-(t-\\mu)^2}}{{(2 \\sigma)^2}}}} + C$",
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
